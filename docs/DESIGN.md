# Reverie Design Document

*Adaptive, curriculum-free reasoning in a continuous latent space. A Coconut successor, in JAX (+ Rust for data).*

Status: design + reference for the implementation in `reverie/`. Codename **Reverie**: wordless thought. Everything below is decisive; alternatives appear only as named ablations or risk fallbacks.

> **Implementation status (read this first).** What ships today in `reverie/` is **output-space** trajectory distillation + depth-supervised halting, not the full dual-pass design below. Each continuous thought `y_j` is supervised, through the *tied* LM head, to decode to its gold reasoning step's concept token (`L_traj` in `reverie/latent.py`). That one param-free term both distills the trajectory and makes latents linearly decodable (the interpretability probe of §2.3-E). The richer **hidden-space** dual-pass of §2.1–2.3 (second teacher-forced pass → per-step hidden targets `t_j` for `L_distill` + `L_explicit`) is **planned**, not implemented. Treat §2.1 dual-mode and hidden-space losses as the roadmap; the trained objective that actually runs is:
> `L = Σₙ pₙ·CE(answer, W yₙ) + α·Σⱼ CE(k_j, W y_j) + γ·(−log p_m) + β·KL(p‖Geom(λ_p))`,
> i.e. `{L_answer, L_traj(=distill/probe fused), L_depth, L_ponder}`. Novelty pillars (§3) hold for this instantiation. Metrics are **candidate-restricted** binary accuracy (chance 0.5), not full-vocab exact match. Inference Pareto is a `halt_bias` sweep on the halt logit (§2.5), not a β/λ_prior dial at train time.

---

## 1. Thesis

Coconut reasons in continuous latent space by feeding the last-layer hidden state back as the next input embedding, but pays three costs:

1. **Brittle multi-stage curriculum.** Language steps become latent slots one stage at a time, with optimizer resets. Adding several latents at once spikes the loss; SIM-CoT reports reasoning collapse to ~12.5% when latents scale.
2. **No thought supervision.** Only the final answer gives gradient, so the latent trajectory is opaque.
3. **Fixed latent budget.** Every problem pays for the deepest problem's serial compute.

**Reverie replaces all three with one loss.** We distill a compressed discrete-CoT trajectory step-by-step into a variable-length continuous-thought chain, and pick chain length with a differentiable PonderNet-style halt targeted at the teacher's per-instance reasoning depth. One stage, no RL, no post-hoc classifier. One trained model traces an accuracy-vs-latent-compute Pareto frontier, and because the halt is depth-supervised, the spent budget is calibrated to problem difficulty.

---

## 2. Method

### 2.0 Notation and substrate

Decoder-only transformer `f_θ` (`reverie/model.py`) maps input embeddings `E ∈ R^{L×d}` to post-final-norm hidden states `H ∈ R^{L×d}`, with tied LM head `W = tok_emb` giving `logits = H Wᵀ`. Batch with `vmap`. The Coconut mechanism we reuse: **a latent thought is a last-layer hidden state fed back, unprojected, as the next input embedding.** No projection between hidden and embedding space.

Per training instance the data generator (`data-gen/`) gives, free and verified:

- **question** `q` (shuffled fact bag + binary "Is E a C⁺ or C⁻?" query)
- **gold trajectory** `s = (s_1, …, s_m)`: membership fact then the chain of `Every A is a B.` edges along the gold path
- **per-instance depth** `m = n_hops`
- **answer** `y` and, per step, key concept token `k_j` (the `B` in `s_j`)

`m` and `k_j` are the two supervision signals prior latent methods lack. Reverie uses both.

### 2.1 Two modes of one model (single stage, curriculum-free)

Same weights `θ`, two input regimes in one training step:

- **Explicit mode (teacher).** Teacher-force `[q, <bot>, s_1, …, s_m, <eot>, ###, y]`. Produces (a) ordinary CoT LM loss `L_explicit`, and (b) step-summary targets `t_j := H[end(s_j)]` under stop-gradient. Shared `θ` → self-distillation (à la CODI): no second model, no frozen checkpoint, no staging.
- **Latent mode (student).** From the prompt up to `<bot>`, run the Coconut recurrence for up to `N` steps. Continuous thoughts `z_1, …, z_N` plus a scalar halt logit per step. Student is pulled toward the teacher's trajectory (content) and depth `m` (length), and reads out the answer under a halting distribution.

Both losses apply from step 1. No curriculum: no stage-replaced tokens, no optimizer resets. `L_explicit` keeps teacher mode sharp; stop-grad on `t_j` stops targets collapsing to the student.

### 2.2 The latent recurrence (static-shape, JAX)

Coconut grows the sequence by one latent per step. For static shapes (one XLA compile, `lax.scan`), pre-allocate a buffer of length `P + N` and fill one latent slot per step under a causal + validity mask:

```
E ← zeros[P+N, d];  E[:P] ← embed(prompt)          # prompt incl. <bot>
carry₀ = (E, cum_p=0, acc=0, mask_valid=[1]*P + [0]*N)

def step(carry, n):                                 # n = 0..N-1  (static length N)
    E, cum_p, acc, valid = carry
    H          = f_θ(E, positions, mask=causal & valid)
    z          = H[P+n-1] if n>0 else H[P-1]
    E          = E.at[P+n].set(z)
    valid      = valid.at[P+n].set(1)
    λ          = sigmoid(w_h · z + b_h)
    p_step     = λ * (1 - cum_p)                    # PonderNet: halt exactly now
    acc        = acc + p_step * z
    cum_p      = cum_p + p_step
    return (E, cum_p, acc, valid), (z, λ, p_step)

(_, (Z, Λ, P_step)) = lax.scan(remat(step), carry₀, arange(N))   # Z: [N, d]
```

`Z = (z_1,…,z_N)` and `p_n = λ_n ∏_{j<n}(1−λ_j)` (one `cumprod` over `Λ`, `λ_N := 1` so `Σ p_n = 1`). `N` (= `MAX_STEPS`) is **static**; effective depth varies by masking and the halt, never by loop length. `remat` on the step bounds activation memory to O(1) in `N`. One shared differentiable unroll: heavy cost is `N` backbone forwards, independent of how many depths we later score.

### 2.3 The four losses

Let `D(a,b)` be cosine distance on layer-normalized vectors, `D(a,b) = ‖ā − b̄‖²` with `ā = a/‖a‖`. (Magnitude rides the residual stream; CCoT uses scaled-MSE, CODI uses normalized L1; cosine is the default.)

**(A) Trajectory distillation (content, per step).**

```
L_distill = (1/m) Σ_{j=1}^{m} D( z_j , sg(t_j) )
```

Latent `z_j` aligns to teacher step `j` for the first `m` latents; later latents have no distill target (the halt is pushed to stop at `m`). Full-trajectory alignment: every latent is supervised. CODI aligns one anchor; Coconut aligns nothing.

**(B) Adaptive-halt answer read-out (PonderNet over depth).**

Score the answer at each candidate depth, weight by the halt distribution:

```
ℓ_n = − log p_θ( y | q, z_1..z_n, <eot> )
L_answer = Σ_{n=1}^{N} p_n · ℓ_n
```

Weight losses, not states (PonderNet, not ACT): latents stay intact, never mean-field-averaged. Cheap via §2.4: one batched teacher-forced decode with `n` on the batch axis.

**(C) Depth supervision (the novel halt target).** Vanilla PonderNet only regularizes toward a global geometric mean. We pin the halt, per instance, to teacher chain length `m`:

```
L_depth = − log p_m           (m = n_hops)
```

This is the crux: continuous thought count is supervised to track the teacher's per-instance depth, differentiably, with no RL and no separate classifier. One MLE term against a data-given target.

**(D) Halting prior (anti-collapse + training compute prior).** KL to a truncated geometric prior. Shipped code (`geometric_prior` in `reverie/latent.py`) is **0-indexed** over depths `m ∈ {0..K}`:

```
p_G(m) ∝ λ_p (1−λ_p)^m ,  m = 0..K,  then renormalize
L_ponder = KL(p ‖ p_G)
```

Untruncated mean of this 0-indexed geometric is `(1−λ_p)/λ_p` (not the 1-indexed PonderNet mean `1/λ_p`). With `L_depth` doing per-instance supervision, `L_ponder` (i) prevents collapse to a single depth, (ii) gently biases native depth during training, (iii) is **not** the inference Pareto dial. Inference Pareto is a halt-logit bias sweep on one trained model (§2.5).

**Optional (E) Decodability probe (`δ`).** Linear probe `V` recovers each step's key concept from its latent:

```
L_probe = (1/m) Σ_{j=1}^{m} − log softmax(V z_j)[k_j]
```

Makes latents linearly decodable and yields a first-class interpretability metric.

**Total objective (one backward pass):**

```
L = L_answer + α·L_distill + γ·L_depth + β·L_ponder + η·L_explicit + δ·L_probe
```

Design defaults (roadmap dual-pass): `α=1.0, γ=0.5, β=0.01, η=1.0, δ=0.1`, `λ_p ≈ 0.15`, `N = 8`. Shipped `ReverieConfig` defaults differ slightly (`γ=1.0`, `λ_p=0.2`, `max_steps=6`) and omit `η`/`δ` (no explicit-mode pass, probe fused into `L_traj`). Core shipped = `{answer, traj, depth, ponder}`. Ablations: *no-distillation* drops `α`; *no-halting* drops `γ+β`, fixes depth.

For math with `c > 1` latents per step (GSM8K): map latents to steps in blocks, supervise the block's last latent to `t_j`, set depth target to `c·m`. Lead task is ProsQA at `c = 1`.

### 2.4 Efficient JAX computation

Three structural facts keep this cheap, all in one differentiable graph:

1. **One teacher forward, one student unroll.** Explicit mode: one teacher-forced pass (`L_explicit` + all `t_j`, stop-gradded). Latent mode: the §2.2 `lax.scan` (`N` backbone steps, remat-bounded). Cost is `N` steps, independent of how many depths we score.
2. **Batched-over-depth answer scoring.** Gold answer `y` is fixed: put the `N` candidate depths on the batch axis. One decoder pass over a `[N, P+N+A]` buffer; row `n` exposes `prompt + z_{1..n} + <eot> + y` via a validity mask. All `ℓ_n` in parallel, not `N` sequential decodes.
3. **Everything else is elementwise.** `p_n` is one `cumprod`; the other losses are closed-form sums. Per-example `m` is handled by padding + masking.

Result: one teacher forward + one student unroll + one batched read-out + masked sums. End-to-end differentiable (no REINFORCE, no per-depth re-decode). Train step is `eqx.filter_jit(donate="all")` with f32 master weights, bf16 compute on GPU only.

### 2.5 Inference and the Pareto knob

At inference use `lax.while_loop` and pay actual depth: roll latents, stop when cumulative halt mass crosses the budget,

```
N* = min{ n : Σ_{j≤n} p_j > 1 − ε }
```

then emit `<eot>` and greedily decode from `z_{N*}`. Shipped eval (`scripts/run.py`) sweeps **`halt_bias`** added to the halt logit before the sigmoid (positive → halt earlier). That is the single-model inference Pareto dial; β·KL / `λ_prior` stay fixed from training.

> **Empirical note.** With teacher-depth supervision the halt is so confident (λ→≈1 exactly at `n_hops`) that bias sweeps are **flat**. The operating point is an exact per-instance decision, not a tunable threshold. A smooth frontier needs a softer halt (temperature on λ, or `λ_prior` swept across *retraining* runs). Honest counterpart to near-perfect calibration (steps = `n_hops`, ρ = +1.00); see `docs/paper.md` §5.3.

### 2.6 Theory: adaptive serial depth is necessary

Let instances have required serial depth `d ∼ P(d)` (heterogeneous; our generator dials `P(d)`). A latent thought adds one *serial* step (each `z_n` attends to `z_{<n}`), unlike filler tokens (parallel width only, TC⁰-bounded).

**Proposition (informal).** On the serial fragment (instances needing `d` mutually-dependent deductions), a latent model with `n < d` thoughts cannot represent the computation. A **fixed**-depth model with budget `N` is correct on the deep tail only if `N ≥ max supp(d)`, and then spends `max supp(d)` serial steps on every instance. A **per-instance adaptive** model with `n = d` is correct everywhere at expected cost `E[d] ≤ max supp(d)`, with strict inequality when `P(d)` is non-degenerate. Adaptive latent depth is necessary to match explicit CoT at `E[d]` compute on heterogeneous-depth distributions, and sufficient on the sub-TC⁰-serial fragment our tasks live in.

*Proof sketch.* An `n`-step latent chain composes `n` attention/MLP updates; a `d`-hop reachability query where hop `j+1`'s fact is unknown until hop `j` is resolved needs ≥ `d` dependent updates, so `n ≥ d`. Cost gap: `E[d] < max supp(d)` for any non-point `P(d)`. Empirically (§4): a fixed-`N` model wastes compute on shallow instances or fails the tail; Reverie matches accuracy at lower mean depth, and steps track `k`.

Scoped as a proposition with empirical validation, not a general theorem.

---

## 3. Novelty

### 3.1 Differentiation table

| Method | Supervised latents? | Adaptive per-instance length? | Single-stage? | vs Reverie |
|---|---|---|---|---|
| **Coconut** (Hao 2024) | No (answer only) | No (fixed, padded) | No (multi-stage curriculum) | No distillation; brittle staging; fixed depth |
| **CCoT** (Cheng & Van Durme 2024) | Yes (teacher hidden subset) | Partial: separate `end_ψ` on fixed ratio | No (multi-stage) | Bolted-on halt, not a differentiable prior; no depth supervision |
| **ICoT-KD / SI** (Deng 2023/2024) | KD: teacher hidden; SI: none | No | KD: no (3 models); SI: curriculum | Not adaptive latent count |
| **Quiet-STaR** (Zelikman 2024) | Reward only, discrete text | No | Pretraining | Discrete; high-variance RL |
| **PonderNet** (Banino 2021) | No (task loss) | Yes (diff. geometric halt) | Yes | No content supervision, no teacher-depth target |
| **CODI** (Shen 2025) | Yes (single anchor) | No (fixed 6) | Yes | One anchor ≠ trajectory; fixed length; no halt |
| **Learning-When-to-Stop** (2511.21581) | No (answer reward) | Yes (RL/PPO) | Bolted on Coconut | RL (high variance); no distillation |
| **Reverie** | **Every latent ← teacher step (+ linear probe)** | **Diff. halt supervised by teacher depth `m`** | **One joint stage, no curriculum, no RL** | **Fused objective + calibrated compute + serial-depth theory** |

### 3.2 What is new

No single ingredient is new: continuous thoughts (Coconut), trajectory distillation (CCoT), single-stage self-distillation (CODI), differentiable halting (PonderNet), per-instance latent halt (2511.21581 via RL). **Unclaimed is the fusion:** one stage that distills the full teacher trajectory into every latent **and** sets chain length with a differentiable geometric-prior halt targeted at teacher depth. No RL, no post-hoc classifier, no staging.

Seams we occupy:

- vs **CODI**: single-anchor → full trajectory; fixed-6 → adaptive, depth-supervised.
- vs **CCoT**: multi-stage + separate `end_ψ` → single-stage + differentiable prior halt by depth.
- vs **2511.21581**: RL/PPO halt → differentiable, distillation-native halt (lower variance; target from teacher, not sparse reward).
- vs **Coconut/PonderNet**: adds the two supervision signals each lacks (per-step content; per-instance length).

### 3.3 The claim

> A single curriculum-free, RL-free model that matches explicit chain-of-thought on a difficulty-calibrated accuracy-vs-latent-compute Pareto frontier, by distilling a discrete reasoning trajectory into a variable-length continuous-thought chain whose length is a differentiable function supervised by the teacher's per-instance depth, with a proposition that per-instance adaptive latent depth is necessary to do so on distributions with heterogeneous serial-reasoning depth.

Three pillars: fused objective (§2.3), single-model Pareto + difficulty calibration (§2.5, §4), variable-serial-depth theory (§2.6). We do not frame as "adaptive halting" or "single-stage distillation" alone; both are taken. Cite CODI, CCoT, and 2511.21581 as closest prior and differentiate on the fusion.

*(Fallback if the fused-objective seam is too narrow: lead with pillars 2+3 as "difficulty-calibrated latent compute." That lane is unoccupied regardless of objective novelty.)*

---

## 4. Experiments

### 4.1 Lead, control, debug, capstone

- **Lead: self-generated ProsQA** (`data-gen/`, fictional tokens, from-scratch). No external corpus, no pretrained weights, no leakage. Difficulty is a dial. Least-confounded latent-planning test.
- **Control: linear chains.** Expect the Reverie−CoT gap collapses to ≈0 (nothing to search). Feature of the argument, not a failure. Use `data-gen --branch 0 --trap-depth 0`.
- **Debug: micro-ProsQA** (k-hop letter-graph, ≤8 nodes, ~20-token vocab). CPU, minutes. CI gate before GPU spend.
- **Capstone (optional, GPU): GSM8K**, pretrained GPT-2-124M, iCoT-augmented, `c=2`. Stretch, not a gate.

### 4.2 Baselines and ablations

Baselines (matched compute): **No-CoT**, **CoT**, **Coconut** (faithful reimpl), **Reverie**. Ablations: no-distillation, no-halting, no-probe, optional RL-halt swap (PPO stop-policy à la 2511.21581).

### 4.3 Metrics

1. **Candidate-restricted accuracy.** ProsQA is binary C₁/C₂; score which of the two candidates the read-out prefers (chance 0.5). Stratified by hop count `k`.
2. **Latent-steps-used.** Mean `N*` at inference; per-hop mean.
3. **Accuracy-vs-latent-compute Pareto.** Sweep halt-logit bias (confident halt makes ε flat).
4. **Difficulty calibration.** Spearman `ρ(N*, k)` using ground-truth `n_hops`; mean `N*` vs `k` curve.
5. **Seed stability.** 3–5 seeds; mean±std; count of collapsed runs.
6. **Interpretability.** Linear-probe decode accuracy of latents → key concept tokens.

### 4.4 Scaling story

1. **Depth sweep.** Test at `k = 2..6` (train fixed); plot Reverie−CoT gap vs `k`. Expect ≈0 at `k=2`, widening with depth.
2. **Branching sweep.** Hold `k`, raise `branch`/`trap_depth`; CoT should degrade faster.
3. **Model-size sweep.** `d_model ∈ {128,256,384}`.
4. **Control.** ProntoQA-5hop: gap collapses.
5. **Depth-variance benchmark.** Train/test mix over `k∈{2..6}`; fixed-`N` must waste compute or fail the tail; Reverie matches accuracy at lower mean depth.

### 4.5 Compute plan

- **Phase 0 (CPU, minutes, gate).** micro-ProsQA; 2 layers, `d_model=128`; ~3k steps. What `make phase0` reproduces today: depth calibration (ρ≈1, steps=`n_hops`) + γ/α ablations under **candidate-restricted** acc. Do not gate on full-vocab EM or Coconut>CoT>No-CoT accuracy order (shortcuts dominate at this scale; see `docs/paper.md` §6).
- **Phase 1 (single GPU, ~30–90 min, headline).** From-scratch, 6–8 layers, `d_model=256`, ~10–20M params. Data: train 20k / val 500 / test 500 (hold out by fresh seeds). `c=1`, `MAX_STEPS=8`, lr `1e-4`, batch 128, AdamW, warmup-cosine. ~3–5k steps. All four baselines at identical compute; 3–5 seeds.
- **Phase 2 (sweeps).** Halt-bias Pareto and calibration need no retrain; depth/branching/size sweeps are minutes each.
- **Phase 3 (optional).** GSM8K if a GPU is free.

Expected headline (aspirational at scale; Phase-0 does **not** claim an accuracy edge): Reverie matches CoT where multi-step reasoning is required, at fewer and adaptive latent steps; halt-bias frontier vs Coconut's fixed point; `N*` correlates with `k`; latents linearly decodable; gap widens with depth heterogeneity when shortcuts are closed.

---

## 5. Repo architecture

Shipped layout (flat package; no separate heads/curriculum/optim/eval modules):

```
reverie/
├── README.md, Makefile, pyproject.toml, LICENSE
├── docs/
│   ├── DESIGN.md
│   └── paper.md
├── reverie/                     # JAX/Equinox package
│   ├── __init__.py
│   ├── model.py                 # Transformer, ModelConfig
│   ├── latent.py                # static-shape scan, halt head, objective
│   ├── data.py                  # vocab, render, collate
│   └── train.py                 # train loop + evaluate
├── scripts/
│   ├── run.py                   # single-method train+eval
│   ├── matrix.py                # multi-method comparison
│   ├── report.py / ablation_table.py
│   ├── phase0.sh / phase0_deep.sh
├── data-gen/                    # Rust crate
│   ├── Cargo.toml
│   └── src/main.rs
└── tests/test_core.py
```

### 5.1 Rust generator interface

Rust owns the abstract, verified problem; Python owns token rendering. JSONL schema:

```json
{"id":int,"n_entities":int,"entities":["ger","scrom",...],"edges":[[a,b],...],
 "source":int,"candidates":[c0,c1],"answer":int,"gold_path":[n0,...,nk],
 "n_hops":int,"n_distractors":int}
```

- `gold_path` → teacher trajectory `s`; `len(gold_path)-1 = n_hops = m` (halt target + calibration ground truth).
- `edges/entities` → shuffled fact bag; `source/candidates` → query; `answer` → `C⁺`.

CLI (shipped): `reverie-datagen --n N --seed S --hops H --branch B --trap-depth D [--connect C] [--out FILE]`. Task variants beyond ProsQA-style DAGs (ProntoQA, micro presets) are planned.

Two refinements: (1) per-example sub-streams `rng_i = SplitMix64(global_seed ^ 0x9E3779B97F4A7C15·i)` so any example is regenerable in isolation; (2) golden-file determinism test for byte-identical corpora across platforms.

### 5.2 Notes on the existing model

`model.py` returns `(logits, hidden)` and exposes `backbone(x, positions)` over input embeddings: the latent hook. `latent.py` calls `backbone`, not `__call__`. Keep the Python list of blocks (fine at 6–8 layers); switch to scan-over-layers + remat only past ~16 layers. Weight tying on; RoPE/RMSNorm/SwiGLU as implemented. Reproducibility: data order and RNG are pure functions of `(seed, step)`; run config is a frozen dataclass written into each `runs/*.json`. Checkpointing is not wired yet (params stay in memory for the run; Orbax or equivalent is planned).

---

## 6. Risks, mitigations, and build order

### 6.1 Risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Self-distillation collapse | Stop-grad on `t_j`; keep `η·L_explicit` strong; cosine distance. Fallback: EMA-teacher or frozen 1-epoch CoT checkpoint. |
| 2 | Halt collapses to `N*=1` or `N*=N` | `β·L_ponder` anti-collapse; `γ·L_depth` pins per-instance target; monitor depth histogram. |
| 3 | Distillation vs answer tension | Cosine distance; warm up `α` over first few hundred steps (loss-weight ramp, not a data curriculum); ablate `α`. |
| 4 | Rigid positional alignment when optimal latent count ≠ `m` | Halt provides slack; `c>1` block mapping. Fallback: DTW soft alignment. |
| 5 | From-scratch model too weak for latent BFS | Phase-0 micro gate; `d_model` sweep; task designed learnable at ~10–20M. |
| 6 | JAX recompilation / scan memory | Static `MAX_STEPS` + masking; remat per step; batched-over-depth read-out; donate all. |
| 7 | Novelty challenge (CODI/CCoT/2511.21581) | Lead with fused objective + Pareto + theory; ship RL-halt ablation; §3.3 fallback. |
| 8 | Rust/Python determinism drift | Per-example sub-stream seeding; golden-file test in CI; record seed+params+version. |
| 9 | GSM8K over-scope | Strictly optional; paper stands on ProsQA + control + micro. |

### 6.2 Build order (five gated steps)

Historical / planned gates. Phase-0 calibration that `make phase0` reproduces today is the output-space + depth-supervised objective on micro ProsQA (see `docs/paper.md`), not every gate below.

1. **Data + harness.** Refine `data-gen/` (sub-streams, lib, prontoqa, micro, tests); build `data.py` + tokenizer. Gate: golden-file + BFS-verify green; rendered ProsQA matches serialization.
2. **Backbone + non-latent baselines (CPU micro).** Wire train/eval; No-CoT and CoT SFT; overfit micro-ProsQA. Gate: candidate-restricted acc high; CoT>No-CoT where the task requires reasoning.
3. **Latent mechanism + Coconut reimpl.** Static-shape scan + halt head; reproduce Coconut baseline. Gate: latent-grad and halt tests green.
4. **Reverie objective.** Output-space traj distill + PonderNet answer + depth + ponder, single stage (hidden-space dual-pass later). Gate: depth histogram tracks `m`; stable across seeds.
5. **Scaling + Pareto + theory eval.** Halt-bias Pareto, calibration ρ, sweeps, ProntoQA control, seed-stability table. Gate: gap widens with `k` when shortcuts are closed; ρ strong-positive; control gap collapses.

---

### Anchor references

Coconut (arXiv:2412.06769) · PonderNet (2107.05407) · CODI (2502.21074) · CCoT (2412.13171) · ICoT-KD (2311.01460) / Stepwise-Internalization (2405.14838) · Quiet-STaR (2403.09629) · Learning-When-to-Stop (2511.21581) · SIM-CoT latent-collapse (2509.20317) · Filler-tokens TC⁰ bound (2404.15758) · ProntoQA (2210.01240) · GSM8K (2110.14168). Stack: Equinox + Optax (Orbax checkpointing planned); determinism after Levanter (data/RNG as pure functions of `(seed, step)`).
