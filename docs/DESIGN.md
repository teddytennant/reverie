# Reverie Design Document

*Adaptive, curriculum-free reasoning in a continuous latent space. A Coconut successor, in JAX (+ Rust for data).*

Design notes and reference for the implementation in `reverie/`. Codename **Reverie**: wordless thought.

## 1. Thesis

Coconut reasons in continuous latent space by feeding the last-layer hidden state back as the next input embedding. It works, and pays three costs:

1. A brittle multi-stage curriculum. Language steps become latent slots one stage at a time, with optimizer resets. Adding several latents at once spikes the loss; SIM-CoT reports reasoning collapse to ~12.5% when latents scale.
2. No thought supervision. Only the final answer gives gradient, so the latent trajectory is opaque.
3. A fixed latent budget. Every problem pays for the deepest problem's serial compute.

Reverie replaces all three with one loss. I distill a compressed discrete-CoT trajectory step by step into a variable-length continuous-thought chain, and pick chain length with a differentiable PonderNet-style halt targeted at the teacher's per-instance reasoning depth. One stage, no RL, no post-hoc classifier. One trained model traces an accuracy-vs-latent-compute Pareto frontier, and because the halt is depth-supervised, the spent budget is calibrated to problem difficulty.

---

## 2. Method

### 2.0 Notation and substrate

Decoder-only transformer `f_θ` (`reverie/model.py`) maps input embeddings `E ∈ R^{L×d}` to post-final-norm hidden states `H ∈ R^{L×d}`, with tied LM head `W = tok_emb` giving `logits = H Wᵀ`. Batch with `vmap`. The Coconut mechanism I reuse: a latent thought is a last-layer hidden state fed back, unprojected, as the next input embedding. No projection between hidden and embedding space.

Per training instance the generator (`data-gen/`) hands over, verified:

- question `q`: a shuffled fact bag plus the binary "Is E a C⁺ or C⁻?" query
- gold trajectory `s = (s_1, …, s_m)`: the membership fact, then the chain of `Every A is a B.` edges along the gold path
- per-instance depth `m = n_hops`
- answer `y`, and per step the key concept token `k_j` (the `B` in `s_j`)

`m` and `k_j` are the two supervision signals prior latent methods don't have. Reverie uses both.

### 2.1 Two modes of one model (single stage, curriculum-free)

Same weights `θ`, two input regimes in one training step.

Explicit mode (teacher) teacher-forces `[q, <bot>, s_1, …, s_m, <eot>, ###, y]`. It produces an ordinary CoT LM loss `L_explicit` and step-summary targets `t_j := H[end(s_j)]` under stop-gradient. Shared `θ` means self-distillation in the CODI sense: no second model, no frozen checkpoint, no staging.

Latent mode (student) starts from the prompt up to `<bot>` and runs the Coconut recurrence for up to `N` steps, giving continuous thoughts `z_1, …, z_N` plus a scalar halt logit per step. The student is pulled toward the teacher's trajectory for content and toward depth `m` for length, and reads out the answer under a halting distribution.

Both losses apply from step 1. No stage-replaced tokens, no optimizer resets. Stop-grad on `t_j` is what keeps the targets from collapsing onto the student.

### 2.2 The latent recurrence (static-shape, JAX)

Coconut grows the sequence by one latent per step. For static shapes (one XLA compile, `lax.scan`) I pre-allocate a buffer of length `P + N` and fill one latent slot per step under a causal-and-validity mask:

```
E ← zeros[P+N, d];  E[:P] ← embed(prompt)          # prompt incl. <bot>
carry₀ = (E, cum_p=0, mask_valid=[1]*P + [0]*N)

def step(carry, n):                                 # n = 0..N-1  (static length N)
    E, cum_p, valid = carry
    H          = f_θ(E, positions, mask=causal & valid)
    z          = H[P+n-1] if n>0 else H[P-1]
    E          = E.at[P+n].set(z)
    valid      = valid.at[P+n].set(1)
    λ          = sigmoid(w_h · z + b_h)
    p_step     = λ * (1 - cum_p)                    # PonderNet: halt exactly now
    cum_p      = cum_p + p_step
    return (E, cum_p, valid), (z, λ, p_step)

(_, (Z, Λ, P_step)) = lax.scan(remat(step), carry₀, arange(N))   # Z: [N, d]
```

`Z = (z_1,…,z_N)` and `p_n = λ_n ∏_{j<n}(1−λ_j)`, one `cumprod` over `Λ` with `λ_N := 1` so `Σ p_n = 1`. `N` (= `MAX_STEPS`) is static; effective depth varies by masking and by the halt, never by loop length. `remat` on the step bounds activation memory to O(1) in `N`. The heavy cost is `N` backbone forwards, and it doesn't grow with how many depths I later score.

### 2.3 The losses

Let `D(a,b)` be cosine distance on layer-normalized vectors, `D(a,b) = ‖ā − b̄‖²` with `ā = a/‖a‖`. Magnitude rides the residual stream, so I don't want it in the metric; CCoT uses scaled-MSE and CODI uses normalized L1, cosine is the default here.

(A) Trajectory distillation, content, per step:

```
L_distill = (1/m) Σ_{j=1}^{m} D( z_j , sg(t_j) )
```

Latent `z_j` aligns to teacher step `j` for the first `m` latents; later latents have no distill target, since the halt is pushed to stop at `m`. Every latent is supervised. CODI aligns one anchor, Coconut aligns nothing.

(B) Adaptive-halt answer read-out, PonderNet over depth. Score the answer at each candidate depth and weight by the halt distribution:

```
ℓ_n = − log p_θ( y | q, z_1..z_n, <eot> )
L_answer = Σ_{n=1}^{N} p_n · ℓ_n
```

Weight the losses, not the states. That's PonderNet rather than ACT, and it matters: the latents stay intact instead of being mean-field-averaged into one blurred vector. Cheap, because of the batched read-out below.

(C) Depth supervision. Vanilla PonderNet only regularizes toward a global geometric mean. I pin the halt, per instance, to the teacher's chain length `m`:

```
L_depth = − log p_m           (m = n_hops)
```

One MLE term against a target the data already contains. That is what produces calibrated compute: thought count tracks per-instance depth, differentiably, with no RL and no separate classifier.

(D) Halting prior: anti-collapse, plus a compute prior during training. KL to a truncated geometric. The prior is 0-indexed over depths `m ∈ {0..K}`:

```
p_G(m) ∝ λ_p (1−λ_p)^m ,  m = 0..K,  then renormalize
L_ponder = KL(p ‖ p_G)
```

Watch the indexing. The untruncated mean of this 0-indexed geometric is `(1−λ_p)/λ_p`, not PonderNet's 1-indexed `1/λ_p`. With `L_depth` doing per-instance supervision, `L_ponder` prevents collapse to a single depth and gently biases native depth during training. It is not the inference Pareto dial; see §2.5.

(E) Decodability probe `δ`, optional. A linear probe `V` recovers each step's key concept from its latent:

```
L_probe = (1/m) Σ_{j=1}^{m} − log softmax(V z_j)[k_j]
```

This makes the latents linearly decodable and gives a first-class interpretability metric.

Total objective, one backward pass:

```
L = L_answer + α·L_distill + γ·L_depth + β·L_ponder + η·L_explicit + δ·L_probe
```

Design defaults: `α=1.0, γ=0.5, β=0.01, η=1.0, δ=0.1`, `λ_p ≈ 0.15`, `N = 8`. Ablations: *no-distillation* drops `α`, *no-halting* drops `γ+β` and fixes depth.

For math with `c > 1` latents per step (GSM8K), map latents to steps in blocks, supervise the block's last latent to `t_j`, and set the depth target to `c·m`. The lead task is ProsQA at `c = 1`.

### 2.4 Efficient JAX computation

Three structural facts keep this cheap, all in one differentiable graph.

One teacher forward and one student unroll. The teacher pass gives `L_explicit` and all the stop-gradded `t_j`; the student is the §2.2 scan, `N` backbone steps, remat-bounded.

Batched-over-depth answer scoring. The gold answer `y` is fixed, so I put the `N` candidate depths on the batch axis: one decoder pass over a `[N, P+N+A]` buffer, where row `n` exposes `prompt + z_{1..n} + <eot> + y` through a validity mask. All `ℓ_n` in parallel instead of `N` sequential decodes. On ProsQA the answer is a single concept token, so it collapses further to one batched matmul of the tied head over `[K+1, d]`.

Everything else is elementwise: `p_n` is one `cumprod`, the rest are closed-form sums, per-example `m` is padding plus masking.

End to end differentiable, no REINFORCE, no per-depth re-decode. Train step is `eqx.filter_jit(donate="all")`, f32 master weights, bf16 compute on GPU only.

### 2.5 Inference and the Pareto knob

Stop when cumulative halt mass crosses the budget:

```
N* = min{ n : Σ_{j≤n} p_j > 1 − ε }
```

then emit `<eot>` and greedily decode from `z_{N*}`. At inference this should use a `lax.while_loop` and pay actual depth rather than every slot.

The Pareto dial is `halt_bias`, added to the halt logit before the sigmoid, positive halts earlier. `scripts/run.py` sweeps it over `[4, 2, 1, 0, −1, −2, −4]` on the one trained model. β·KL and `λ_prior` stay fixed from training.

The sweep is flat, and I should say so plainly. With teacher-depth supervision the halt gets so confident (λ jumps to about 1 exactly at `n_hops`) that biasing the logit by ±4 doesn't move the operating point. What I have is an exact per-instance decision, not a tunable threshold. A smooth frontier would need a softer halt: a temperature on λ, or `λ_prior` swept across separate retraining runs. That's the honest counterpart to near-perfect calibration (steps = `n_hops`, ρ = +1.00); numbers in `docs/paper.md` §5.3.

### 2.6 Theory: adaptive serial depth is necessary

Let instances have required serial depth `d ∼ P(d)`, heterogeneous, with the generator dialing `P(d)`. A latent thought adds one *serial* step, since each `z_n` attends to `z_{<n}`. Filler tokens don't: they buy parallel width only, and stay TC⁰-bounded.

**Proposition (informal).** On the serial fragment, instances needing `d` mutually-dependent deductions, a latent model with `n < d` thoughts cannot represent the computation. A fixed-depth model with budget `N` is correct on the deep tail only if `N ≥ max supp(d)`, and then spends `max supp(d)` serial steps on every instance. A per-instance adaptive model with `n = d` is correct everywhere at expected cost `E[d] ≤ max supp(d)`, strict when `P(d)` is non-degenerate. So adaptive latent depth is necessary to match explicit CoT at `E[d]` compute on heterogeneous-depth distributions, and sufficient on the sub-TC⁰-serial fragment these tasks live in.

*Proof sketch.* An `n`-step latent chain composes `n` attention/MLP updates. A `d`-hop reachability query where hop `j+1`'s fact is unknown until hop `j` resolves needs at least `d` dependent updates, so `n ≥ d`. For the cost gap, `E[d] < max supp(d)` for any non-point `P(d)`. Empirically a fixed-`N` model either wastes compute on shallow instances or fails the tail, while Reverie matches accuracy at lower mean depth with steps tracking `k`.

This is a scoped proposition with empirical validation, not a general theorem.

---

## 3. Novelty

### 3.1 Differentiation table

| Method | Supervised latents? | Adaptive per-instance length? | Single-stage? | vs Reverie |
|---|---|---|---|---|
| Coconut (Hao 2024) | No (answer only) | No (fixed, padded) | No (multi-stage curriculum) | No distillation; brittle staging; fixed depth |
| CCoT (Cheng & Van Durme 2024) | Yes (teacher hidden subset) | Partial: separate `end_ψ` on fixed ratio | No (multi-stage) | Bolted-on halt, not a differentiable prior; no depth supervision |
| ICoT-KD / SI (Deng 2023/2024) | KD: teacher hidden; SI: none | No | KD: no (3 models); SI: curriculum | Not adaptive latent count |
| Quiet-STaR (Zelikman 2024) | Reward only, discrete text | No | Pretraining | Discrete; high-variance RL |
| PonderNet (Banino 2021) | No (task loss) | Yes (diff. geometric halt) | Yes | No content supervision, no teacher-depth target |
| CODI (Shen 2025) | Yes (single anchor) | No (fixed 6) | Yes | One anchor is not a trajectory; fixed length; no halt |
| Learning-When-to-Stop (2511.21581) | No (answer reward) | Yes (RL/PPO) | Bolted on Coconut | RL (high variance); no distillation |
| **Reverie** | Every latent supervised to its teacher step | Diff. halt supervised by teacher depth `m` | One joint stage, no curriculum, no RL | Fused objective + calibrated compute + serial-depth theory |

### 3.2 The fusion

No single ingredient here is mine: continuous thoughts are Coconut, trajectory distillation is CCoT, single-stage self-distillation is CODI, differentiable halting is PonderNet, per-instance latent halting exists in 2511.21581 via RL. What nobody has put together is one stage that distills the full teacher trajectory into every latent *and* sets chain length with a differentiable geometric-prior halt targeted at teacher depth, with no RL, no post-hoc classifier and no staging.

Against CODI: single anchor becomes the full trajectory, fixed-6 becomes adaptive and depth-supervised. Against CCoT: multi-stage plus a separate `end_ψ` classifier becomes one stage with a differentiable prior halt keyed on depth. Against 2511.21581: an RL/PPO halt becomes a distillation-native one, lower variance, target from the teacher instead of sparse reward. Coconut and PonderNet are each missing exactly one of the two supervision signals; I use both.

### 3.3 The claim

> A single curriculum-free, RL-free model that matches explicit chain-of-thought on a difficulty-calibrated accuracy-vs-latent-compute Pareto frontier, by distilling a discrete reasoning trajectory into a variable-length continuous-thought chain whose length is a differentiable function supervised by the teacher's per-instance depth, with a proposition that per-instance adaptive latent depth is necessary to do so on distributions with heterogeneous serial-reasoning depth.

Three pillars: the fused objective, single-model Pareto plus difficulty calibration, and the variable-serial-depth argument. I don't frame this as "adaptive halting" or "single-stage distillation" alone, because both are taken. CODI, CCoT and 2511.21581 are the closest prior work and the differentiation is on the fusion.

---

## 4. Experiments

### 4.1 Tasks

- Lead: self-generated ProsQA (`data-gen/`, fictional tokens, trained from scratch). No external corpus, no pretrained weights, no leakage, and difficulty is a dial.
- Control: linear chains, `data-gen --branch 0 --trap-depth 0`. I expect the Reverie-minus-CoT gap to collapse to about 0 because there's nothing to search. That's part of the argument, not a failure.
- Debug: micro-ProsQA, a k-hop letter graph, at most 8 nodes and a ~20-token vocab. CPU, minutes. The CI gate before spending GPU.
- Capstone, optional: GSM8K with a pretrained GPT-2-124M, iCoT-augmented, `c=2`. Not a gate.

### 4.2 Baselines and ablations

Baselines at matched compute: No-CoT, CoT, Coconut (faithful reimplementation), Reverie. Ablations: no-distillation, no-halting, no-probe, and optionally swapping in an RL halt (PPO stop-policy, as in 2511.21581).

### 4.3 Metrics

1. Candidate-restricted accuracy. ProsQA is binary C₁/C₂, so score which of the two candidates the read-out prefers (chance 0.5), stratified by hop count `k`.
2. Latent steps used: mean `N*` at inference, and per-hop mean.
3. Accuracy-vs-latent-compute Pareto, sweeping halt-logit bias.
4. Difficulty calibration: Spearman `ρ(N*, k)` against ground-truth `n_hops`, plus the mean `N*` vs `k` curve.
5. Seed stability: 3 to 5 seeds, mean±std, and a count of collapsed runs.
6. Interpretability: linear-probe decode accuracy of latents against key concept tokens.

### 4.4 Scaling story

Sweep depth at `k = 2..6` with training fixed and plot the Reverie-minus-CoT gap: about 0 at `k=2`, widening. Sweep branching by holding `k` and raising `branch` and `trap_depth`, where CoT should degrade faster. Sweep `d_model ∈ {128,256,384}`. Run ProntoQA-5hop as a second control, where the gap should collapse. The one that tests §2.6 is the depth-variance run: train and test on a mix over `k∈{2..6}`, where a fixed-`N` model has to either waste compute or fail the tail while Reverie matches accuracy at lower mean depth.

### 4.5 Compute plan

Phase 0 is the CPU gate, minutes: micro-ProsQA, 2 layers, `d_model=128`, ~3k steps. What `make phase0` reproduces today is depth calibration (ρ ≈ 1, steps = `n_hops`) plus the γ and α ablations under candidate-restricted accuracy. Don't gate on full-vocab EM or on a Coconut > CoT > No-CoT accuracy ordering; shortcuts dominate at this scale, see `docs/paper.md` §6.

Phase 1 is the headline, a single GPU for 30 to 90 minutes: from scratch, 6 to 8 layers, `d_model=256`, 10 to 20M params, data 20k train / 500 val / 500 test held out by fresh seeds, `c=1`, `MAX_STEPS=8`, lr `1e-4`, batch 128, AdamW, warmup-cosine, 3 to 5k steps, all four baselines at identical compute, 3 to 5 seeds. The halt-bias Pareto and calibration need no retraining after that, and the sweeps are minutes each. GSM8K only if a GPU is idle. What I'd hope to see there, and what Phase 0 does not claim, is Reverie matching CoT at fewer and adaptive latent steps once shortcuts are closed and the problems are deep enough to need more than one pass.

---

## 5. Repo architecture

Flat package, no separate heads/curriculum/optim/eval modules:

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

Rust owns the abstract, BFS-verified problem; Python owns token rendering. JSONL schema:

```json
{"id":int,"n_entities":int,"entities":["ger","scrom",...],"edges":[[a,b],...],
 "source":int,"candidates":[c0,c1],"answer":int,"gold_path":[n0,...,nk],
 "n_hops":int,"n_distractors":int}
```

`gold_path` becomes the teacher trajectory `s`, and `len(gold_path)-1 = n_hops = m` is both the halt target and the calibration ground truth. `edges` and `entities` render into the shuffled fact bag, `source` and `candidates` into the query, and `answer` marks `C⁺`.

CLI: `reverie-datagen --n N --seed S --hops H --branch B --trap-depth D [--out FILE]`. Task variants beyond ProsQA-style DAGs (ProntoQA, micro presets) are planned.

Two refinements I want next: per-example sub-streams, `rng_i = SplitMix64(global_seed ^ 0x9E3779B97F4A7C15·i)`, so any single example is regenerable in isolation; and a golden-file determinism test so corpora are byte-identical across platforms.

### 5.2 Notes on the existing model

`model.py` returns `(logits, hidden)` and exposes `backbone(x, positions)` over input embeddings, which is the latent hook; `latent.py` calls `backbone`, never `__call__`. The Python list of blocks is fine at 6 to 8 layers, and I'd only switch to scan-over-layers plus remat past about 16. Weight tying on, RoPE/RMSNorm/SwiGLU as implemented. Data order and RNG are pure functions of `(seed, step)`, and the run config is a frozen dataclass written into each `runs/*.json`. Checkpointing isn't wired yet; params stay in memory for the run, and Orbax or equivalent is planned.

---

## 6. What could go wrong

The failure I worry about most is self-distillation collapse once the dual-pass lands, which is what the stop-grad on `t_j` and a strong `η·L_explicit` are for; an EMA teacher or a frozen 1-epoch CoT checkpoint is the fallback. Next is the halt pinning to `N*=1` or `N*=N`, which `β·L_ponder` guards against and the depth histogram makes visible. Distillation can also fight the answer loss, so `α` warms up over the first few hundred steps (a loss-weight ramp, not a data curriculum) and gets ablated. Positional alignment stays rigid when the optimal latent count differs from `m`; the halt gives some slack, the `c>1` block mapping gives more, and DTW soft alignment is there if I need it.

A from-scratch model at this size may simply be too weak to run latent BFS at all, which is what the Phase-0 micro gate and the `d_model` sweep are there to catch early; the task is designed to be learnable at 10 to 20M. Static `MAX_STEPS` with masking, per-step remat and donated buffers keep JAX from recompiling or blowing up memory, and sub-stream seeding plus the golden-file test pin down Rust/Python determinism. The novelty challenge from CODI, CCoT and 2511.21581 is the one I can't engineer around; the answer is the fused objective plus the Pareto plus the theory, with the RL-halt ablation shipped so the comparison is direct. GSM8K stays strictly optional. The paper stands on ProsQA, the control and the micro task.

---

### Anchor references

Coconut (arXiv:2412.06769) · PonderNet (2107.05407) · CODI (2502.21074) · CCoT (2412.13171) · ICoT-KD (2311.01460) / Stepwise-Internalization (2405.14838) · Quiet-STaR (2403.09629) · Learning-When-to-Stop (2511.21581) · SIM-CoT latent-collapse (2509.20317) · Filler-tokens TC⁰ bound (2404.15758) · ProntoQA (2210.01240) · GSM8K (2110.14168). Stack: Equinox + Optax (Orbax checkpointing planned); determinism after Levanter, data and RNG as pure functions of `(seed, step)`.
