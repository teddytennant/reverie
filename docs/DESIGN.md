# Reverie — Design Document

*Adaptive, curriculum-free reasoning in a continuous latent space. A successor to Coconut, in JAX (+ Rust for data).*

Status: design + reference for the implementation in `reverie/`. Method codename: **Reverie** — wordless thought. Everything below is decisive; alternatives appear only as named ablations or risk fallbacks.

> **Implementation status (read this first).** The shipped code (`reverie/latent.py`) realizes trajectory distillation in **output space**: each continuous thought `y_j` is supervised, through the *tied* LM head, to decode to its gold reasoning step's concept token — one param-free term (`L_traj`) that simultaneously distills the trajectory *and* makes the latents linearly decodable (so it also serves as the interpretability probe of §2.3-E). This is a cleaner, self-contained instantiation of "supervise every latent against the teacher step." The richer **hidden-space** variant of §2.1–2.3 — a second teacher-forced explicit pass producing per-step hidden targets `t_j` for `L_distill` + an `L_explicit` multitask term — is the planned enhancement (a strict superset). The core objective actually trained is:
> `L = Σₙ pₙ·CE(answer, W yₙ) + α·Σⱼ CE(k_j, W y_j) + γ·(−log p_m) + β·KL(p‖Geom(λ_p))`,
> i.e. `{L_answer, L_traj(=distill/probe fused), L_depth, L_ponder}`. The novelty pillars (§3) — the fused depth-supervised differentiable halt, the single-model Pareto, the serial-depth theory — hold identically for this instantiation.

---

## 1. Thesis

Coconut reasons in a continuous latent space by feeding the last-layer hidden state back as the next input embedding, but it pays for that three separate times. **(i) A brittle multi-stage token-replacement curriculum:** language reasoning steps are swapped for latent slots one stage at a time, optimizer state is reset per stage, and adding several latents at once (its final transition) spikes the training loss; follow-up work (SIM-CoT) reports the reasoning pattern *collapses to ~12.5%* when latents are scaled. **(ii) No supervision of the thoughts themselves:** only the final answer produces gradient, so the latent trajectory is un-audited, hard to optimize, and opaque (Coconut ships no interpretability or safety evaluation). **(iii) A fixed, hand-set number of latent steps** (padded to a constant), so every problem pays the *deepest* problem's serial-compute budget.

**Reverie replaces all three at once with a single loss.** We distill a compressed discrete-CoT trajectory **step-by-step** into a variable-length continuous-thought chain, and we choose the chain's length with a **differentiable PonderNet-style halt whose target is the teacher's own per-instance reasoning depth** — one curriculum-free stage, no reinforcement learning, no post-hoc classifier. The single knob on the halt lets **one trained model trace an accuracy-vs-latent-compute Pareto frontier**, and because the halt is supervised by ground-truth depth we can show the spent budget is **calibrated to problem difficulty** — the model spends serial latent steps only where the instance demands them.

---

## 2. Method

### 2.0 Notation and substrate

A decoder-only transformer backbone `f_θ` (already implemented in `reverie/model.py`) maps a sequence of **input embeddings** `E ∈ R^{L×d}` to post-final-norm last-layer hidden states `H ∈ R^{L×d}`, with a (tied) LM head `W = tok_emb` giving `logits = H Wᵀ`. `f_θ` operates on one sequence; batch with `vmap`. The Coconut mechanism is the single fact we reuse: **a latent thought is a last-layer hidden state fed back, unprojected, as the next input embedding.** No projection matrix between hidden and embedding space (confirmed against Coconut's `coconut.py`).

For each training instance the data generator (`data-gen/`) gives us, *for free and verified*:

- a **question** `q` (shuffled fact bag + the binary "Is E a C⁺ or C⁻?" query),
- a **gold reasoning trajectory** `s = (s_1, …, s_m)` — the membership fact then the chain of `Every A is a B.` edges along the gold path,
- the **per-instance depth** `m = n_hops` (the number of serial deductions this instance requires),
- the **answer** `y` (`E is a C⁺.`) and, per step, its **key concept token** `k_j` (the target of hop `j`, i.e. the `B` in `s_j`).

`m` and `k_j` are the two supervision signals prior latent methods lack: a per-instance *length* target and a per-step *content* target. Reverie uses both.

### 2.1 Two modes of one model (single stage, curriculum-free)

The same weights `θ` run in two input regimes within one training step:

- **Explicit mode (the teacher).** Teacher-force the discrete sequence
  `[q, <bot>, s_1, …, s_m, <eot>, ###, y]`
  in one forward pass. This produces (a) the ordinary CoT language-model loss `L_explicit`, and (b) the **step-summary targets** `t_j := H[end(s_j)]` — the last-layer hidden state at the final token of step `s_j` (post final-norm; the contextualized summary of "step j is done"). We take `t_j` under stop-gradient. Because teacher and student share `θ`, this is *self-distillation* (à la CODI) — no second model, no frozen checkpoint, no staging.

- **Latent mode (the student).** Starting from the prompt embeddings up to `<bot>`, run the Coconut recurrence for up to `N` steps, producing continuous thoughts `z_1, …, z_N` and, at each step, a scalar halting logit. The student is pulled toward the teacher's trajectory (content) and toward the teacher's depth `m` (length), and reads out the answer under a halting distribution.

Both losses are applied together from step 1. There is **no curriculum**: we never stage-replace tokens, never reset the optimizer. `L_explicit` keeps the teacher mode sharp (stable, non-degenerate targets); stop-grad on `t_j` prevents the targets from collapsing to meet the student.

### 2.2 The latent recurrence (exact, and JAX-shaped)

Coconut's recurrence grows the sequence by one latent per step. To keep static shapes (single XLA compile, `lax.scan`), we pre-allocate a buffer of length `P + N` (`P` = prompt length incl. `<bot>`) and fill one latent slot per step under a causal + validity mask:

```
E ← zeros[P+N, d];  E[:P] ← embed(prompt)          # prompt incl. <bot>
carry₀ = (E, cum_p=0, acc=0, mask_valid=[1]*P + [0]*N)

def step(carry, n):                                 # n = 0..N-1  (static length N)
    E, cum_p, acc, valid = carry
    H          = f_θ(E, positions, mask=causal & valid)   # full padded forward
    z          = H[P+n-1] if n>0 else H[P-1]        # last valid position's hidden
    E          = E.at[P+n].set(z)                   # write latent slot (dyn-update)
    valid      = valid.at[P+n].set(1)
    λ          = sigmoid(w_h · z + b_h)             # conditional halt prob at step n
    p_step     = λ * (1 - cum_p)                    # PonderNet: prob of halting exactly now
    acc        = acc + p_step * z                   # (used only for logging E[z])
    cum_p      = cum_p + p_step
    return (E, cum_p, acc, valid), (z, λ, p_step)

(_, (Z, Λ, P_step)) = lax.scan(remat(step), carry₀, arange(N))   # Z: [N, d]
```

`Z = (z_1,…,z_N)` and the halting distribution `p_n = λ_n ∏_{j<n}(1−λ_j)` (a single `cumprod` over `Λ`, with `λ_N := 1` so `Σ_n p_n = 1`). `N` (= `MAX_STEPS`, e.g. 8) is **static**; *effective* depth varies by masking and by the halt, never by changing the loop length (the recompilation trap). `remat` on the step bounds activation memory to O(1) in `N` at the cost of one recompute. This is one shared, fully-differentiable unroll — the heavy cost is `N` backbone forwards, independent of how many depths we later score.

### 2.3 The four losses

Let `D(a,b)` be a direction-sensitive distillation distance — cosine distance on layer-normalized vectors, `D(a,b) = ‖ā − b̄‖²` with `ā = a/‖a‖` (magnitude is carried by the residual stream and is not what we want to match; CCoT uses scaled-MSE, CODI uses normalized L1 — cosine is the robust default).

**(A) Trajectory distillation — content, per step (this is what "supervises the latents").**

```
L_distill = (1/m) Σ_{j=1}^{m} D( z_j , sg(t_j) )
```

Latent `z_j` is aligned to teacher step `j` for the first `m` latents; latents `z_{m+1..N}` carry no distillation target (the halt is meanwhile pushed to stop at `m`). This is *full-trajectory* alignment — every latent is supervised — which no prior single-stage method does (CODI aligns one anchor token; Coconut aligns nothing).

**(B) Adaptive-halt answer read-out — PonderNet reconstruction over depth.**

Read the answer out of *each* candidate depth and weight by the halting distribution. For a depth `n`, score the answer teacher-forced from the state after `n` latents:

```
ℓ_n = − log p_θ( y | q, z_1..z_n, <eot> )          # NLL of the gold answer at depth n
L_answer = Σ_{n=1}^{N} p_n · ℓ_n
```

Critically the weighting is over **losses**, not over states (PonderNet, not ACT) — the latents `z_n` are never mean-field-averaged, so each thought stays intact. Computed cheaply (§2.4): one batched teacher-forced decode with `n` on the batch axis, not `N` sequential generations.

**(C) Depth supervision — the novel halt target.** Vanilla PonderNet regularizes depth only toward a *global* geometric mean `1/λ_p`. We additionally pin the halt, per instance, to the teacher's own chain length `m` via the negative log-likelihood of committing to depth `m`:

```
L_depth = − log p_m           (m = n_hops for this instance)
```

This is the crux of the contribution: **the number of continuous thoughts is supervised to track the teacher's per-instance reasoning depth, differentiably, with no RL and no separately-trained classifier.** It is a single term, an MLE of the halting distribution against a data-given target.

**(D) Halting prior — anti-collapse + the global compute knob.** KL from the halting distribution to a truncated geometric prior:

```
p_G(n) ∝ λ_p (1−λ_p)^{n-1}
L_ponder = KL(p ‖ p_G) = Σ_n p_n [ log p_n − log λ_p − (n−1) log(1−λ_p) ]
```

With `L_depth` doing per-instance supervision, `L_ponder`'s job is (i) prevent the known PonderNet failure of collapsing to a single depth, (ii) set where the model natively lives (prior mean depth `1/λ_p`), and (iii) act as a *training-time* compute prior. The *inference-time* Pareto knob is separate (§2.5).

**Optional (E) Decodability probe — interpretability (ablatable, `δ`).** A single linear probe `V` must recover each step's key concept token from its latent:

```
L_probe = (1/m) Σ_{j=1}^{m} − log softmax(V z_j)[k_j]
```

This makes latents *linearly decodable back to the trajectory* and yields a first-class interpretability metric (probe-decode accuracy) — directly answering the "latent thoughts are opaque" critique that Coconut, Soft-Thinking and Quiet-STaR all draw. `k_j` = the concept introduced at hop `j`.

**Total objective (single backward pass):**

```
L = L_answer + α·L_distill + γ·L_depth + β·L_ponder + η·L_explicit + δ·L_probe
```

Defaults: `α=1.0, γ=0.5, β=0.01, η=1.0, δ=0.1`, `λ_p ≈ 0.15` (prior mean depth ≈ 6–7, matching ProsQA), `N = MAX_STEPS = 8`. Core method = `{answer, distill, depth, ponder, explicit}`; `probe` is the interpretability add-on. **Ablations map directly onto terms:** *no-distillation* drops `α·L_distill` (→ answer-only adaptive Coconut); *no-halting* drops `γ·L_depth + β·L_ponder`, fixes `n = m` (or `N`), and replaces `L_answer` with `ℓ_m` (→ single-stage trajectory-distilled Coconut, fixed depth); *no-probe* drops `δ·L_probe`.

For math tasks with `c > 1` latents per step (GSM8K), map latents to steps in contiguous blocks: align `z_{(j-1)c+1..jc}` — supervise the block's last latent to `t_j` and leave the intermediate `c−1` free — and set the depth target to `c·m`. Lead task is ProsQA at `c = 1`.

### 2.4 Efficient JAX computation

The naïve reading — "decode an answer at every candidate depth" (O(N) full generations) plus "run a teacher and a student" — is avoided by three structural facts, all in one differentiable graph:

1. **One teacher forward, one student unroll.** Explicit mode is a single teacher-forced pass (gives `L_explicit` and all `t_j`, stop-gradded). Latent mode is the single `lax.scan` of §2.2 (`N` backbone steps, `remat`-bounded memory), producing all `z_n`, `λ_n`, `p_n` at once. Cost of the heavy transformer is `N` steps, **independent of how many depths we score**.
2. **Batched-over-depth answer scoring — never a generation per depth.** The gold answer `y` is fixed, so teacher-force it and put the `N` candidate depths on the **batch axis**: one decoder pass over a `[N, P+N+A]` buffer, where row `n` exposes `prompt + z_{1..n} + <eot> + y` via a per-row validity mask and hides `z_{>n}`. That yields all `ℓ_n` in parallel — replacing `N` sequential autoregressive decodes with one batched teacher-forced forward.
3. **Everything else is elementwise.** `p_n` is one `cumprod`; `L_ponder`, `L_depth`, `L_distill`, `L_probe` are closed-form sums over stacked vectors. Per-example `m` in a batch is handled by padding to `max(m)` and masking the `distill`/`depth`/`probe` terms.

Result: **one teacher forward + one student unroll + one batched read-out + a handful of masked sums**, differentiable end-to-end (no REINFORCE, no per-depth re-decode, no auxiliary router). Train step is `eqx.filter_jit(donate="all")` with f32 master weights, bf16 compute on GPU only (guard `to_bf16` behind `jax.default_backend()=="gpu"`; bf16 needs no loss scaling; RMSNorm/softmax/CE stay f32).

### 2.5 Inference and the Pareto knob

At inference, gradients are unnecessary, so use `lax.while_loop` and pay the *actual* depth: roll latents and stop at the first depth whose cumulative halt mass crosses the budget,

```
N* = min{ n : Σ_{j≤n} p_j > 1 − ε }
```

then emit `<eot>` and greedily decode the answer from `z_{N*}`. **Sweeping ε from one trained model traces the accuracy-vs-latent-compute Pareto frontier** (small ε → deeper → more accurate/costly; large ε → shallower). Equivalent knob: add a scalar bias to the halt logit and sweep it. This is the single-model dialable frontier that point-result papers (CODI, CCoT, Coconut) do not provide.

### 2.6 Theory: why adaptive serial depth is necessary (proposition + sketch)

Let a distribution present instances with required serial reasoning depth `d ∼ P(d)` (heterogeneous — the realistic case; our generator dials `P(d)` exactly). A latent thought adds one step of *serial* computation (each `z_n` attends to `z_{<n}`), unlike Pfau et al.'s filler tokens, which add only *parallel* width and are provably TC⁰-bounded and useless for inherently serial work.

**Proposition (informal).** *On the serial fragment* (instances requiring `d` mutually-dependent deductions that cannot be parallelized), a latent model that emits `n < d` thoughts cannot represent the computation. Therefore a **fixed**-depth model with budget `N` is correct on the deep tail only if `N ≥ max supp(d)`, and then spends `max supp(d)` serial steps on *every* instance; its expected latent cost is `max supp(d)`. A **per-instance adaptive** model that sets `n = d` is correct everywhere at expected cost `E[d] ≤ max supp(d)`, with strict inequality whenever `P(d)` is non-degenerate. Hence adaptive latent depth is *necessary* to match explicit CoT at `E[d]` compute on heterogeneous-depth distributions, and *sufficient* on the sub-TC⁰-serial fragment our tasks live in.

*Proof sketch.* Serial-depth lower bound: an `n`-step latent chain composes `n` attention/MLP updates; a `d`-hop reachability query where hop `j+1`'s relevant fact is unknown until hop `j` is resolved requires ≥ `d` dependent updates, so `n ≥ d` (matches the empirical "gap widens with `k`"). Cost gap: `E[d] < max supp(d)` for any non-point `P(d)`. We validate both empirically with the depth-variance benchmark (§4): a fixed-`N` model must choose between wasting compute (`N=6` on `k=2` instances) or failing the tail (`N=3` fails `k≥4`); Reverie matches accuracy at lower mean depth and `steps-used` correlates with `k`.

This is deliberately scoped as a proposition with an empirical validation, not an overclaimed theorem — it is the reviewer-legible framing that ties the compute result to the TC⁰/serial-depth literature and answers the surveys' "generalization to novel structures" open problem.

---

## 3. Novelty

### 3.1 Differentiation table

| Method | Supervised latents? | Adaptive length (per-instance)? | Single-stage? | Key differentiator vs Reverie |
|---|---|---|---|---|
| **Coconut** (Hao 2024) | No — answer-only | No — fixed, padded | No — multi-stage token-replacement curriculum + optimizer resets | No distillation; brittle staging (collapses when latents scaled); fixed depth |
| **CCoT** (Cheng & Van Durme 2024) | Yes — teacher hidden subset (trajectory) | Partial — separate learned `end_ψ` classifier on a **fixed** train-time ratio | No — multi-stage (φ then ψ, layer-wise unfreeze) | Halt is a bolted-on classifier, not a differentiable prior; multi-stage; no depth supervision |
| **ICoT-KD / Stepwise-Internalization** (Deng 2023/2024) | KD: teacher hidden (indirect); SI: no latents at all | No — depth tied to layers (KD) / zero tokens (SI) | KD: no (3 coupled models); SI: token-deletion curriculum | Not a per-instance adaptive *latent count*; either 3-model or answer-only |
| **Quiet-STaR** (Zelikman 2024) | Reward-only (REINFORCE), **discrete text** thoughts | No — fixed thought length, per-token | Pretraining objective | Discrete not continuous; high-variance RL; not a compression/efficiency method |
| **PonderNet** (Banino 2021) | No — task-loss only | Yes — differentiable geometric-prior halt | Yes | Not CoT/latent at all; tied step function; **no content supervision, no teacher-depth target** |
| **CODI** (Shen 2025) | Yes — **single anchor** token hidden | No — fixed 6 | Yes — self-distillation | One anchor ≠ trajectory; fixed length; no halt |
| **Learning-When-to-Stop** (2511.21581) | No — answer reward | Yes — **RL/PPO** per-instance halt | (bolted on a Coconut backbone) | RL (high-variance, separate reward loop); no distillation; not differentiable-native |
| **Reverie (ours)** | **Yes — every latent ← teacher step (+ linear-decodability probe)** | **Yes — differentiable geometric-prior halt supervised by teacher depth `m`** | **Yes — one joint self-distillation stage, no curriculum, no RL** | **Fuses trajectory-distillation + teacher-depth-supervised differentiable halt in one loss; dialable single-model Pareto; difficulty-calibrated; serial-depth theory** |

### 3.2 What is genuinely new, stated plainly

No single ingredient is new, and we say so in the intro: continuous thoughts (Coconut), trajectory distillation (CCoT), single-stage self-distillation (CODI), differentiable halting (PonderNet), per-instance latent halting (2511.21581, via RL). **What is unclaimed is the fusion:** a *single-stage, curriculum-free* objective that distills the *full* teacher trajectory into *every* latent **and** sets the chain length with a *differentiable geometric-prior halt whose target is the teacher's own per-instance depth* — one loss, no RL, no post-hoc classifier, no staging. Concretely, the seams we occupy that the closest works leave open:

- vs **CODI**: single-anchor → **full trajectory**, and fixed-6 → **adaptive, depth-supervised**.
- vs **CCoT**: multi-stage + separate `end_ψ` classifier on a fixed ratio → **single-stage + differentiable prior halt supervised by depth**.
- vs **2511.21581**: RL/PPO halt → **differentiable, distillation-native halt** (lower variance, one backward pass, and the halt target comes from the teacher rather than from sparse correctness reward).
- vs **Coconut/PonderNet**: adds the two supervision signals each lacks (per-step content; per-instance length).

### 3.3 The defensible NeurIPS claim

> A single curriculum-free, RL-free model that **matches explicit chain-of-thought on a difficulty-calibrated accuracy-vs-latent-compute Pareto frontier**, by distilling a discrete reasoning trajectory into a variable-length continuous-thought chain whose length is a differentiable function supervised by the teacher's per-instance depth — with a proposition (and controlled empirical validation) that **per-instance adaptive latent depth is necessary** to do so on distributions with heterogeneous serial-reasoning depth.

Three pillars, each independently reviewer-legible: the **fused objective** (§2.3), the **single-model Pareto + difficulty-calibration deliverable** (§2.5, §4), and the **variable-serial-depth theory** (§2.6). We do **not** frame the paper as "adaptive halting" or "single-stage distillation" alone — both are taken. We cite CODI, CCoT, and 2511.21581 as closest prior work and differentiate on the fusion.

*(If review pressure shows the fused-objective seam is still too narrow, the sharper fallback that does not overlap CCoT/ICoT is to lead entirely with pillars 2+3 — reframe as "difficulty-calibrated latent compute": the contribution becomes the single-model Pareto frontier with a proven difficulty-calibration guarantee and the serial-depth theorem, with the objective demoted to the vehicle. That lane is unoccupied regardless of the objective's novelty.)*

---

## 4. Experiments

### 4.1 Lead, control, debug, capstone

- **Lead — self-generated ProsQA** (`data-gen/`, fictional tokens, from-scratch model). No external corpus, no pretrained weights, no leakage; difficulty is a dial, so the "advantage grows with required search depth" curve is producible on a laptop-to-single-GPU budget. This is where latent planning is least confounded.
- **Control — linear chains.** Expected: the Reverie−CoT gap *collapses* to ≈0 (nothing to search). This is a feature of the argument (rules out "latent just helps everywhere"), not a failure. *No new generator code needed:* running `data-gen` with `--branch 0 --trap-depth 0` removes all distractor branches, leaving a pure source→…→answer chain plus a disjoint decoy — the linear-reasoning control. (A named ProntoQA `--task` mode is an optional cosmetic addition.)
- **Debug harness — micro-ProsQA** (k-hop letter-graph pointer-chase, ≤8 nodes, ~20-token vocab). CPU, minutes. CI gate before any GPU spend.
- **Capstone (optional, GPU-only) — GSM8K**, pretrained GPT-2-124M, iCoT-augmented (~385k), `c=2`. Explicitly optional; produces the weakest planning story and needs pretraining, so it is a "does it transfer to real math" stretch, not a gate.

### 4.2 Baselines (all at matched compute) and ablations

Baselines: **No-CoT** (question→answer), **CoT** (explicit reasoning SFT), **Coconut** (our faithful reimpl — multi-stage curriculum, optimizer resets, fixed padded latents), **Reverie** (ours). Ablations (map to §2.3 terms): **no-distillation** (drop `L_distill`), **no-halting** (drop `L_depth+L_ponder`, fix depth), **no-probe** (drop `L_probe`), and an optional **RL-halt** swap (replace the differentiable halt with a PPO stop-policy à la 2511.21581) to show the differentiable halt is ≥ in accuracy and lower in seed-variance.

### 4.3 Metrics

1. **Final-answer exact-match accuracy** (greedy), overall and **stratified by hop count** `k`.
2. **Latent-steps-used** — mean `N*` at inference (the realized adaptive depth) and its histogram.
3. **Accuracy-vs-latent-FLOPs Pareto** — sweep ε; latent-FLOPs ≈ `E[N*] ×` per-step backbone FLOPs; plot Reverie's frontier against Coconut's single fixed point and CoT's (many decoded tokens) point.
4. **Difficulty calibration** — Spearman `ρ(N*, k)` using the generator's ground-truth `n_hops`; plus mean `N*` vs `k` curve. This is the claim that compute is spent where needed; ground-truth difficulty is available *because we generate the data* — an advantage no natural-corpus paper has.
5. **Training stability across seeds** — 3–5 seeds; report mean±std and count of "collapsed" runs (accuracy < random + margin). Target: reproduce Coconut's instability when latents scale (the SIM-CoT collapse) and show Reverie is stable single-stage.
6. **Interpretability** — linear-probe decode accuracy of latents → key concept tokens (from `L_probe`), reported with vs without the probe term.

### 4.4 Scaling story (all cheap)

1. **Depth sweep** — regenerate *test* at `k = 2..6` (train fixed); plot Reverie−CoT gap vs `k`. Expected ≈0 at `k=2`, widening monotonically — the planning claim, demonstrated (and the empirical face of §2.6).
2. **Branching sweep** — hold `k`, raise `branch`/`trap_depth`; CoT should degrade faster than Reverie (more live branches to hedge over).
3. **Model-size sweep** — `d_model ∈ {128,256,384}`; show the advantage is not a capacity artifact.
4. **Control** — ProntoQA-5hop: gap collapses.
5. **Depth-variance benchmark** — a train/test mix over `k∈{2..6}`; show a fixed-`N` model must trade wasted compute for tail failure while Reverie matches accuracy at lower mean depth (validates the proposition).

### 4.5 Compute-limited plan (yields real numbers)

- **Phase 0 — CPU, minutes (gate).** micro-ProsQA; 2 layers, `d_model=128`, ~1M params; 20k train / 1k val; ~3k steps. Pass condition: Coconut>CoT>No-CoT, EM>95%, latent gap ↑ with `k`, gradient-flow + halt tests green.
- **Phase 1 — single GPU, ~30–90 min (headline).** From-scratch, 6–8 layers, `d_model=256`, 8 heads, ~10–20M params (the `ModelConfig` defaults, `n_heads` bumped to 8). Data: **train 20,000 / val 500 / test 500** (exceed Coconut's 17,886 for free; hold out by *fresh seeds*, not row-splitting, so test graphs are genuinely unseen). Curriculum-free: `c=1`, `MAX_STEPS=8`, lr `1e-4`, effective batch 128 (grad-accum via `optax.MultiSteps`), AdamW `(0.9,0.95)` wd `0.1`, warmup-cosine, global-norm clip 1.0. ~3–5k optimizer steps. Train **all four baselines at identical compute**; 3–5 seeds for the stability table.
- **Phase 2 — sweeps** (cheap; reuse the Phase-1 model): ε-Pareto and calibration need *no retraining*; depth/branching/size sweeps are minutes each of data regen + short train.
- **Phase 3 (optional) — GSM8K capstone** on GPU if free.

Expected headline: Reverie **reaches CoT accuracy on ProsQA at fewer and adaptive latent steps**, its ε-frontier **dominates Coconut's fixed point**, it is **stable across seeds where our Coconut reimpl collapses**, `N*` **correlates with `k`** (calibration), latents are **linearly decodable**, the **gap widens with depth heterogeneity**, and it **collapses to ≈CoT on the linear ProntoQA control**.

---

## 5. Repo architecture

Builds on the existing scaffold (`reverie/model.py` `Transformer`/`ModelConfig`; `data-gen/` `reverie-datagen`). New files marked `＋`.

```
reverie/                                  # repo root (pyproject: jax, equinox, optax, orbax, jaxtyping, tyro)
├── docs/
│   └── DESIGN.md                         # this file
├── configs/                              ＋ frozen-dataclass configs, hashed into run-dir names
│   ├── base.py                           ＋ shared model/optim/data defaults
│   ├── prosqa_reverie.py                 ＋ ours (full objective)
│   ├── prosqa_coconut.py                 ＋ multi-stage Coconut reimpl baseline
│   ├── prosqa_cot.py                     ＋ explicit-CoT baseline
│   ├── prosqa_nocot.py                   ＋ no-CoT baseline
│   └── micro.py                          ＋ CPU debug (Phase 0)
├── reverie/                              # JAX/Equinox package
│   ├── __init__.py                       # exists
│   ├── model.py                          # exists: Transformer, ModelConfig, backbone()/embed()/project()
│   ├── latent.py                         ＋ latent recurrence: static-shape scan (§2.2), halt head, inference while_loop
│   ├── heads.py                          ＋ halting head, decodability probe V, answer read-out helpers
│   ├── objective.py                      ＋ teacher pass + student unroll + total loss L (§2.3), batched-over-depth answer scoring (§2.4)
│   ├── losses.py                         ＋ L_answer, L_distill, L_depth, L_ponder, L_explicit, L_probe (pure fns)
│   ├── curriculum.py                     ＋ Coconut-baseline staging ONLY (stage schedule, optimizer reset); Reverie uses none
│   ├── optim.py                          ＋ make_optimizer (warmup-cosine + clip + AdamW + MultiSteps), to_bf16 policy
│   ├── data.py                           ＋ JSONL loader; render abstract graph → token text (facts/query/steps/answer); (seed,step) order
│   ├── tokenizer.py                      ＋ fictional-concept vocab + <bot>/<eot>/latent id/### ; deterministic
│   ├── train.py                          ＋ train loop: filter_jit(donate) step, orbax ckpt, per-step fold_in RNG
│   ├── eval.py                           ＋ EM, steps-used, Pareto sweep, calibration ρ, probe accuracy
│   └── rng.py                            ＋ root key, jr.fold_in(root, step); example_index = permute(seed, step)
├── scripts/                              ＋ thin tyro entrypoints
│   ├── make_data.py                      ＋ shells the Rust bin per split with fresh seeds → data/*.jsonl
│   ├── train.py                          ＋ `python scripts/train.py --config configs/prosqa_reverie.py`
│   ├── eval.py                           ＋ metrics + Pareto/calibration plots
│   └── sweep.py                          ＋ depth/branching/size sweeps (regen test + eval)
├── data-gen/                             # Rust crate (exists)
│   ├── Cargo.toml                        # exists (zero-dep, LTO release)
│   └── src/
│       ├── main.rs                       # exists: SplitMix64 RNG, layered DAG, traps, decoy, BFS-verify, JSONL
│       ├── lib.rs                        ＋ extract gen_instance/reachable as a lib (unit-testable)
│       ├── prontoqa.rs                   ＋ linear-chain control generator (--task prontoqa)
│       └── micro.rs                      ＋ tiny letter-graph task (--task micro)
├── tests/                                # python (exists, empty)
│   ├── test_shapes.py                    ＋ forward/latent shapes, masks
│   ├── test_latent_grad.py               ＋ gradient flows answer→prompt through latent chain; no stray stop_grad
│   ├── test_halting.py                   ＋ Σ p_n = 1; cumprod correctness; ε early-exit matches training p_n
│   ├── test_distill_align.py             ＋ z_j↔t_j alignment + masking on ragged m
│   └── test_overfit.py                   ＋ micro batch overfits (CI gate)
└── data-gen/tests/                       ＋ Rust
    ├── determinism.rs                    ＋ same (seed,params) ⇒ identical bytes (golden file)
    └── reachability.rs                   ＋ every instance: answer reachable, decoy not
```

### 5.1 Rust generator interface (contract with Python)

Rust owns the **abstract, verified problem**; Python owns **token rendering** (as the existing `main.rs` header states). Current emitted JSONL schema per line is already exactly what Reverie needs:

```json
{"id":int,"n_entities":int,"entities":["ger","scrom",...],"edges":[[a,b],...],
 "source":int,"candidates":[c0,c1],"answer":int,"gold_path":[n0,...,nk],
 "n_hops":int,"n_distractors":int}
```

- `gold_path` → the teacher trajectory `s` (rendered by `reverie/data.py` as the membership fact + `Every {A} is a {B}.` edges); `len(gold_path)-1 = n_hops = m`, the halt's depth target and the calibration ground truth.
- `edges/entities` → the shuffled fact bag; `source/candidates` → the `Is E a C⁺ or C⁻?` query; `answer` → `C⁺`.
- `n_hops`, `n_distractors` → per-instance difficulty covariates for stratified metrics and calibration.

CLI (existing + `＋` additions): `reverie-datagen --n N --seed S --hops H --branch B --trap-depth D [--task prosqa|prontoqa|micro] [--out FILE]`.

**Two required refinements to the current `main.rs`** (specified now, low-risk): (1) **per-example sub-streams** — seed each instance as `rng_i = SplitMix64(global_seed ^ 0x9E3779B97F4A7C15·i)` instead of one shared advancing stream, so any example is regenerable in isolation and generation is embarrassingly parallel *and* still bit-reproducible (needed for "hold out by fresh seed"). (2) **golden-file determinism test** so the corpus is byte-identical across platforms/CI. The existing `reachable()` assertions already give repair-or-reject label verification — keep them. (SplitMix64 is fine and cross-platform; ChaCha8 is an optional upgrade only if we ever want cryptographic stream independence — not required.)

### 5.2 Notes on the existing model

`model.py` already returns `(logits, hidden)` and exposes `backbone(x, positions)` over **input embeddings** — the exact latent hook; `latent.py` calls `backbone`, not `__call__`. Keep the Python-list-of-blocks (readable, compiles fine at 6–8 layers); switch to scan-over-layers + per-layer `remat` only if depth grows past ~16 (the MaxText/Levanter trick). Weight tying is on; RoPE/RMSNorm/SwiGLU as implemented. Reproducibility follows Levanter's principle: **data order and RNG are pure functions of `(seed, step)`** (`rng.py`), config is a frozen dataclass hashed into the run dir, Orbax async checkpoints save params+opt_state+step+rng.

---

## 6. Risks, mitigations, and build order

### 6.1 Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Self-distillation collapse** (student ignores teacher, or shared-weight targets drift) | Stop-grad on `t_j`; keep `η·L_explicit` strong so teacher mode stays sharp; cosine (direction) distance so magnitude drift is ignored; **fallback:** EMA-teacher (`t_j` from an EMA copy of `θ`) or a frozen 1-epoch CoT checkpoint (reintroduces one model but is the ICoT-KD-safe fallback). |
| 2 | **Halt collapses** to `N*=1` or `N*=N` (known PonderNet failure) | `β·L_ponder` KL prior is explicitly anti-collapse; `γ·L_depth` pins per-instance target `m`; monitor the depth histogram each eval; tune `β, λ_p`. |
| 3 | **Distillation vs answer tension** (matching teacher hidden fights answer loss) | Normalized/cosine distance; warm up `α` over the first few hundred steps (a loss-weight ramp, *not* a data curriculum — still single-stage); ablate `α`. |
| 4 | **Rigid positional alignment** when the student's optimal latent count ≠ `m` | The halt provides slack (it can stop before `m`); `c>1` block mapping for multi-latent steps; **fallback:** monotonic/DTW soft alignment `z_i↔t_j` if strict `j↔j` underperforms. |
| 5 | **From-scratch model too weak** to exhibit latent BFS | Phase-0 micro gate before GPU; `d_model` sweep; the task is designed learnable at ~10–20M (Coconut used GPT-2-124M, we test the mechanism, not scale). |
| 6 | **JAX recompilation / scan memory** | Static `MAX_STEPS` + masking (never vary loop length); `remat` per latent step; batched-over-depth read-out; `donate="all"`; guard bf16 to GPU. |
| 7 | **Novelty challenge** (CODI/CCoT/2511.21581) | Lead with the fused objective + Pareto + theory (§3.3); ship the RL-halt ablation to prove the differentiable halt is competitive and lower-variance; §3.3 fallback reframes on pillars 2+3 if needed. |
| 8 | **Rust/Python determinism drift** | Per-example sub-stream seeding; golden-file byte test in CI; record dataset seed+params+version in the checkpoint. |
| 9 | **GSM8K over-scope** eats the timeline | Strictly optional capstone; the paper stands on ProsQA + control + micro; only touch GSM8K if a GPU is idle. |

### 6.2 Build order (five steps, each with a gate)

1. **Data + harness.** Refine `data-gen/` (per-example sub-streams, `lib.rs`, `prontoqa.rs`, `micro.rs`, Rust determinism + reachability tests); build `reverie/data.py` + `tokenizer.py` to render abstract graphs → tokens. **Gate:** golden-file determinism green, BFS-verify green, a rendered ProsQA example matches the real serialization.
2. **Backbone + non-latent baselines (CPU micro).** Wire `train.py`/`eval.py`, implement No-CoT and CoT SFT, overfit micro-ProsQA. **Gate:** EM>95% on micro; CoT>No-CoT.
3. **Latent mechanism + Coconut reimpl.** `latent.py` static-shape scan + halt head; `curriculum.py` staging; reproduce Coconut≥CoT on micro/ProsQA *and* its instability when latents scale. **Gate:** `test_latent_grad`/`test_halting` green; Coconut qualitative behavior reproduced.
4. **Reverie objective.** Teacher pass + step-aligned `L_distill` + PonderNet `L_answer` + `L_depth` + `L_ponder` (+ `L_probe`), single stage; batched-over-depth scoring. **Gate:** on ProsQA, Reverie ≥ Coconut at fewer/adaptive steps; depth histogram tracks `m`; stable across 3 seeds.
5. **Scaling + Pareto + theory eval.** ε-Pareto, calibration ρ, depth/branching/size sweeps, ProntoQA control, depth-variance benchmark, seed-stability table; write results. **Gate:** gap widens with `k`; ε-frontier dominates the Coconut point; `ρ(N*,k)` strong-positive; control gap collapses. *(Optional Phase 3: GSM8K.)*

---

### Anchor references

Coconut (arXiv:2412.06769) · PonderNet (2107.05407) · CODI (2502.21074) · CCoT (2412.13171) · ICoT-KD (2311.01460) / Stepwise-Internalization (2405.14838) · Quiet-STaR (2403.09629) · Learning-When-to-Stop / RL latent halt (2511.21581) · SIM-CoT latent-collapse (2509.20317) · Filler-tokens TC⁰ bound (2404.15758) · ProntoQA (2210.01240) · GSM8K (2110.14168). Implementation substrate: Equinox + Optax + Orbax; determinism after Levanter (data/RNG as pure functions of `(seed, step)`).
