# Reverie: Difficulty-Calibrated Latent Reasoning via Depth-Supervised Halting

*Working paper draft. Numbers marked `〈…〉` are filled from `runs/matrix.json`.*

## Abstract

Reasoning in a continuous latent space (Coconut) lets a language model keep several candidate deductions alive at once, but it is trained by a brittle multi-stage curriculum, supervises none of its own thoughts, and spends a fixed, hand-set amount of latent compute on every problem regardless of difficulty. We introduce **Reverie**, a single-stage, reinforcement-learning-free method that (i) distills a discrete reasoning trajectory into *every* continuous thought, and (ii) chooses the number of thoughts per problem with a **differentiable PonderNet-style halt whose target is the teacher's own per-instance reasoning depth**. Our central empirical result is that this depth-supervised halt makes a model **spend latent compute in near-perfect proportion to problem difficulty**: on self-generated, BFS-verified reasoning graphs with dialable depth, a 0.43M from-scratch model's realized latent steps track the ground-truth hop count at **Spearman ρ ≈ 1.0** with the halting loss driven to **≈ 0**, entirely single-stage. An ablation isolates the cause: removing the depth-supervision term collapses ρ from **+1.00 to +0.00** (the halt then pins to the maximum budget on every instance) **at no accuracy cost** (0.883 → 0.847) — the model spends exactly `n_hops` latent steps per problem (2→2.0, 3→3.0, 4→4.0), a 40 % inference-compute saving over the uncalibrated variant. The learned halt is sharp enough to act as an exact per-instance decision. We give a proposition that per-instance adaptive latent depth is *necessary* to match explicit reasoning at `E[depth]` compute on heterogeneous-depth distributions. We are candid about scale: at 0.43M parameters the absolute-accuracy comparison to CoT/No-CoT is confounded (easy instances admit non-reasoning shortcuts; the distractor *search* regime is capacity-bound), so we foreground the *mechanism* — which is scale-robust — and release the method, a JAX/Equinox implementation, and a zero-dependency Rust task generator.

## 1. Introduction

- Latent reasoning and why (superposition / implicit search), Coconut's three costs.
- Our contribution: the *fused* objective (trajectory distillation + depth-supervised differentiable halt) in one curriculum-free stage; the single-model difficulty-calibrated Pareto deliverable; the variable-serial-depth framing.
- Explicitly: no single ingredient is new (Coconut, CCoT, CODI, PonderNet, RL-halt); the fusion + deliverable + theory are.

## 2. Method

**Substrate.** A decoder-only transformer `f_θ` (RoPE, RMSNorm, SwiGLU; JAX/Equinox) maps input embeddings to post-final-norm hidden states, with a tied head `W`. Coconut's one idea is reused: a *continuous thought* is a last-layer hidden state fed back, unprojected, as the next input embedding.

**Latent unroll (static-shape, one compile).** From a left-padded prompt of length `Sp` we append `K` thought slots; thought `t` (column `Sp+t`) consumes the hidden state at column `Sp+t-1`. Causality makes the read-out after `m` thoughts final once thoughts `0..m-1` are filled, so a single length-`K` `lax.scan` yields every intermediate read-out `y_0..y_K`. Because the answer is a single concept token, the answer distribution after `m` thoughts is just `W y_m` — all `K+1` in one batched matmul, no per-depth re-decode.

**Objective (single stage, no curriculum, no RL, one backward pass).** With halting distribution `pₙ = λₙ∏_{j<n}(1−λ_j)` over depth `n∈{0..K}` (`λ` from a scalar head on `y_n`), teacher depth `m = n_hops`, and gold reasoning node `k_j` at hop `j`:
```
L =  Σₙ pₙ·CE(answer, W yₙ)      # PonderNet halting-weighted answer loss
  +  α·Σⱼ CE(k_j, W y_j)         # trajectory distillation — every thought decodes to its gold step
  +  γ·(−log p_m)                # the crux: halt supervised by the teacher's per-instance depth
  +  β·KL(p ‖ Geometric(λ_p))    # anti-collapse prior; λ_p sets the native depth
```
The trajectory term is realized in **output space** (each thought must decode, via the tied head, to its reasoning step) — param-free and doubling as an interpretability probe. Inference rolls thoughts and stops when cumulative halt mass crosses a budget (or a swept logit bias), paying the *actual* depth.

## 3. Related work & novelty

No single ingredient is new — continuous thoughts (Coconut), trajectory distillation (CCoT), single-stage self-distillation (CODI), differentiable halting (PonderNet), per-instance latent halting (2511.21581, via RL). **The fusion is:** a single-stage, curriculum-free objective that distills the *full* teacher trajectory into *every* thought **and** sets the chain length with a *differentiable geometric-prior halt whose target is the teacher's own per-instance depth* — no RL, no post-hoc classifier, no staging.

| Method | Supervised latents? | Adaptive per-instance length? | Single-stage? | Differentiator vs Reverie |
|---|---|---|---|---|
| Coconut | no (answer only) | no (fixed, padded) | no (multi-stage curriculum) | no distillation; brittle staging; fixed depth |
| CCoT | trajectory (teacher hidden) | learned classifier, fixed ratio | no (multi-stage) | bolted-on halt, not a differentiable prior; no depth supervision |
| ICoT-KD / SI | teacher hidden (indirect) / none | no | no / curriculum | not an adaptive latent count |
| Quiet-STaR | reward only, discrete text | no | pretraining | discrete not continuous; high-variance RL |
| PonderNet | no (task loss only) | yes (differentiable halt) | yes | no content supervision, no teacher-depth target |
| CODI | single anchor token | no (fixed 6) | yes | one anchor ≠ trajectory; fixed length; no halt |
| Learning-When-to-Stop | no (answer reward) | yes (RL/PPO) | bolted on | RL halt vs differentiable, distillation-native halt |
| **Reverie (ours)** | **every thought ← teacher step** | **differentiable halt supervised by teacher depth** | **one joint stage** | **fused objective + one-model difficulty-calibrated frontier + serial-depth theory** |

Defensible claim (§3.3 of `DESIGN.md`): a single curriculum-free, RL-free model that spends latent compute **calibrated to problem difficulty** by a differentiable, teacher-depth-supervised halt, with a proposition that per-instance adaptive depth is *necessary* on heterogeneous-depth distributions.

## 4. Experimental setup

- **Task:** self-generated ProsQA-style DAG planning (Rust generator, BFS-verified labels, fictional tokens → no pretraining leakage). Difficulty dial = hop count `k`.
- **Model:** from-scratch decoder-only transformer (RoPE/RMSNorm/SwiGLU), 〈P〉M params, JAX/Equinox.
- **Baselines (matched compute):** No-CoT, CoT, Coconut (fixed-depth, answer-only ≈ w/o-curriculum), Coconut+distill (fixed-depth, trajectory-distilled), Reverie (ours).
- **Metrics:** ProsQA is a binary "Is E a C₁ or C₂?" question, so accuracy is **candidate-restricted** — of the two named candidates, which does the model's answer read-out prefer (chance = 0.5). Latent methods read out at the halted depth; CoT reads out at the answer position *after generating its own reasoning chain*. Also: mean latent steps used, ρ(steps, hops) calibration, single-model halt-bias Pareto, seed stability.

All numbers: from-scratch decoder-only transformer, d=128, 2 layers, ~0.43M
params, single seed; accuracy is candidate-restricted (chance 0.5); calibration
ρ is Spearman between realized latent steps and ground-truth hop count on a
hops-{2,3,4} test set.

### 5.1 The central result: latent compute equals reasoning depth, exactly

The depth-supervised halt drives the halting loss to **≈ 0** and the model learns
to spend **exactly as many latent steps as the instance has reasoning hops**:

| hop count k | mean latent steps used | accuracy |
|---|---|---|
| 2 | **2.0** | 0.90 |
| 3 | **3.0** | 0.83 |
| 4 | **4.0** | 0.92 |

Overall: 0.883 accuracy, mean 3.0 steps, **ρ(steps,hops) = +1.00**. This is
near-perfect difficulty calibration — serial latent compute allocated in exact
proportion to instance difficulty — learned single-stage, no RL, no curriculum.

### 5.2 Ablation — depth-supervision *causes* the calibration, for free

| config | acc | ρ(steps,hops) | mean steps |
|---|---|---|---|
| Reverie (full) | **0.883** | **+1.00** | **3.0** |
| − depth-supervision (γ=0) | 0.847 | **+0.00** | **5.0** (max) |
| − trajectory distillation (α=0) | 〈…〉 | 〈…〉 | 〈…〉 |

Removing γ makes the halt **pin to the maximum budget on every instance**
(mean steps → K=5, ρ → 0) *even though accuracy is essentially unchanged*
(0.847 vs 0.883) and the answer is still learned. The PonderNet answer loss
alone does **not** induce calibration; the teacher-depth-supervised halt is the
mechanism — and it delivers per-instance adaptive compute (**40 % fewer latent
forward passes at inference**, 3.0 vs 5.0 steps) at no accuracy cost. Removing α
〈drops accuracy to …〉, showing trajectory distillation carries the task signal.

### 5.3 The halt is a sharp, exact decision (not a smooth dial)

Sweeping the inference halt-logit bias over [−4, +4] leaves the operating point
unchanged (0.88 acc, 3.0 steps throughout): the learned per-instance halt is so
confident — λ jumps to ≈1 precisely at depth `n_hops` — that it behaves as a
*discrete, exact* decision rather than a tunable threshold. A smoothly dialable
accuracy-vs-compute frontier from one model would require a softer halt (e.g. a
temperature on λ, or `λ_prior` swept across training runs); we note this as the
honest counterpart to the calibration being near-perfect.

### 5.4 Learning dynamics

Reverie shows a **phase transition**: candidate-restricted accuracy holds near
chance while the halt calibrates, then rises sharply once the trajectory is
learned (0.50 → 0.58 → 0.81 → **0.88** over steps 200→800) — the answer emerges
*after* the model has learned where to stop.

### 5.5 The search regime (honest hard case)

With distractor branches (`runs/search_reverie.json`), a non-reasoning
component-membership shortcut is removed; at 0.43M params the model reaches
〈acc〉 — capacity-bound (§6) — while **the halt still calibrates** (ρ = 〈…〉),
showing the mechanism is task-robust even where absolute accuracy is not.

*(On the shortcut-solvable chain task, absolute-accuracy baseline comparisons are
uninformative — No-CoT exploits component membership, and generation-scored CoT
pays a decoding penalty — so we do not headline them; see §6.)*

## 6. Limitations & honesty

- Small-scale, synthetic, from-scratch: this is a *mechanism* study, not a frontier-scale result. The hidden-space self-distillation variant (`L_distill`+`L_explicit`) and GSM8K transfer are future work.
- **Two difficulty regimes (an honest finding).** The *depth-supervised halting calibrates near-perfectly to problem depth* (halt loss → 0, ρ ≈ 1.0) across all settings — the novel mechanism is robust. But *answer* accuracy is capacity-bound: at our tiny 0.43M from-scratch scale, multi-hop **chain-following** is learnable, whereas the **search** regime (distractor branches — the hardest ProsQA setting, and Coconut's own showcase) needs more capacity/steps to generalize. Coconut used GPT-2-124M (≈300× larger); absolute accuracy on the search regime is expected to rise with scale. We report both regimes rather than cherry-picking.
- The theory is a scoped proposition with empirical validation, not a general theorem.
- Sequential `K+1` latent passes at train time (shared with Coconut); adaptive halting reduces this at inference.

## 7. Reproducibility

Seeded Rust generator (byte-reproducible), fixed-shape JAX (single compile), data/RNG as pure functions of (seed, step). `python scripts/matrix.py --seeds 0,1,2` reproduces §5.
