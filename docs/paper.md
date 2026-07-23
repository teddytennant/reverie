# Reverie: Difficulty-Calibrated Latent Reasoning via Depth-Supervised Halting

*Working paper draft. Reproduce every number with `make phase0` (`scripts/phase0.sh` + `scripts/ablation_table.py`).*

## Abstract

Coconut reasons in continuous latent space, but trains with a brittle multi-stage curriculum, supervises none of its thoughts, and spends a fixed latent budget on every problem. **Reverie** is a single-stage, RL-free method that (i) distills a discrete reasoning trajectory into every continuous thought, and (ii) chooses thought count with a differentiable PonderNet-style halt targeted at the teacher's per-instance depth.

Central result: the depth-supervised halt makes latent compute track problem difficulty. On self-generated, BFS-verified reasoning graphs, a 0.43M from-scratch model hits **Spearman ρ ≈ 1.0** (steps vs hops) with halting loss ≈ 0, entirely single-stage. Ablating depth-supervision drops ρ from +1.00 to +0.00 (halt pins to max budget: 5.0 steps everywhere) at no accuracy cost (0.883 vs 0.887). With it, the model spends exactly `n_hops` steps (2→2.0, 3→3.0, 4→4.0): a 40% inference-compute saving. The learned halt is sharp enough to act as an exact per-instance decision.

We state a proposition that per-instance adaptive latent depth is necessary to match explicit reasoning at `E[depth]` compute on heterogeneous-depth distributions. At 0.43M params the accuracy comparison to CoT/No-CoT is confounded (easy shortcuts; search regime is capacity-bound), so we foreground the mechanism and release the method, a JAX/Equinox implementation, and a zero-dep Rust generator.

## 1. Introduction

- Latent reasoning and Coconut's three costs (curriculum, no thought supervision, fixed budget).
- Contribution: fused objective (trajectory distillation + depth-supervised differentiable halt) in one stage; compute calibrated to difficulty (steps = n_hops); variable-serial-depth framing.
- No single ingredient is new (Coconut, CCoT, CODI, PonderNet, RL-halt). The fusion, deliverable, and theory are.

## 2. Method

**Substrate.** Decoder-only transformer `f_θ` (RoPE, RMSNorm, SwiGLU; JAX/Equinox). Tied head `W`. Continuous thought = last-layer hidden state fed back, unprojected, as the next input embedding (Coconut's mechanism).

**Latent unroll (static-shape, one compile).** From a left-padded prompt of length `Sp`, append `K` thought slots. Thought `t` (column `Sp+t`) consumes the hidden at `Sp+t-1`. One length-`K` `lax.scan` yields every intermediate read-out `y_0..y_K`. Answer after `m` thoughts is `W y_m`; all `K+1` depths in one batched matmul.

**Objective (single stage, no curriculum, no RL).** Halting distribution `pₙ = λₙ∏_{j<n}(1−λ_j)` over `n∈{0..K}`, teacher depth `m = n_hops`, gold node `k_j` at hop `j`:

```
L =  Σₙ pₙ·CE(answer, W yₙ)      # PonderNet answer loss
  +  α·Σⱼ CE(k_j, W y_j)         # trajectory distillation
  +  γ·(−log p_m)                # halt at teacher depth
  +  β·KL(p ‖ Geometric(λ_p))    # anti-collapse prior
```

Trajectory term is in **output space** (each thought decodes via the tied head to its reasoning step): param-free and doubles as an interpretability probe. Inference stops when cumulative halt mass crosses a budget.

## 3. Related work and novelty

Ingredients: continuous thoughts (Coconut), trajectory distillation (CCoT), single-stage self-distillation (CODI), differentiable halting (PonderNet), per-instance latent halt (2511.21581, via RL). **The fusion is new:** one stage that distills the full teacher trajectory into every thought **and** sets chain length with a differentiable geometric-prior halt targeted at teacher depth. No RL, no post-hoc classifier, no staging.

| Method | Supervised latents? | Adaptive length? | Single-stage? | vs Reverie |
|---|---|---|---|---|
| Coconut | no (answer only) | no (fixed) | no (curriculum) | no distillation; fixed depth |
| CCoT | trajectory (teacher hidden) | classifier, fixed ratio | no | bolted-on halt; no depth target |
| ICoT-KD / SI | teacher hidden / none | no | no / curriculum | not adaptive latent count |
| Quiet-STaR | reward only, discrete | no | pretraining | discrete; high-variance RL |
| PonderNet | no | yes (diff. halt) | yes | no content / depth supervision |
| CODI | single anchor | no (fixed 6) | yes | one anchor ≠ trajectory; no halt |
| Learning-When-to-Stop | no (answer reward) | yes (RL/PPO) | bolted on | RL vs distillation-native halt |
| **Reverie** | **every thought ← teacher step** | **diff. halt by teacher depth** | **one stage** | **fused objective + steps = n_hops** |

Claim: a curriculum-free, RL-free model that spends latent compute calibrated to difficulty via a teacher-depth-supervised halt, with a proposition that adaptive depth is necessary on heterogeneous-depth distributions. (See [`docs/DESIGN.md`](DESIGN.md) §3.3.)

## 4. Experimental setup

- **Task:** ProsQA-style DAG planning (Rust generator, BFS-verified, fictional tokens). Difficulty dial = hop count `k`.
- **Model:** from-scratch decoder-only, 0.43M params (d=128, 2 layers, 4 heads, K=5), JAX/Equinox.
- **Baselines (matched compute):** No-CoT, CoT, Coconut (fixed-depth, answer-only), Coconut+distill, Reverie.
- **Metrics:** candidate-restricted accuracy (binary C₁/C₂; chance 0.5), mean latent steps, ρ(steps, hops), halt-bias Pareto, seed stability.

All numbers: single seed; test set hops {2,3,4}.

### 5.1 Central result: latent compute equals reasoning depth

Halting loss → ≈ 0; steps match hops exactly:

| hop count k | mean latent steps | accuracy |
|---|---|---|
| 2 | **2.0** | 0.90 |
| 3 | **3.0** | 0.83 |
| 4 | **4.0** | 0.92 |

Overall: 0.883 acc, mean 3.0 steps, **ρ = +1.00**. Single-stage, no RL, no curriculum.

### 5.2 Ablation: depth-supervision causes calibration, for free

| config | acc | ρ(steps,hops) | mean steps |
|---|---|---|---|
| Reverie (full) | 0.883 | **+1.00** | **3.0** ({2:2, 3:3, 4:4}) |
| − depth-supervision (γ=0) | 0.887 | **+0.00** | **5.0** ({2:5, 3:5, 4:5}) |
| − trajectory distillation (α=0) | 0.905 | **+1.00** | **3.0** ({2:2, 3:3, 4:4}) |

Drop γ → halt pins to max budget (K=5 everywhere, ρ→0) while accuracy is unchanged. PonderNet answer loss alone does not induce calibration; γ does, and saves 40% latent passes at inference (3.0 vs 5.0). Drop α → accuracy and calibration hold; α's job is interpretability (force each latent to decode to its step), not accuracy. Orthogonal knobs: γ buys calibrated compute, α buys decodable latents; neither costs accuracy.

### 5.3 Halt is a sharp decision, not a smooth dial

Sweeping halt-logit bias over [−4, +4] leaves the operating point fixed (0.88 acc, 3.0 steps). λ jumps to ≈1 exactly at `n_hops`, so the halt is discrete, not a tunable threshold. A smooth accuracy-vs-compute frontier would need a softer halt (temperature on λ, or `λ_prior` swept across runs).

### 5.4 Learning dynamics

Phase transition: accuracy holds near chance while the halt calibrates, then rises once the trajectory is learned (0.50 → 0.58 → 0.81 → **0.88** over steps 200→800). The answer arrives after the model learns where to stop.

### 5.5 Search regime (distractor branches)

With `--branch 1 --trap-depth 1`, every hop has a trap edge. Reverie reaches **0.85 acc with the same exact calibration** (ρ = +1.00, steps = {2:2, 3:3, 4:4}). Calibration holds on easy chains and hard search.

| method (search) | acc | ρ(steps,hops) | mean steps |
|---|---|---|---|
| Reverie | 0.850 | **+1.00** | 3.0 latent |
| No-CoT | 0.847 | +0.00 | **0.0** |
| CoT (generation-scored) | 0.647 | – | 28 decoded |

**No-CoT matches Reverie with zero reasoning steps.** Generation-scored CoT trails (decode errors over a 28-token chain). The generator's disjoint-component decoy leaves a shortcut, so this task does not require reasoning. That makes calibration a clean demonstration: the model spends exactly `n_hops` steps even though the steps buy no accuracy. Calibration tracks depth, not correctness.

## 6. Limitations

- Small-scale, synthetic, from-scratch: mechanism study, not frontier scale. Hidden-space self-distillation and GSM8K transfer are future work.
- **No accuracy advantage at this scale (probed negative).** Calibration is solid across chain (0.88) and search (0.85). Accuracy does not separate: No-CoT 0.847 ≈ Reverie 0.850 on search; CoT trails at 0.647. Component-membership shortcut test (`--connect` cross-edges) left No-CoT unbroken:

| decoy→source cross-edges | 0 | 12 | 24 |
|---|---|---|---|
| No-CoT test accuracy | 0.847 | 0.882 | 0.940 |

A 2-layer transformer solves directed reachability over these small graphs in one pass, so multi-step latent reasoning is not needed for accuracy. An accuracy edge needs deeper problems than one pass can compute, plus GPT-2 scale and a near-miss decoy generator. Future work. We headline the mechanism (calibration), which no shortcut explains: the halt learns depth, not the answer.
- Theory is a scoped proposition with empirical validation, not a general theorem.
- Train time still pays sequential `K+1` latent passes (shared with Coconut); adaptive halt reduces this at inference.

## 7. Reproducibility

Seeded Rust generator (byte-reproducible), fixed-shape JAX (one compile), train/val/test held out by distinct seeds.

```bash
bash scripts/phase0.sh          # full Reverie + γ=0 / α=0 ablations + search
python scripts/ablation_table.py
```

Each run writes a self-describing JSON blob under `runs/`.
