# Reverie: Difficulty-Calibrated Latent Reasoning via Depth-Supervised Halting

*Working paper draft. Every number below comes out of `make phase0` (`scripts/phase0.sh` + `scripts/ablation_table.py`).*

## Abstract

Coconut reasons in continuous latent space. It also trains through a brittle multi-stage curriculum, supervises none of its thoughts, and spends a fixed latent budget on every problem. Reverie is a single-stage, RL-free method that does two things instead: it distills a discrete reasoning trajectory into every continuous thought, and it picks the number of thoughts with a differentiable PonderNet-style halt targeted at the teacher's per-instance depth.

Depth supervision makes latent compute track problem difficulty. On self-generated, BFS-verified reasoning graphs, a 0.43M from-scratch model reaches Spearman ρ ≈ 1.0 between latent steps and hop count, with halting loss near zero, in one stage. Removing depth supervision moves ρ from +1.00 to +0.00; the halt then pins to the maximum budget, 5.0 steps on every instance, and accuracy does not move (0.883 vs 0.887). With depth supervision the model spends exactly `n_hops` steps (2 → 2.0, 3 → 3.0, 4 → 4.0), a 40% saving in inference compute. The halt is sharp enough to act as an exact per-instance decision.

I state a proposition that per-instance adaptive latent depth is necessary to match explicit reasoning at `E[depth]` compute on distributions with heterogeneous depth. At 0.43M params the accuracy comparison against CoT and No-CoT is confounded, since shortcuts are easy and the search regime is capacity-bound. The claim here is therefore about the mechanism. What ships alongside it is the method, a JAX/Equinox implementation, and a zero-dependency Rust generator.

## 1. Introduction

Latent reasoning replaces the discrete tokens of a chain of thought with continuous hidden states fed back as inputs. Coconut showed the mechanism works and left three costs behind. Language steps become latent slots one stage at a time, with optimizer resets, so training is a curriculum. Only the final answer carries gradient, so the latent trajectory is opaque. And the latent budget is fixed, so a two-hop problem pays the serial cost of the deepest problem in the distribution. Later work patches these one at a time.

Reverie folds all three into a single objective trained in one stage. Trajectory distillation supervises each continuous thought, through the tied output head, to decode to its gold reasoning step. A differentiable halt sets chain length, and its target is the teacher's per-instance depth rather than a global prior. The consequence I care about is the second-order one: the trained model spends latent compute in proportion to how deep the problem actually is, instead of spending the maximum everywhere.

None of the ingredients is new on its own. Continuous thoughts, trajectory distillation, single-stage self-distillation, differentiable halting, and a per-instance latent halt all exist in prior work. The fusion is what I claim, together with the released artifact and the variable-serial-depth framing that motivates it.

## 2. Method

The backbone is a decoder-only transformer `f_θ` (RoPE, RMSNorm, SwiGLU) written in JAX/Equinox, with a tied head `W`. A continuous thought is the last-layer hidden state fed back, unprojected, as the next input embedding. That is Coconut's mechanism, unchanged.

The unroll keeps static shapes so the model compiles once. From a left-padded prompt of length `Sp` I append `K` thought slots; thought `t` occupies column `Sp+t` and consumes the hidden at `Sp+t-1`. A single length-`K` `lax.scan` yields every intermediate read-out `y_0..y_K`. The answer after `m` thoughts is `W y_m`, and all `K+1` depths come out of one batched matmul rather than `K+1` decodes.

The objective is one stage, no curriculum, no RL. With halting distribution `pₙ = λₙ∏_{j<n}(1−λ_j)` over `n∈{0..K}`, teacher depth `m = n_hops`, and gold node `k_j` at hop `j`:

```
L =  Σₙ pₙ·CE(answer, W yₙ)      # PonderNet answer loss
  +  α·Σⱼ CE(k_j, W y_j)         # trajectory distillation
  +  γ·(−log p_m)                # halt at teacher depth
  +  β·KL(p ‖ Geometric(λ_p))    # anti-collapse prior
```

The trajectory term lives in output space: each thought decodes through the tied head to its reasoning step. That makes it parameter-free, and it doubles as an interpretability probe since the latents become linearly decodable. At inference the model stops once cumulative halt mass crosses the budget.

## 3. Related work and novelty

The ingredients are continuous thoughts (Coconut), trajectory distillation (CCoT), single-stage self-distillation (CODI), differentiable halting (PonderNet), and a per-instance latent halt learned by RL (2511.21581). What I have not found in prior work is the combination: one stage that distills the full teacher trajectory into every thought and sets chain length with a differentiable geometric-prior halt aimed at teacher depth. No RL, no post-hoc classifier, no staging.

| Method | Supervised latents? | Adaptive length? | Single-stage? | vs Reverie |
|---|---|---|---|---|
| Coconut | no (answer only) | no (fixed) | no (curriculum) | no distillation; fixed depth |
| CCoT | trajectory (teacher hidden) | classifier, fixed ratio | no | bolted-on halt; no depth target |
| ICoT-KD / SI | teacher hidden / none | no | no / curriculum | not adaptive latent count |
| Quiet-STaR | reward only, discrete | no | pretraining | discrete; high-variance RL |
| PonderNet | no | yes (diff. halt) | yes | no content or depth supervision |
| CODI | single anchor | no (fixed 6) | yes | one anchor ≠ trajectory; no halt |
| Learning-When-to-Stop | no (answer reward) | yes (RL/PPO) | bolted on | RL vs distillation-native halt |
| Reverie | every thought ← teacher step | diff. halt by teacher depth | one stage | fused objective + steps = n_hops |

The claim, then, is a curriculum-free and RL-free model that spends latent compute calibrated to difficulty via a teacher-depth-supervised halt, with a proposition that adaptive depth is necessary on heterogeneous-depth distributions. (See [`docs/DESIGN.md`](DESIGN.md) §3.3.)

## 4. Experimental setup

- Task: ProsQA-style DAG planning (Rust generator, BFS-verified, fictional tokens). The difficulty dial is hop count `k`.
- Model: from-scratch decoder-only, 0.43M params (d=128, 2 layers, 4 heads, K=5), JAX/Equinox.
- Baselines at matched compute: No-CoT, CoT, Coconut (fixed-depth, answer-only), Coconut+distill, Reverie.
- Metrics: candidate-restricted accuracy (binary C₁/C₂; chance 0.5), mean latent steps, ρ(steps, hops), halt-bias Pareto, seed stability.

## 5. Results

Everything below is a single seed, with test hops drawn from {2,3,4}.

### 5.1 Latent compute equals reasoning depth

Halting loss goes to ≈ 0 and steps match hops exactly:

| hop count k | mean latent steps | accuracy |
|---|---|---|
| 2 | 2.0 | 0.90 |
| 3 | 3.0 | 0.83 |
| 4 | 4.0 | 0.92 |

Overall the model gets 0.883 accuracy at a mean of 3.0 steps, with ρ = +1.00. One stage, no RL, no curriculum.

### 5.2 Ablation: depth supervision is what causes the calibration

| config | acc | ρ(steps,hops) | mean steps |
|---|---|---|---|
| Reverie (full) | 0.883 | +1.00 | 3.0 ({2:2, 3:3, 4:4}) |
| − depth-supervision (γ=0) | 0.887 | +0.00 | 5.0 ({2:5, 3:5, 4:5}) |
| − trajectory distillation (α=0) | 0.905 | +1.00 | 3.0 ({2:2, 3:3, 4:4}) |

Drop γ and the halt pins to the maximum budget, K=5 on every instance, with ρ collapsing to zero and accuracy unchanged. The PonderNet answer loss on its own does not induce calibration. γ does, and it cuts 40% of the latent passes at inference (3.0 vs 5.0). Drop α instead and both accuracy and calibration hold, which places α's job squarely in interpretability: it forces each latent to decode to its step, and it is not paying for accuracy. The two terms are orthogonal knobs. γ buys calibrated compute, α buys decodable latents, and neither costs accuracy here.

### 5.3 The halt is a discrete decision

Sweeping the halt-logit bias over [−4, +4] leaves the operating point where it was, at 0.88 accuracy and 3.0 steps. λ jumps to ≈1 exactly at `n_hops`, so the bias never moves the crossing point. This is the honest counterpart to near-perfect calibration: there is no tunable threshold to trade off against. A smooth accuracy-versus-compute frontier would need a softer halt, either a temperature on λ or `λ_prior` swept across retraining runs.

### 5.4 Learning dynamics

Training shows a phase transition. Accuracy sits near chance while the halt calibrates, then climbs once the trajectory is learned: 0.50 → 0.58 → 0.81 → 0.88 over steps 200→800. The model works out where to stop before it works out the answer.

### 5.5 Search regime (distractor branches)

With `--branch 1 --trap-depth 1` every hop carries a trap edge. Reverie reaches 0.850 accuracy with the same exact calibration, ρ = +1.00 and steps = {2:2, 3:3, 4:4}. Calibration survives both the easy chains and the harder search.

| method (search) | acc | ρ(steps,hops) | mean steps |
|---|---|---|---|
| Reverie | 0.850 | +1.00 | 3.0 latent |
| No-CoT | 0.847 | +0.00 | 0.0 |
| CoT (generation-scored) | 0.647 | n/a | 28 decoded |

No-CoT matches Reverie while taking zero reasoning steps. Generation-scored CoT trails it, losing accuracy to decode errors over a 28-token chain. The generator's disjoint-component decoy leaves a shortcut open, so the task does not require reasoning to be solved. That makes the calibration result a cleaner one than it looks: the model spends exactly `n_hops` steps even though those steps buy no accuracy at all.

## 6. Limitations

This is a small, synthetic, from-scratch study of a mechanism. Nothing here runs at frontier scale. Hidden-space self-distillation and GSM8K transfer remain undone.

There is no accuracy advantage at this scale, and I went looking for one. Calibration is solid in both regimes, 0.883 on chains and 0.850 on search. Accuracy refuses to separate: No-CoT reaches 0.847 against Reverie's 0.850 on search, and CoT trails at 0.647. I suspected a component-membership shortcut and tried to close it by adding decoy-to-source cross-edges (`--connect`). The probe failed. No-CoT was not broken, it improved:

| decoy→source cross-edges | 0 | 12 | 24 |
|---|---|---|---|
| No-CoT test accuracy | 0.847 | 0.882 | 0.940 |

The explanation is that a 2-layer transformer solves directed reachability over graphs this small in a single pass, so multi-step latent reasoning is not needed to be accurate on them. An accuracy edge would need problems deeper than one pass can compute, GPT-2 scale, and a decoy generator that produces near misses instead of disjoint components. That is future work. What no shortcut explains is the calibration: the halt learns depth, not the answer.

The theory is a scoped proposition with empirical support, not a general theorem.

Training still pays `K+1` sequential latent passes, a cost shared with Coconut. Only inference gets cheaper from the adaptive halt.

## 7. Reproducibility

The Rust generator is seeded and byte-reproducible, the JAX side is fixed-shape and compiles once, and train, val and test are held out from each other by distinct seeds.

```bash
bash scripts/phase0.sh          # full Reverie + γ=0 / α=0 ablations + search
python scripts/ablation_table.py
```

Each run writes a self-describing JSON blob under `runs/`.
