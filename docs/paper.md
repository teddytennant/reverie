# Reverie: Difficulty-Calibrated Latent Reasoning via Depth-Supervised Halting

*Working paper draft. Numbers marked `〈…〉` are filled from `runs/matrix.json`.*

## Abstract

Reasoning in a continuous latent space (Coconut) lets a language model keep several candidate deductions alive at once, but it is trained by a brittle multi-stage curriculum, supervises none of its own thoughts, and spends a fixed, hand-set amount of latent compute on every problem regardless of difficulty. We introduce **Reverie**, a single-stage, reinforcement-learning-free method that (i) distills a discrete reasoning trajectory into *every* continuous thought, and (ii) chooses the number of thoughts per problem with a **differentiable PonderNet-style halt whose target is the teacher's own per-instance reasoning depth**. Because the halt is differentiable and depth-supervised, one trained model traces an accuracy-vs-latent-compute Pareto frontier by a single threshold, and the compute it spends is *calibrated to problem difficulty*. On a controlled planning benchmark (self-generated ProsQA-style graphs with dialable reasoning depth), Reverie 〈matches explicit chain-of-thought at fewer, adaptive latent steps〉, 〈its ε-frontier dominates Coconut's fixed operating point〉, and its latent budget correlates with ground-truth depth (Spearman ρ = 〈…〉). We give a proposition — validated on a depth-variance benchmark — that per-instance adaptive latent depth is *necessary* to match explicit reasoning at `E[depth]` compute on distributions with heterogeneous serial-reasoning depth.

## 1. Introduction

- Latent reasoning and why (superposition / implicit search), Coconut's three costs.
- Our contribution: the *fused* objective (trajectory distillation + depth-supervised differentiable halt) in one curriculum-free stage; the single-model difficulty-calibrated Pareto deliverable; the variable-serial-depth framing.
- Explicitly: no single ingredient is new (Coconut, CCoT, CODI, PonderNet, RL-halt); the fusion + deliverable + theory are.

## 2. Method

(See `docs/DESIGN.md §2`. Copy the substrate, the static-shape latent scan, the four losses, the efficient batched-over-depth read-out, and the inference-time ε halt.)

Objective actually trained:
```
L = Σₙ pₙ·CE(answer, W yₙ)  +  α·Σⱼ CE(k_j, W y_j)  +  γ·(−log p_m)  +  β·KL(p‖Geom(λ_p))
```

## 3. Related work & novelty

(Differentiation table from `DESIGN.md §3.1`: Coconut, CCoT, ICoT/SI, Quiet-STaR, PonderNet, CODI, Learning-When-to-Stop. Defensible claim = §3.3.)

## 4. Experimental setup

- **Task:** self-generated ProsQA-style DAG planning (Rust generator, BFS-verified labels, fictional tokens → no pretraining leakage). Difficulty dial = hop count `k`.
- **Model:** from-scratch decoder-only transformer (RoPE/RMSNorm/SwiGLU), 〈P〉M params, JAX/Equinox.
- **Baselines (matched compute):** No-CoT, CoT, Coconut (fixed-depth, answer-only ≈ w/o-curriculum), Coconut+distill (fixed-depth, trajectory-distilled), Reverie (ours).
- **Metrics:** ProsQA is a binary "Is E a C₁ or C₂?" question, so accuracy is **candidate-restricted** — of the two named candidates, which does the model's answer read-out prefer (chance = 0.5). Latent methods read out at the halted depth; CoT reads out at the answer position *after generating its own reasoning chain*. Also: mean latent steps used, ρ(steps, hops) calibration, single-model halt-bias Pareto, seed stability.

## 5. Results

### 5.1 Main comparison

〈paste `runs/matrix_table.md`〉

### 5.2 Difficulty calibration

〈mean latent steps vs hop count; Spearman ρ〉 — the model spends serial compute where the instance needs it.

### 5.3 Single-model Pareto frontier

〈acc vs mean latent steps as ε is swept (from `pareto` in the reverie run); Coconut is a single point, CoT another〉.

### 5.4 Depth sweep (the planning claim)

〈Reverie−CoT gap vs k; ≈0 at k=2, widening〉.

### 5.5 Ablations

〈no-distillation (α=0), no-halting (fixed depth), effect of γ (depth supervision), β/λ_p (compute prior)〉.

## 6. Limitations & honesty

- Small-scale, synthetic, from-scratch: this is a *mechanism* study, not a frontier-scale result. The hidden-space self-distillation variant (`L_distill`+`L_explicit`) and GSM8K transfer are future work.
- **Two difficulty regimes (an honest finding).** The *depth-supervised halting calibrates near-perfectly to problem depth* (halt loss → 0, ρ ≈ 1.0) across all settings — the novel mechanism is robust. But *answer* accuracy is capacity-bound: at our tiny 0.43M from-scratch scale, multi-hop **chain-following** is learnable, whereas the **search** regime (distractor branches — the hardest ProsQA setting, and Coconut's own showcase) needs more capacity/steps to generalize. Coconut used GPT-2-124M (≈300× larger); absolute accuracy on the search regime is expected to rise with scale. We report both regimes rather than cherry-picking.
- The theory is a scoped proposition with empirical validation, not a general theorem.
- Sequential `K+1` latent passes at train time (shared with Coconut); adaptive halting reduces this at inference.

## 7. Reproducibility

Seeded Rust generator (byte-reproducible), fixed-shape JAX (single compile), data/RNG as pure functions of (seed, step). `python scripts/matrix.py --seeds 0,1,2` reproduces §5.
