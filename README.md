# Reverie

**Adaptive, curriculum-free reasoning in a continuous latent space. A Coconut successor, in JAX.**

Coconut ([Hao et al., 2024](https://arxiv.org/abs/2412.06769)) reasons in continuous latent space by feeding the last-layer hidden state back as the next input embedding. It works, but pays three costs:

1. **Brittle multi-stage curriculum.** Language steps become latent slots one stage at a time, with optimizer resets. Adding several latents at once spikes the loss; SIM-CoT reports reasoning collapse to ~12.5% when latents scale.
2. **No thought supervision.** Only the final answer gives gradient, so the latent trajectory is opaque.
3. **Fixed latent budget.** Every problem pays for the deepest problem's serial compute.

**Reverie replaces all three with one differentiable loss, one stage, no RL.**

- **Trajectory distillation.** Every continuous thought is supervised, through the tied output head, to decode to its gold reasoning step. Full-trajectory supervision (vs CODI's single anchor), and latents that are linearly decodable.
- **Depth-supervised adaptive halting.** A PonderNet-style differentiable halt picks the number of thoughts per problem, targeted at the teacher's per-instance depth (`n_hops`). Not RL, not a bolted-on classifier.
- **Compute calibrated to difficulty.** The trained model spends exactly `n_hops` latent steps per problem (ρ(steps,hops) = +1.00). The halt is sharp enough to act as an exact per-instance decision.

Full method, novelty audit, and theory: [`docs/DESIGN.md`](docs/DESIGN.md).

## Objective

For an instance with gold path length `m = n_hops`, over latent depths `n ∈ {0..K}` with `pₙ = λₙ ∏_{j<n}(1−λⱼ)`:

```
L =  Σₙ pₙ · CE(answer, W yₙ)          # PonderNet expected answer loss
  +  α · Σᵢ CE(path[i], W yᵢ)          # trajectory distillation
  +  γ · (−log p_m)                    # halt at teacher depth
  +  β · KL(p ‖ Geometric(λ_prior))    # anti-collapse prior
```

Turning terms off recovers baselines: `α=0, fixed depth` → Coconut without curriculum; `α>0, fixed depth` → trajectory-distilled fixed-depth Coconut; `K=0` → No-CoT.

## Stack

- **JAX + Equinox + Optax** (same substrate as Levanter/Haliax; grok-1 was JAX).
- **Rust** (`data-gen/`). Zero-dep, deterministic, BFS-verified ProsQA-style generator. Rust owns the abstract problem (graph + gold path + `n_hops`); Python owns token rendering.

## Quickstart

```bash
# 1. build the data generator
cargo build --release --manifest-path data-gen/Cargo.toml

# 2. install the JAX stack
uv venv .venv && uv pip install --python .venv/bin/python -e .

# 3. train + evaluate
.venv/bin/python scripts/run.py --method reverie --steps 3000 --hops 4
.venv/bin/python scripts/run.py --method coconut --steps 3000 --hops 4   # baseline

# 4. tests
.venv/bin/python -m pytest -q
cargo test --manifest-path data-gen/Cargo.toml

# optional: short demo, or full Phase-0 table (matches Results below)
make demo
make phase0
```

`scripts/run.py` generates train/val/test with distinct seeds (hold-out by fresh seed), trains, and reports candidate-restricted accuracy (binary C₁/C₂; chance 0.5), mean latent steps, per-hop accuracy, and Spearman ρ(steps, depth). Adaptive runs also sweep `halt_bias` on the halt logit for a single-model Pareto curve.

## Results (0.43M from-scratch, CPU, candidate-restricted acc)

The model spends exactly as many latent steps as the problem has hops.

| hop count k | mean latent steps | accuracy |
|---|---|---|
| 2 | **2.0** | 0.90 |
| 3 | **3.0** | 0.83 |
| 4 | **4.0** | 0.92 |

Overall 0.883 acc, mean 3.0 steps, **ρ(steps, hops) = +1.00**, halting loss → 0. Single-stage, no curriculum, no RL.

**Ablation: depth-supervision (γ) causes the calibration, for free.**

| config | acc | ρ(steps,hops) | mean steps |
|---|---|---|---|
| Reverie (full) | 0.883 | **+1.00** | **3.0** |
| − depth-supervision (γ=0) | 0.887 | +0.00 | 5.0 (max) |
| − trajectory distillation (α=0) | 0.905 | +1.00 | 3.0 |

The two terms are orthogonal knobs, neither paid for in accuracy (0.88–0.90):
- γ buys difficulty-calibrated compute (drop it → halt pins to max depth, 40% more latent passes, ρ→0)
- α buys linearly-decodable latents (interpretability)

**Honest caveat.** Latent reasoning gives **no accuracy edge at this scale**. On the search task No-CoT hits 0.847 with zero reasoning steps, matching Reverie's 0.850. We tested a component-membership shortcut by adding decoy→source cross-edges (`--connect`); No-CoT stayed unbroken (0.847 → 0.882 → 0.940 for connect 0 → 12 → 24). Real reason: a 2-layer transformer solves directed reachability over these small graphs in one pass, so multi-step reasoning isn't needed for accuracy. An accuracy edge needs deeper problems and GPT-2 scale (future work). What no shortcut explains is the **calibration**: the halt spends exactly `n_hops` steps, a fact about depth, not the answer. Details: [`docs/paper.md`](docs/paper.md) §6. Reproduce: `make phase0`.

## Status

Research artifact. Method and experimental design in [`docs/DESIGN.md`](docs/DESIGN.md).
