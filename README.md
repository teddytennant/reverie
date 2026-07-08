# Reverie

**Adaptive, curriculum-free reasoning in a continuous latent space — a successor to Coconut, in JAX.**

Coconut ([Hao et al., 2024](https://arxiv.org/abs/2412.06769)) showed that a transformer can reason in a *continuous latent space* by feeding its last-layer hidden state back as the next input embedding — no tokens decoded in between. It works, but it pays for it three times:

1. **A brittle multi-stage curriculum** — language reasoning steps are swapped for latent slots one stage at a time, with optimizer resets; adding several latents at once spikes the loss, and follow-up work (SIM-CoT) reports the reasoning pattern *collapses to ~12.5%* when latents are scaled.
2. **No supervision of the thoughts themselves** — only the final answer gives gradient, so the latent trajectory is un-audited and opaque.
3. **A fixed, hand-set number of latent steps** — every problem pays the *deepest* problem's serial-compute budget.

**Reverie replaces all three with a single differentiable loss, in one curriculum-free stage, with no reinforcement learning.**

- **Trajectory distillation.** Every continuous thought is supervised to decode, through the tied output head, to its gold reasoning step — full-trajectory supervision (vs CODI's single anchor), and latents that are *linearly decodable back to the reasoning* (interpretability for free).
- **Depth-supervised adaptive halting.** A PonderNet-style differentiable halt sets the number of thoughts per problem, with its target pinned to the **teacher's own per-instance reasoning depth** (`n_hops`) — not RL, not a bolted-on classifier.
- **One knob, one Pareto frontier.** Sweeping the halting threshold traces an accuracy-vs-latent-compute frontier from a *single* trained model, and because the halt is depth-supervised, the spent budget is **calibrated to problem difficulty**.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full method, novelty audit vs the latent-CoT landscape (Coconut, CCoT, ICoT, CODI, PonderNet, Quiet-STaR), and the variable-serial-depth theory.

## The objective

For each instance with gold path of length `m = n_hops`, over latent depths `n ∈ {0..K}` with halting distribution `pₙ = λₙ ∏_{j<n}(1−λⱼ)`:

```
L =  Σₙ pₙ · CE(answer, W yₙ)          # PonderNet expected answer loss
  +  α · Σᵢ CE(path[i], W yᵢ)          # trajectory distillation (decodability)
  +  γ · (−log p_m)                    # halt at the teacher's per-instance depth
  +  β · KL(p ‖ Geometric(λ_prior))    # compute prior / anti-collapse
```

Turning terms off recovers the baselines exactly: `α=0, fixed depth` → Coconut-without-curriculum; `α>0, fixed depth` → trajectory-distilled fixed-depth Coconut; `K=0` → No-CoT.

## Stack

- **JAX + Equinox + Optax** — the readable, reproducible research stack (also the base of Levanter/Haliax; xAI's grok-1 release was JAX).
- **Rust** (`data-gen/`) — a zero-dependency, deterministic, BFS-verified generator of ProsQA-style planning tasks. Rust owns the *abstract, verified problem* (graph + gold path + `n_hops`); Python owns token rendering.

## Quickstart

```bash
# 1. build the data generator
cargo build --release --manifest-path data-gen/Cargo.toml

# 2. install the JAX stack
uv venv .venv && uv pip install --python .venv/bin/python -e .

# 3. train + evaluate a method on self-generated ProsQA
.venv/bin/python scripts/run.py --method reverie --steps 3000 --hops 4
.venv/bin/python scripts/run.py --method coconut --steps 3000 --hops 4   # baseline

# 4. tests
.venv/bin/python -m pytest -q          # python
cargo test --manifest-path data-gen/Cargo.toml   # rust
```

`scripts/run.py` generates train/val/test with **distinct seeds** (hold-out by fresh seed — genuinely unseen test graphs), trains, and reports exact-match accuracy, mean latent steps used, per-hop accuracy, and the Spearman correlation between steps-used and problem depth.

## Preliminary findings (0.43M from-scratch, CPU)

The novel mechanism is robust across every setting tried: the depth-supervised
differentiable halt **calibrates near-perfectly to problem depth** — halting loss
drives to ≈0 and the latent steps used correlate with the ground-truth hop count
at **Spearman ρ ≈ 1.0** (the model spends serial latent compute exactly where the
instance needs it). On learnable multi-hop reachability, candidate-restricted
accuracy generalizes above chance and climbs with training; the harder *search*
regime (distractor branches, Coconut's own showcase) is capacity-bound at this
tiny scale — see [`docs/paper.md`](docs/paper.md) §6. Full matrix: `scripts/phase0.sh`.

## Status

Research artifact under active development. Method and experimental design in [`docs/DESIGN.md`](docs/DESIGN.md).
