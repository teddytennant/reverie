# Reverie

Latent reasoning that decides how long to think, and gets it exactly right.

[Coconut](https://arxiv.org/abs/2412.06769) feeds the last hidden state back in as the next input embedding. Reverie supervises every thought to decode to its gold reasoning step, and lets a PonderNet halt pick how many thoughts each problem gets. One loss, one stage, no RL.

```
L =  Σₙ pₙ · CE(answer, W yₙ)          # PonderNet answer loss
  +  α · Σᵢ CE(path[i], W yᵢ)          # every thought decodes to its gold step
  +  γ · (−log p_m)                    # halt at the teacher's hop count
  +  β · KL(p ‖ Geometric(λ_prior))    # don't collapse
```

γ is the point. The generator knows how many hops each problem needs, so train the halt against that. One MLE term.

## Run

```bash
cargo build --release --manifest-path data-gen/Cargo.toml
uv venv .venv && uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python scripts/run.py --method reverie --steps 3000 --hops 4
.venv/bin/python scripts/run.py --method coconut --steps 3000 --hops 4   # baseline

.venv/bin/python -m pytest -q
```

0.43M params, from scratch. Binary choice, so chance is 0.5. Rust generates the graphs, JAX and Equinox do the rest. Method and experimental design in [docs/DESIGN.md](docs/DESIGN.md).
