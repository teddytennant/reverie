# Reverie

[Coconut](https://arxiv.org/abs/2412.06769)-style latent thoughts, plus a PonderNet halt aimed at the teacher's hop count. Every thought is trained to decode to its gold step. One loss, one stage, no RL.

Extra latent steps only help when the backbone cannot one-shot the task. On BFS-verified graph reachability, 0.43M params, ~600 seeds per cell:

- 1 layer, d=192: Reverie 0.864 vs No-CoT 0.761 (+44 sigma)
- 2 layers: Reverie loses at every width (−23 to −32 sigma)

A two-layer transformer resolves these graphs in one pass. Shrink it to one layer and the same extra thoughts start paying, and the edge grows with width.

The halt still pins. 2 hops get 2.0 steps, 3 get 3.0, 4 get 4.0, ρ = +1.00 on 606 seeds. Drop γ and it collapses to zero steps, which *helps* two-layer accuracy (8 sigma). Calibration is real. It costs.

Toy graphs, binary labels, chance = 0.5. Not language, not scale.

```
L =  Σₙ pₙ · CE(answer, W yₙ)          # PonderNet answer
  +  α · Σᵢ CE(path[i], W yᵢ)          # decode each thought
  +  γ · (−log p_m)                    # halt at teacher depth
  +  β · KL(p ‖ Geometric(λ_prior))    # don't collapse
```

```bash
make install
make demo      # ~8 min CPU
make test
```

`make phase0` is the original single-seed table. The numbers above are the H200 sweep in [docs/multiseed.md](docs/multiseed.md). Method in [docs/DESIGN.md](docs/DESIGN.md).
