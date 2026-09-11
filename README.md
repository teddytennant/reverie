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

It works. 2 hops gets 2.0 steps, 3 gets 3.0, 4 gets 4.0, ρ = +1.00. Not "correlated". Exact, across 606 seeds now, not just one. Kill γ and the halt does not pin to max depth, it pins to zero: the model stops thinking entirely, and at that scale that's a real accuracy win over full Reverie (8 sigma), not a wash. Calibration costs something. See [docs/multiseed.md](docs/multiseed.md) for the current numbers and what's queued to fix it.

## What didn't work, and what did

No accuracy edge at the default 2-layer width. No-CoT gets 0.847 on the search task with zero reasoning steps, Reverie gets 0.850. I figured it was cheating off component membership and added cross-edges to force real directed reachability (`--connect`). No-CoT went *up*: 0.847 → 0.882 → 0.940 as cross-edges went 0 → 12 → 24. A 2-layer transformer resolves reachability on graphs this small in one pass, so nothing here needs to think twice, and the multi-seed sweep confirms it: Reverie loses to No-CoT at every width tested, 2 layers deep.

Shrink the backbone to one layer and the story flips. A 1-layer transformer can't one-shot this task, and Reverie beats No-CoT at every width past 64, growing from noise at d=96 to 44 sigma at d=192. The extra latent passes only pay for themselves where the backbone actually needs them. Full numbers, and a live diagnosis of why the forcing terms (γ, α) cost more than they should even where Reverie wins, in [docs/multiseed.md](docs/multiseed.md).

## Run

```bash
make install
make demo      # ~8 min CPU
make phase0    # the numbers above
make test
```

0.43M params, from scratch. Binary choice, so chance is 0.5. Rust generates the graphs, JAX and Equinox do the rest. Method in [docs/DESIGN.md](docs/DESIGN.md), numbers in [docs/paper.md](docs/paper.md).
