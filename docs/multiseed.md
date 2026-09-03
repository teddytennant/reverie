# Multi-seed results

Still running. These are the numbers as of 2026-09-02, on commit cd00232 (the
embedding init fix). Cell counts are given per line because some grids are still
filling; anything here can move.

All runs use the `scripts/phase0.sh` configuration unless stated: 1000 steps,
hops 2,3,4, branch 0, trap-depth 0, d_model 128, 2 layers, 4 heads, max-steps 5,
batch 64, lr 3e-3. One run per seed, run on H200s.

## Why everything was re-run

`eqx.nn.Embedding` initialises to N(0, 1). With `tie_embeddings` that matrix is
also the LM head, so initial logits came out at a scale of sqrt(d_model): step 0
cost around 130 nats per token on a 24-token vocab against log(24) = 3.2, and the
first few hundred updates went into shrinking the embedding rather than learning.

Every arm paid it. Only cot showed it, because the other arms score a two-way
readout that saturates inside 1000 steps either way, while cot has to free-run 26
tokens and was still mid-transition when training stopped. Paired over 10 shared
seeds, fixing the init moved cot by +0.2245 (worst seed +0.405) and moved nocot,
coconut and coconut_distill by 0.013 or less. Everything below is post-fix.

## Method comparison, n=14

| method | acc | latent steps | rho(steps, hops) |
|---|---|---|---|
| nocot | 0.8948 ± 0.0175 | 0 | 0 |
| cot | 0.9227 ± 0.0526 | 28 | 0 |
| coconut | 0.8955 ± 0.0144 | 5 | 0 |
| coconut+distill | 0.8938 ± 0.0126 | 5 | 0 |
| reverie | 0.8836 ± 0.0183 | 2.998 | +1.000 ± 0.000 |

Contrasts, as a difference in means over the pooled standard error:

| contrast | delta | sigma | |
|---|---|---|---|
| cot vs reverie | +0.0391 | 2.63 | significant |
| cot vs nocot | +0.0279 | 1.88 | noise |
| reverie vs nocot | -0.0112 | 1.66 | noise |
| reverie vs coconut | -0.0120 | 1.92 | noise |

Two things to say plainly. The four non-CoT arms are statistically
indistinguishable around 0.89, so "no accuracy edge" holds and is now a tight
bound rather than an absence of evidence. And explicit chain of thought is the
most accurate method here, beating reverie by 2.63 sigma. It costs 28 decode
steps to reverie's 2.998.

cot's spread is 0.0526, three times any other arm. It is the only method that
needs the full 1000 steps to converge, so it stays seed-sensitive.

## Halt calibration, n=14

| hops | latent steps used |
|---|---|
| 2 | 2.0000 ± 0.0000 |
| 3 | 3.0000 ± 0.0000 |
| 4 | 4.0000 ± 0.0000 |

rho = +1.000 ± 0.000 across all 14 seeds. Exact, not correlated, and zero
variance. This is the result that survived every change made today.

## Objective ablations, n=14

| arm | acc | latent steps | rho |
|---|---|---|---|
| reverie | 0.8836 ± 0.0183 | 2.998 | +1.000 |
| alpha = 0 (no trajectory) | 0.8950 ± 0.0124 | 2.998 | +1.000 |
| gamma = 0 (no depth supervision) | 0.8995 ± 0.0172 | 0.000 | 0 |

Dropping alpha costs nothing measurable: -0.0114 at 1.93 sigma, and calibration
is untouched. At this scale the trajectory term does no work.

Dropping gamma no longer pins the halt to max depth. It pins it to zero: the
model stops thinking entirely and gets +0.0159 accuracy for it, at 2.37 sigma.
That is a different behaviour from what the README describes and it needs a
rewrite. The honest reading is that on this task the halt is free to collapse in
either direction once nothing supervises it, and gamma is what makes it track
hop count rather than what makes it accurate.

## Capacity, in progress

One layer instead of two, on the harder task (branch 1, trap-depth 1), each arm
at its own best learning rate over {5e-4, 1e-3, 2e-3, 3e-3}. n=12 per cell.

| d_model | nocot best | reverie best | delta |
|---|---|---|---|
| 64 | 0.7085 @ 1e-3 | 0.5873 @ 2e-3 | -0.121 |
| 96 | 0.7515 @ 5e-4 | 0.8021 @ 3e-3 | +0.051 |
| 128 | 0.7404 @ 5e-4 | 0.8469 @ 3e-3 | +0.107 |
| 160 | 0.7535 @ 5e-4 | 0.7698 @ 5e-4 | +0.016 |

d=192 and the 2-layer half are still running.

The learning rate matters more than expected and it is why this table exists. A
first pass ran both arms at 3e-3, inherited from the phase0 config, and got
+0.267 at d=128 with nocot apparently pinned near chance. That was an artifact:
nocot wants 5e-4 and reaches 0.740 when it gets it. Comparing two methods at one
learning rate tuned for neither is not a capacity result.

What is left is a modest, width-dependent edge peaking near +0.11 at one layer,
and a more consistent pattern underneath it: nocot wants small learning rates
(5e-4 at three of four widths) while reverie prefers 3e-3 and tolerates a wider
range. Latent iteration may buy optimisation robustness more than accuracy.
