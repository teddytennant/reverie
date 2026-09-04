# Multi-seed results

Still running. These are the numbers as of 2026-09-03, on commit cd00232 (the
embedding init fix). Cell counts are given per line because some grids are still
filling; anything here can move, and some of it already has, twice.

All runs use the `scripts/phase0.sh` configuration unless stated: 1000 steps,
hops 2,3,4, branch 0, trap-depth 0, d_model 128, 2 layers, 4 heads, max-steps 5,
batch 64, lr 3e-3. One run per seed, run on H200s.

Updated 2026-09-03 with a much larger sweep (n=54 for the method comparison,
n=144 for most of the capacity grid; d=192 two-layer is at n=73 and still
filling, everything else is done). The method comparison and ablation numbers
below are unchanged since the last update at n=54; the capacity section moved
again, from n=72 to n=144, and this time it added a real observation rather
than just tightening the same one. Read that section for what changed and why
the earlier n=12 framing was premature.

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

## Method comparison, n=54

| method | acc | latent steps | rho(steps, hops) |
|---|---|---|---|
| nocot | 0.8958 ± 0.0145 | 0 | 0 |
| cot | 0.9111 ± 0.0452 | 28 | 0 |
| coconut | 0.8961 ± 0.0135 | 5 | 0 |
| coconut+distill | 0.8921 ± 0.0149 | 5 | 0 |
| reverie | 0.8845 ± 0.0190 | 2.998 | +1.000 ± 0.000 |

Contrasts, as a difference in means over the pooled standard error:

| contrast | delta | sigma | |
|---|---|---|---|
| cot vs reverie | +0.0265 | 3.98 | significant |
| cot vs nocot | +0.0153 | 2.36 | significant |
| reverie vs nocot | -0.0112 | 3.46 | significant |
| reverie vs coconut | -0.0116 | 3.65 | significant |

This moved from the n=14 read. At n=14, reverie vs nocot and reverie vs coconut
were both noise (1.66 and 1.92 sigma) and the honest statement was "no accuracy
edge, in either direction, at this sample size." At n=54 the same-sized gaps are
now significant: reverie is about 1.1 to 1.2 points behind nocot and coconut,
reliably, not just on this draw. "No accuracy edge" was true as a statement
about statistical power, not about the underlying difference, and the
underlying difference turns out to be small but real and unfavorable to reverie
on this task.

cot remains the most accurate arm and the gap to it widened with more data,
now 3.98 sigma. Its spread is still three times any other arm (0.0452 vs
~0.015), consistent with it being the only method that needs the full 1000
steps to converge.

## Halt calibration, n=14

| hops | latent steps used |
|---|---|
| 2 | 2.0000 ± 0.0000 |
| 3 | 3.0000 ± 0.0000 |
| 4 | 4.0000 ± 0.0000 |

rho = +1.000 ± 0.000 across all 14 seeds. Exact, not correlated, and zero
variance. This is the result that survived every change made today.

## Objective ablations, n=54

| arm | acc | latent steps | rho |
|---|---|---|---|
| reverie | 0.8845 ± 0.0190 | 2.998 | +1.000 |
| alpha = 0 (no trajectory) | 0.8973 ± 0.0156 | 2.998 | +1.000 |
| gamma = 0 (no depth supervision) | 0.8970 ± 0.0149 | 0.000 | 0 |

Both ablations are now significant wins over full reverie: alpha=0 is +0.0127
at 3.81 sigma, gamma=0 is +0.0125 at 3.80 sigma. At n=14 these read as noise
(1.93 and 2.37 sigma); the direction did not change, the confidence did.

Dropping alpha costs nothing and, at this sample size, measurably helps. At this
scale the trajectory term is not just inert, it is a mild drag.

Dropping gamma does not pin the halt to max depth, the way an earlier version of
this doc and the README both said. It pins it to zero: the model stops thinking
entirely and that is now a significant accuracy gain, not a wash. The honest
reading is that on this task the halt is free to collapse in either direction
once nothing supervises it, gamma is what makes it track hop count rather than
what makes it accurate, and calibration is bought at a real, if small, accuracy
cost rather than for free.

## Capacity

Branch 1, trap-depth 1 (the harder regime), each arm at its own best learning
rate over {5e-4, 1e-3, 2e-3, 3e-3}. n=144 per cell except d=192 two-layer,
n=73 and still filling; everything else is done.

**One layer**, sigma from a pooled two-sample test on each cell's own n:

| d_model | nocot best | reverie best | delta | sigma |
|---|---|---|---|---|
| 64 | 0.7277 ± 0.057 @ 1e-3 | 0.6026 ± 0.074 @ 2e-3 | -0.125 | -16.1 |
| 96 | 0.7357 ± 0.051 @ 5e-4 | 0.8041 ± 0.049 @ 3e-3 | +0.068 | +11.6 |
| 128 | 0.7638 ± 0.049 @ 5e-4 | 0.8385 ± 0.047 @ 3e-3 | +0.075 | +13.3 |
| 160 | 0.7610 ± 0.045 @ 5e-4 | 0.8562 ± 0.024 @ 2e-3 | +0.095 | +22.2 |
| 192 | 0.7571 ± 0.049 @ 5e-4 | 0.8645 ± 0.038 @ 2e-3 | +0.107 | +20.8 |

**Two layers**, n=144 except d=192 (n=73 reverie, n=120 nocot):

| d_model | nocot best | reverie best | delta | sigma |
|---|---|---|---|---|
| 64 | 0.8842 ± 0.019 @ 3e-3 | 0.8477 ± 0.023 @ 2e-3 | -0.037 | -14.7 |
| 96 | 0.8919 ± 0.016 @ 3e-3 | 0.8575 ± 0.019 @ 2e-3 | -0.034 | -16.7 |
| 128 | 0.8929 ± 0.016 @ 3e-3 | 0.8620 ± 0.021 @ 2e-3 | -0.031 | -14.2 |
| 160 | 0.8948 ± 0.017 @ 2e-3 | 0.8692 ± 0.019 @ 1e-3 | -0.026 | -11.8 |
| 192 | 0.8956 ± 0.016 @ 1e-3 | 0.8771 ± 0.019 @ 1e-3 | -0.019 | -7.0 |

The one-layer edge is unchanged in shape from n=72 to n=144, just more certain:
it rises from d=96 through d=192 rather than peaking and closing, and every
cell past d=64 clears 11 sigma. The n=12 reading, a window that peaks at d=128,
does not survive contact with more seeds at all.

What n=144 adds is new: the two-layer disadvantage is not flat across width,
it is shrinking. -0.037 at d=64 down to -0.019 at d=192, monotonic, and the
d=192 cell is markedly weaker than the rest at -7.0 sigma against -12 to -17
everywhere else in that row (that cell is also the one still at n=73, so watch
it, though the trend was already visible at n=72 before this update). Put next
to the one-layer table, which rises with width in the other direction, this
reads less like two separate regimes and more like one relationship: reverie's
disadvantage shrinks and its advantage grows as the model gets wider, and
something in between one and two layers is where the sign flips. That is a
sharper mechanistic claim than "helps at one layer, hurts at two," and it is
the kind of thing worth checking at layers in between if this project keeps
going.

The learning rate finding that motivated this table still stands: a first pass
ran both arms at 3e-3 (phase0's default) and got a much larger and differently
shaped one-layer edge, because nocot was badly tuned at that rate and reaches
0.73 to 0.77 once given 5e-4. Comparing methods at one learning rate tuned for
neither was never a fair capacity result; this table is each arm at its own
best.
