# Multi-seed results

Still running. These are the numbers as of 2026-09-03, on commit cd00232 (the
embedding init fix). Cell counts are given per line because some grids are still
filling; anything here can move, and some of it already has, twice.

All runs use the `scripts/phase0.sh` configuration unless stated: 1000 steps,
hops 2,3,4, branch 0, trap-depth 0, d_model 128, 2 layers, 4 heads, max-steps 5,
batch 64, lr 3e-3. One run per seed, run on H200s.

Updated 2026-09-03 with a much larger sweep (n=54 for the method comparison,
n=72 for the capacity grid). Two things moved enough to say plainly: contrasts
that were noise at n=14 are now significant in the same direction, and the
capacity edge that looked like a window peaking at d=128 does not close by
d=160 the way it first appeared to. Read the capacity section below for what
changed and why the earlier framing was premature at n=12.

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
rate over {5e-4, 1e-3, 2e-3, 3e-3}. n=72 per cell at one layer, n=26 to 72 at two
(d=192 two-layer is still filling; everything else is done).

**One layer:**

| d_model | nocot best | reverie best | delta |
|---|---|---|---|
| 64 | 0.7258 @ 1e-3 | 0.6039 @ 2e-3 | -0.122 |
| 96 | 0.7325 @ 5e-4 | 0.8068 @ 3e-3 | +0.074 |
| 128 | 0.7670 @ 5e-4 | 0.8399 @ 3e-3 | +0.073 |
| 160 | 0.7578 @ 5e-4 | 0.8593 @ 2e-3 | +0.101 |
| 192 | 0.7557 @ 5e-4 | 0.8703 @ 2e-3 | +0.115 |

**Two layers:**

| d_model | nocot best | reverie best | delta |
|---|---|---|---|
| 64 | 0.8844 @ 3e-3 | 0.8508 @ 3e-3 | -0.034 |
| 96 | 0.8938 @ 3e-3 | 0.8586 @ 2e-3 | -0.035 |
| 128 | 0.8952 @ 2e-3 | 0.8649 @ 2e-3 | -0.030 |
| 160 | 0.8985 @ 2e-3 | 0.8679 @ 1e-3 | -0.031 |
| 192 | 0.8962 @ 1e-3 | 0.8764 @ 1e-3 | -0.020 (n=26) |

This changed shape from n=12 to n=72, not just tightened. At n=12 the one-layer
edge looked like a window: it peaked at d=128 (+0.107) and had mostly closed by
d=160 (+0.016), which read as "latent reasoning helps in a band, then the
advantage of extra width outpaces it." At n=72 that reading does not survive:
d=160 is +0.101 and d=192 is +0.115, the largest margin measured. Whatever
closed the gap in the n=12 data was sampling noise in four to eight seeds per
cell, not the start of a real decline. The honest current shape is that the
edge holds and mildly grows from d=96 through d=192, with d=64 as a floor below
which neither arm can do much (both near or below 0.73) and nocot wins by
process of elimination rather than by being good.

Two layers is now a clean, uniform negative result across every width measured:
reverie loses by 0.020 to 0.035 everywhere, the gap does not depend much on
d_model, and it is the mirror image of the one-layer story. Put together: at one
layer, where the architecture cannot solve the task in a single pass, adaptive
latent reasoning is worth ten-plus points of accuracy over the best-tuned
baseline. At two layers, where it can, that same reasoning is worth negative
three points. The mechanism in the README, that reasoning substitutes for depth
the architecture does not have, is the right frame; the two-layer table is what
makes it a real claim rather than a plausible one, since it shows the effect
reversing exactly where the mechanism predicts it should.

The learning rate finding that motivated this table still stands: a first pass
ran both arms at 3e-3 (phase0's default) and got a much larger and differently
shaped one-layer edge, because nocot was badly tuned at that rate and reaches
0.73 to 0.77 once given 5e-4. Comparing methods at one learning rate tuned for
neither was never a fair capacity result; this table is each arm at its own
best.
