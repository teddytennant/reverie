# Multi-seed results

Rebuilt 2026-09-11 from the full opportunist sweep on `runs/opp`: 79,527
finished runs, aggregated by `scripts/aggregate_opp.py` (new; nothing in the
repo did this before, the n=144 table two revisions back came from an ad hoc
process that was never committed). That script reads every run.py-shaped
JSON directly off the cluster and does the pooled two-sample arithmetic
itself, so the numbers below can be regenerated with:

```
ssh ncshare
python3 aggregate_opp.py --dir $HOME/reverie/runs/opp   # or scp it up first
```

Still running, still growing. Everything here is real numbers off real GPUs,
not projected.

All runs use the `scripts/phase0.sh` configuration unless stated: 1000 steps,
hops 2,3,4, branch 0, trap-depth 0, d_model 128, 2 layers, 4 heads, max-steps
5, batch 64, lr 3e-3. One run per seed, run on H200s.

## Method comparison, n=606

| method | acc | latent steps | rho(steps, hops) |
|---|---|---|---|
| nocot | 0.8947 ± 0.0146 | 0 | 0 |
| cot | 0.9124 ± 0.0466 | 28 | 0 |
| coconut | 0.8938 ± 0.0151 | 5 | 0 |
| coconut+distill | 0.8928 ± 0.0153 | 5 | 0 |
| reverie | 0.8863 ± 0.0183 | 2.998 | +1.000 ± 0.000 |

Contrasts, pooled two-sample:

| contrast | delta | sigma |
|---|---|---|
| cot vs reverie | +0.0261 | 12.84 |
| cot vs nocot | +0.0177 | 8.95 |
| reverie vs nocot | -0.0084 | 8.81 |
| reverie vs coconut | -0.0075 | 7.77 |

Same story as n=54, eleven times the seeds. Reverie is about 0.8 points
behind nocot and coconut now rather than 1.1-1.2; the gap shrank as more data
came in, the way the gamma/alpha ablation gaps below also shrank, but it
never crossed zero and the confidence only went up, 8-9 sigma on both
contrasts. cot still leads and the gap to it widened again, 12.8 sigma. rho
is still exactly +1.000, zero variance, across 606 seeds now instead of 14.

## Objective ablations, n=606

| arm | acc | latent steps | rho |
|---|---|---|---|
| reverie (full) | 0.8863 ± 0.0183 | 2.998 | +1.000 |
| alpha = 0 (no trajectory) | 0.8942 ± 0.0156 | 2.998 | +1.000 |
| gamma = 0 (no depth supervision) | 0.8945 ± 0.0152 | 0.000 | 0 |

| contrast | delta | sigma |
|---|---|---|
| alpha=0 vs full | +0.0079 | 8.08 |
| gamma=0 vs full | +0.0081 | 8.42 |

Both effects held up and both got smaller: +0.0127/+0.0125 at n=54 is now
+0.0079/+0.0081 at n=606. That is the regression-to-the-mean this project
keeps running into at low n, same shape as the n=14-to-n=54 move, and it is
exactly why the diagnosis below waits for real data instead of trusting the
first read. What did not move is the direction or the story: forcing the
halt onto the teacher's exact hop count, and forcing every latent step to
decode the teacher's path node, each cost accuracy on their own, now at 8
sigma instead of 3.8. Removing gamma still collapses the halt to zero steps,
not max depth (`mean_steps` above is exact 0.000, `rho` is undefined at zero
variance), matching the finding two revisions of this doc corrected the
README on.

Not yet run: alpha=0 and gamma=0 *together*, at this same phase0 config. The
two ablations were only ever run as separate arms; nothing in the sweep so
far says whether they stack, or whether one just subsumes the other. Queued
as `ablate_noTrajNoDepth_s*` (see Diagnosis and fix below).

## Capacity

Branch 1, trap-depth 1 (the harder regime), each arm at its own best
learning rate over {5e-4, 1e-3, 2e-3, 3e-3}. This is the `lrw` grid:
n=624 per (layers, d_model) cell at that cell's best-performing rate, out of
a target ceiling of 648 seeds per (layers, d_model, lr, method) cell that is
still filling in behind this snapshot. 74,787 of the 79,527 runs in this
sweep are this one grid.

New in this rebuild: coconut is in the table. It was always part of the
`lrw` family but the n=144 version of this doc never reported it. Coconut is
Reverie's own machinery minus every extra term, `adaptive=False` so there is
no PonderNet weighting and no gamma or alpha loss at all, just a fixed K=5
latent unroll scored by one cross-entropy at the end. It is the cleanest
available control for what alpha and gamma are costing.

**One layer**, sigma from a pooled two-sample test on reverie vs nocot, each
cell's own n:

| d_model | nocot best | reverie best | coconut best | delta (rev-noc) | sigma |
|---|---|---|---|---|---|
| 64 | 0.7246 ± 0.056 @ 1e-3 | 0.5992 ± 0.066 @ 2e-3 | 0.7730 ± 0.055 @ 2e-3 | -0.1254 | -36.3 |
| 96 | 0.7443 ± 0.051 @ 5e-4 | 0.7990 ± 0.047 @ 3e-3 | 0.8008 ± 0.041 @ 1e-3 | +0.0548 | +19.7 |
| 128 | 0.7646 ± 0.048 @ 5e-4 | 0.8425 ± 0.037 @ 3e-3 | 0.8424 ± 0.037 @ 1e-3 | +0.0779 | +32.4 |
| 160 | 0.7664 ± 0.047 @ 5e-4 | 0.8560 ± 0.027 @ 2e-3 | 0.8461 ± 0.044 @ 1e-3 | +0.0895 | +41.5 |
| 192 | 0.7605 ± 0.047 @ 5e-4 | 0.8636 ± 0.034 @ 2e-3 | 0.8519 ± 0.025 @ 5e-4 | +0.1031 | +44.2 |

**Two layers**:

| d_model | nocot best | reverie best | coconut best | delta (rev-noc) | sigma |
|---|---|---|---|---|---|
| 64 | 0.8832 ± 0.019 @ 3e-3 | 0.8455 ± 0.023 @ 3e-3 | 0.8798 ± 0.017 @ 3e-3 | -0.0377 | -32.0 |
| 96 | 0.8910 ± 0.016 @ 3e-3 | 0.8576 ± 0.020 @ 2e-3 | 0.8865 ± 0.016 @ 3e-3 | -0.0334 | -32.9 |
| 128 | 0.8917 ± 0.016 @ 2e-3 | 0.8611 ± 0.019 @ 1e-3 | 0.8869 ± 0.015 @ 2e-3 | -0.0305 | -31.3 |
| 160 | 0.8929 ± 0.015 @ 1e-3 | 0.8679 ± 0.018 @ 1e-3 | 0.8902 ± 0.015 @ 1e-3 | -0.0250 | -26.7 |
| 192 | 0.8939 ± 0.015 @ 1e-3 | 0.8716 ± 0.019 @ 1e-3 | 0.8913 ± 0.016 @ 1e-3 | -0.0223 | -22.9 |

Both shapes from the n=144 read survive fully filled: one-layer reverie
beats nocot at every width past 64 and the edge grows with width, up to 44
sigma at d=192; two-layer reverie loses to nocot at every width and the gap
shrinks with width, from -32.0 sigma at d=64 to -22.9 at d=192, still
monotonic, d=192 no longer an exception (it was the thin cell last time,
n=73; at n=624 it is the strongest two-layer row, not the weakest). The
sign-flip-somewhere-between-one-and-two-layers reading from the last update
stands.

What is new: coconut. At two layers coconut is within 0.002-0.005 of nocot
at every width, nowhere near reverie's 0.022-0.038 deficit; it is not
matching nocot's edge but it recovers nearly all of what reverie gives up.
At one layer coconut beats nocot too, by close to what reverie does at
d=96/128, and actually *edges out full reverie* at d=64 and d=96 (0.7730 vs
0.5992 at d=64; 0.8008 vs 0.7990 at d=96) before reverie pulls ahead again at
160 and 192. A method with no adaptive halting, no depth supervision and no
trajectory distillation, just a fixed extra K=5 latent passes, is closer to
reverie's own one-layer win and closer to nocot's two-layer floor than full
reverie is to either. That is strong indirect evidence for where the deficit
lives: not in giving the model extra latent compute, which coconut also
does, but in gamma and alpha's forcing of that compute onto a specific
schedule and a specific intermediate target.

## Diagnosis and fix (in progress)

Put the three results together:

1. Ablating gamma or alpha alone, each a *forcing* term (gamma pins the halt
   to the teacher's hop count, alpha pins every latent step's readout to the
   teacher's path node), improves accuracy on the easy branch=0/trap=0 task,
   8 sigma each at n=606.
2. Coconut, which has neither term and isn't even adaptive, sits much closer
   to nocot at two layers and close to or ahead of reverie at one layer,
   across the harder branch=1/trap=1 capacity grid.
3. The two-layer-loses/one-layer-wins split lines up exactly with the
   project's own earlier finding (README, the `--connect` shortcut probe)
   that a 2-layer transformer solves this reachability task in one forward
   pass. Where the backbone can already do the task without iterating,
   forcing it to also hit an exact per-example halt depth and decode every
   intermediate step to a specific graph node adds constraint with no
   compensating benefit, since there's no real multi-step computation for
   that constraint to be shaping. Where the backbone can't one-shot it
   (one layer), the extra latent passes are doing real work, and reverie
   wins, growing with width. Gamma and alpha look like a tax that is fixed
   regardless of whether the model needed the extra structure, so it costs
   least where compute was already being used well and shows up worst
   exactly where reverie should have no business losing to a model with no
   reasoning step at all.

The fix candidate this points to: keep reverie's adaptive PonderNet halting
(so it can still learn to spend less compute on easy instances) but drop
both forcing terms, alpha=0 and gamma=0 together, and see whether that
closes or reverses the two-layer gap without giving up the one-layer edge.
This is different from coconut: coconut is non-adaptive and always spends
the full K=5 steps; alpha=0/gamma=0 reverie is still adaptive, still has the
beta anti-collapse prior, it just answers to the task loss alone instead of
being pinned to the teacher's schedule.

**Status, updated 2026-09-13: the easy-task result is in and real; the
capacity-grid question this was actually for is still open.** The queue
described below cleared. `scripts/aggregate_opp.py` did not have a `fix`
family reader when this was first written despite an earlier draft of this
doc claiming it did; that was wrong and is fixed now, alongside adding
`ablation_noTrajNoDepth`.

On the phase0 (branch=0/trap=0) task, at n=542:

| arm | acc | n | latent steps | rho |
|---|---|---|---|---|
| reverie (full) | 0.8863 ± 0.0183 | 630 | 2.998 | +1.000 |
| alpha=0, gamma=0 (the fix) | 0.8955 ± 0.0137 | 542 | 0.009 | +0.008 |
| nocot | 0.8946 ± 0.0145 | 630 | 0.000 | +0.000 |
| coconut | 0.8937 ± 0.0151 | 630 | 5.000 | +0.000 |

The fix beats full reverie by +0.0092 (9.77 sigma, real) and is statistically
even with nocot (+0.0008, 1.01 sigma) and a little ahead of coconut (+0.0017,
2.08 sigma). But read `mean_steps` and `rho` before calling this a win for
reverie's mechanism: they collapsed to 0.009 and +0.008, the same shape as
the gamma=0-alone ablation, not the alpha=0-alone one (which kept
`mean_steps=2.998`, `rho=+1.000`). Without gamma pinning it to the teacher's
hop count, the halt just always fires almost immediately. The fix ties nocot
in accuracy by **behaving like nocot** (near-zero latent steps), not by
spending reverie's extra compute more usefully. That is a real, useful
result (it says the forcing terms were pure cost with nothing to show for it
on this task), but it is not evidence that reverie's continuous-latent
mechanism itself does anything nocot's zero-step baseline doesn't already do
here.

**Status, updated 2026-09-13 (later the same day): the capacity-grid
question is answered, and it changes the conclusion above.** The `fix_*`
family filled to its full ceiling, 960 runs (2 layers x 5 widths x 4 lrs x
24 seeds), read with `aggregate_opp.py`'s new per-cell-best-lr reader for
the `fix` family (it only read the easy-task `ablation_noTrajNoDepth` family
before; that gap is what the "not yet" above was waiting on).

**Two layers, each arm at its own best lr, n=24 for the fix, n=624 for the
others:**

| d_model | nocot best | reverie best | fix best | steps | rho | fix vs nocot | fix vs reverie |
|---|---|---|---|---|---|---|---|
| 64 | 0.8832 ± 0.019 | 0.8455 ± 0.023 | 0.8814 ± 0.014 | 0.20 | -0.02 | -0.0018 (-0.6σ) | +0.0358 (+12.1σ) |
| 96 | 0.8910 ± 0.016 | 0.8576 ± 0.020 | 0.8914 ± 0.014 | 0.01 | +0.02 | +0.0004 (+0.1σ) | +0.0338 (+11.7σ) |
| 128 | 0.8917 ± 0.016 | 0.8611 ± 0.019 | 0.8919 ± 0.013 | 0.79 | -0.13 | +0.0002 (+0.1σ) | +0.0307 (+11.5σ) |
| 160 | 0.8929 ± 0.015 | 0.8679 ± 0.018 | 0.8963 ± 0.013 | 0.28 | -0.05 | +0.0033 (+1.3σ) | +0.0283 (+10.4σ) |
| 192 | 0.8939 ± 0.015 | 0.8716 ± 0.019 | 0.8936 ± 0.015 | 0.00 | +0.01 | -0.0003 (-0.1σ) | +0.0220 (+7.2σ) |

The two-layer gap is gone, fully, at every width: the fix is a statistical
tie with nocot everywhere (|sigma| under 1.3) and beats full reverie by
7-12 sigma at every width. That confirms the diagnosis's prediction. But
`steps` is 0.00-0.79 at every width and `rho` is noise around zero (even
negative in two cells) -- same shape as the easy-task result above. The fix
closes the gap by behaving like nocot here too, not by using reverie's
mechanism.

**One layer, the regime where full reverie was actually winning (up to 45
sigma over nocot in the capacity table above) -- this is new, and it is not
a clean win:**

| d_model | nocot best | reverie best | fix best | steps | rho | fix vs nocot | fix vs reverie |
|---|---|---|---|---|---|---|---|
| 64 | 0.7255 ± 0.056 | 0.5993 ± 0.065 | 0.7065 ± 0.064 | 1.43 | -0.18 | -0.0191 (-1.4σ) | +0.1072 (+8.1σ) |
| 96 | 0.7441 ± 0.051 | 0.7988 ± 0.047 | 0.7350 ± 0.042 | 1.85 | -0.20 | -0.0091 (-1.0σ) | -0.0638 (-7.3σ) |
| 128 | 0.7644 ± 0.047 | 0.8425 ± 0.037 | 0.7617 ± 0.052 | 1.86 | -0.31 | -0.0027 (-0.3σ) | -0.0809 (-7.5σ) |
| 160 | 0.7661 ± 0.047 | 0.8560 ± 0.026 | 0.7407 ± 0.037 | 1.86 | -0.36 | -0.0254 (-3.3σ) | -0.1153 (-15.1σ) |
| 192 | 0.7603 ± 0.048 | 0.8636 ± 0.034 | 0.7449 ± 0.044 | 1.68 | -0.39 | -0.0154 (-1.7σ) | -0.1187 (-13.2σ) |

At d=64 the fix rescues reverie from its own anomalous collapse there (full
reverie inexplicably loses to nocot by 37 sigma at d=64, one layer; the fix
recovers most of that, +8.1 sigma over full reverie, though still not
significantly ahead of nocot). Everywhere else, d=96 through d=192, the fix
is a wash against nocot (ties or, at d=160, a real 3.3 sigma loss) and loses
hard to full reverie: 7.3 to 15.1 sigma, worse in absolute terms than the
two-layer win was good. `steps` is 1.4-1.9 here, not collapsed, but `rho` is
consistently negative (-0.18 to -0.39), meaning steps move slightly opposite
to hop count, not with it. Whatever the halt is doing at one layer without
gamma, it isn't collapsing to zero and it isn't tracking difficulty either.

**So the fix is not a fix, it's a trade.** Alpha and gamma together are a
tax at two layers (pure cost, gap closes to zero when dropped) and a
subsidy at one layer (dropping them gives back most of reverie's edge over
nocot, sometimes reversing it into a loss). Erasing both terms erases
reverie's mechanism everywhere, which is harmless where the mechanism
wasn't earning anything and actively bad where it was.

**Does forcing the halt away from collapse buy anything at two layers?**
Tested directly: `reverie/ensemble.py`, alpha=0, gamma=0, layers=2, d=128,
lr=1e-3 (the fix's own best lr for this cell), beta_reg swept over
{0.01, 0.1, 0.3, 1.0, 3.0} (`lambda_prior=0.2` unchanged), 24 seeds each,
branch=1/trap-depth=1:

| beta | acc | n | mean_steps | rho |
|---|---|---|---|---|
| 0.01 | 0.8929 ± 0.0133 | 24 | 0.77 ± 0.20 | -0.120 ± 0.072 |
| 0.1 | 0.8916 ± 0.0125 | 24 | 4.45 ± 0.06 | +0.001 ± 0.096 |
| 0.3 | 0.8893 ± 0.0141 | 24 | 4.39 ± 0.05 | +0.068 ± 0.068 |
| 1.0 | 0.8873 ± 0.0194 | 24 | 4.20 ± 0.05 | +0.056 ± 0.052 |
| 3.0 | 0.8933 ± 0.0178 | 24 | 4.07 ± 0.04 | +0.070 ± 0.051 |

Going from beta=0.01 to beta=0.1 (10x) is enough to blow the halt open:
mean_steps jumps from 0.77 (near-collapsed, matching the fix table above)
to 4.45 (near the geometric prior's target depth, close to K=5). Accuracy
does not move: 0.887-0.893 across the whole sweep, every value a tie with
nocot (|sigma| under 1.1) and 6.5-11.4 sigma ahead of full reverie,
regardless of whether the halt is collapsed or not. Two conclusions. First,
the two-layer regime is not merely indifferent to *forced* extra
computation, it is indifferent to *any* computation, collapsed or maxed
out; the backbone solves the task in one pass and nothing about the latent
steps matters there, which is the strongest confirmation yet of the
one-shot-in-two-layers reading. Second, and this is the real negative
result: `rho` never leaves the range +0.001 to +0.070 across nearly three
orders of magnitude of beta. Beta is a global depth-bias knob, not an
adaptivity mechanism -- it moves the mean of the halting distribution but
never makes it track per-instance difficulty. Cranking the anti-collapse
prior is not a route to real adaptive halting; only gamma's per-instance
supervision has ever produced nonzero rho in any run in this doc (+1.000,
every reverie-full row, phase0 task).

**Next step, queued and running, not yet read.** The combined fix can't say
which single term the one-layer edge depends on. `make-tasks.sh` now also
generates `fixA_l1_d{64..192}_{lr}_s{0..23}` (alpha=0 only, gamma kept) and
`fixG_l1_d{64..192}_{lr}_s{0..23}` (gamma=0 only, alpha kept), one layer
only since the two-layer question is answered by the beta sweep above and
doesn't need a term-by-term breakdown. 24 seeds/cell, same as the fix grid,
placed right after it in priority order. `aggregate_opp.py` has a reader
for both families (`## Capacity (fixA/fixG)`) ready for when they land. If
one term turns out to carry the one-layer edge on its own, dropping only
the other one might close the two-layer gap without the one-layer loss;
if both matter, there's no single-term fix and the real problem is that
gamma's exact per-instance pin is currently the only thing in this codebase
that produces adaptive (rho != 0) halting at all, and something coarser
than an exact hop-count target (whether to halt at all vs definitely not,
rather than the precise count) is the next real candidate to build.

## K-sweep: does step count itself help

Every comparison above confounds latent step count with either forced
scheduling (reverie's gamma/alpha) or a fixed default (coconut always runs
K=5, nothing in this doc had varied it). Isolated K on its own:
`scripts/ensemble.py --method coconut --adaptive 0 --alpha 0 --gamma 0
--max-steps K`, same recipe as coconut everywhere else in this doc (fixed
depth, answer-only cross-entropy, no PonderNet weighting, no trajectory
distillation, no depth supervision), only K varies. n=32 per K, seeds 0-31,
`runs/opp_ksweep` (`python3 scripts/aggregate_opp.py --dir runs/opp_ksweep`
regenerates the tables below).

Easy task (branch=0/trap=0, d=128, 2 layers):

| K | acc | n |
|---|---|---|
| 0 | 0.8955 ± 0.0137 | 32 |
| 1 | 0.8980 ± 0.0142 | 32 |
| 2 | 0.8945 ± 0.0161 | 32 |
| 3 | 0.8969 ± 0.0154 | 32 |
| 5 | 0.8966 ± 0.0161 | 32 |
| 8 | 0.8941 ± 0.0150 | 32 |
| 13 | 0.8940 ± 0.0155 | 32 |

Every adjacent contrast is under 1.1 sigma (K=1 vs K=0 is the largest, at
0.72). Flat from K=0 to K=13. On the easy task, at 2 layers, step count on
its own buys nothing, in either direction.

Hard task (branch=1/trap=1, 2 layers), each width at its own established
best coconut lr from the lrw grid above (d=64: 3e-3, d=128: 2e-3, d=192:
1e-3), so the recipe is held fixed within each width's row and only K moves:

**d_model=64**

| K | acc | n |
|---|---|---|
| 0 | 0.8825 ± 0.0159 | 32 |
| 1 | 0.8802 ± 0.0157 | 32 |
| 2 | 0.8792 ± 0.0149 | 32 |
| 3 | 0.8805 ± 0.0175 | 32 |
| 5 | 0.8800 ± 0.0179 | 32 |
| 8 | 0.8790 ± 0.0193 | 32 |

**d_model=128**

| K | acc | n |
|---|---|---|
| 0 | 0.8935 ± 0.0133 | 32 |
| 1 | 0.8972 ± 0.0131 | 32 |
| 2 | 0.8882 ± 0.0174 | 32 |
| 3 | 0.8909 ± 0.0170 | 32 |
| 5 | 0.8877 ± 0.0149 | 32 |
| 8 | 0.8873 ± 0.0178 | 32 |

**d_model=192**

| K | acc | n |
|---|---|---|
| 0 | 0.8945 ± 0.0137 | 32 |
| 1 | 0.8933 ± 0.0153 | 32 |
| 2 | 0.8959 ± 0.0156 | 32 |
| 3 | 0.8930 ± 0.0118 | 32 |
| 5 | 0.8921 ± 0.0140 | 32 |
| 8 | 0.8913 ± 0.0167 | 32 |

Same story at every width: every adjacent contrast under 2.4 sigma (the one
exception, K=2 vs K=1 at d=128, 2.33 sigma, sits inside an otherwise
non-monotonic, flat neighborhood and isn't the kind of effect that survives
a second look, the same lesson this doc has already paid for twice with
low-n reads). d=64, the width where full reverie inexplicably lost to nocot
by 37 sigma in the capacity table, isn't a K story either: 0.8790-0.8825
across the whole K=0..8 range, no trend.

**This closes the question this doc had been circling without asking
directly: at 2 layers, on this task family, latent step count has no
causal effect on accuracy, on either the easy or the hard regime, at any
width tested, once every forcing term is gone and the model is just told
to spend K steps and answer.** That is not evidence against continuous
latent reasoning in general. It's evidence that a 2-layer transformer on
multi-hop reachability already computes the whole answer inside the layers
it has, so handing it more sequence positions to compute over doesn't help,
because there's nothing left for those positions to contribute. It sharpens
the earlier README/`--connect` finding that 2 layers solves this task in
one forward pass, and it lines up with the beta sweep above, where accuracy
sat at 0.887-0.893 whether the halt collapsed to near-zero steps or was
forced open to near-K: the backbone doesn't care how many latent steps it's
given, forced or free, collapsed or not.

## Why everything before 2026-09-03 was re-run

`eqx.nn.Embedding` initialises to N(0, 1). With `tie_embeddings` that matrix
is also the LM head, so initial logits came out at a scale of sqrt(d_model):
step 0 cost around 130 nats per token on a 24-token vocab against
log(24) = 3.2, and the first few hundred updates went into shrinking the
embedding rather than learning.

Every arm paid it. Only cot showed it, because the other arms score a
two-way readout that saturates inside 1000 steps either way, while cot has
to free-run 26 tokens and was still mid-transition when training stopped.
Paired over 10 shared seeds, fixing the init moved cot by +0.2245 (worst
seed +0.405) and moved nocot, coconut and coconut_distill by 0.013 or less.
Everything in this doc is post-fix.
