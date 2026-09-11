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

**Status: queued, not yet resolved.** `make-tasks.sh` (the opportunist
daemon's task generator) was extended, additively, with two new task
families that ride the existing backfill-friendly scheduling instead of
competing with it for a separate allocation:

- `ablate_noTrajNoDepth_s*`: phase0 config, reverie, `--alpha 0 --gamma 0`
  together, same seed range as the existing single-term ablations. Direct
  answer to whether the two terms stack or one subsumes the other.
- `fix_l{1,2}_d{64,96,128,160,192}_{lr}_s{0..23}`: the same cells and
  learning rates as the `lrw` capacity grid above, reverie only, alpha and
  gamma both zeroed. n=24/cell target, placed ahead of the still-growing
  `lrw` grid in `make-tasks.sh`'s priority order since this is the
  experiment the diagnosis actually turns on, not another axis to fill in
  at leisure.

Both were submitted to the queue on 2026-09-11 along with a standalone
4-seed smoke job (`revfix-smoke`, job 729094, distinct name prefix, to be
reaped once it either runs or the investigation concludes) to sanity-check
`scripts/ensemble.py` on a real GPU before trusting it for the fix grid; it
ran and matched schema locally on CPU first. As of this writing every job
on the account, including 15-minute ones, is estimated `(Priority)` two
days out; the whole `gpu` partition is unusually saturated even by this
cluster's normal standard, not something this session caused or can route
around by resubmitting. `fix_*` and `ablate_noTrajNoDepth_*` have 0 runs
finished. Once the queue clears, this doc's next update should read
`runs/opp/fix_*` and `runs/opp/ablate_noTrajNoDepth_*` through
`scripts/aggregate_opp.py` (it already has a `fix` family; the `ablate_*`
family already exists) and report whether the two-layer gap actually closes.

Separately, `reverie/ensemble.py` (a vmapped multi-seed trainer, verified
against sequential training to <1e-4 after ten steps, see
`tests/test_core.py`) is committed and available if the opportunist queue
stays this congested: it can train a whole `fix_*` cell's 24 seeds in one
job instead of 24 separate queue entries, which matters when the bottleneck
is queue depth rather than GPU-hours.

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
