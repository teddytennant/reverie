#!/usr/bin/env bash
# Phase-0 headline: multi-hop reachability (learnable at 0.43M from-scratch).
# Clean is-a chains (branch=0): No-CoT is bottlenecked to ~2 attention-hops and
# fails deeper instances, while CoT and the latent methods use sequential steps.
# Depth (hops 2-4) varies for the calibration + Pareto story.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--steps 1000 --hops-mix 2,3,4 --branch 0 --trap-depth 0 \
  --n-train 12000 --n-val 300 --n-test 400 \
  --d-model 128 --layers 2 --heads 4 --max-steps 5 --batch-size 64 --lr 3e-3"

# main comparison: No-CoT floor, CoT reference, Coconut (fixed, answer-only),
# Coconut+distill (fixed, trajectory), Reverie (adaptive+distill+depth-halt).
PYTHONUNBUFFERED=1 .venv/bin/python scripts/matrix.py \
  --methods nocot,cot,coconut,coconut_distill,reverie --seeds 0 $COMMON

# objective ablations (adaptive Reverie minus one term each)
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run.py --method reverie $COMMON \
  --alpha 0 --seed 0 --out runs/ablate_noTraj.json
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run.py --method reverie $COMMON \
  --gamma 0 --seed 0 --out runs/ablate_noDepthSup.json

# harder "search" regime (distractor branches) — expected to need more scale
PYTHONUNBUFFERED=1 .venv/bin/python scripts/run.py --method reverie \
  --steps 1000 --hops-mix 2,3,4 --branch 1 --trap-depth 1 \
  --n-train 12000 --n-val 300 --n-test 400 \
  --d-model 128 --layers 2 --heads 4 --max-steps 5 --batch-size 64 --lr 3e-3 \
  --seed 0 --out runs/search_reverie.json

.venv/bin/python scripts/report.py runs/matrix.json runs/reverie_s0.json
.venv/bin/python scripts/ablation_table.py
