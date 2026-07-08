#!/usr/bin/env bash
# Phase-0 headline experiment: method comparison on heterogeneous-depth ProsQA.
# CPU-friendly settings; ~10 min/method. Produces runs/matrix.{json,_table.md}.
set -euo pipefail
cd "$(dirname "$0")/.."

# Learnable regime: shallow branching so a small from-scratch model learns a
# *generalizable* reachability algorithm (not memorization). Depth still varies
# (hops 2-4) for the calibration story.
COMMON="--steps 1200 --hops-mix 2,3,4 --branch 1 --trap-depth 1 \
  --n-train 12000 --n-val 300 --n-test 400 \
  --d-model 128 --layers 2 --heads 4 --max-steps 5 --batch-size 64 --lr 3e-3"

# core comparison (No-CoT floor, CoT reference, Coconut baseline, Reverie ours)
PYTHONUNBUFFERED=1 .venv/bin/python scripts/matrix.py \
  --methods nocot,cot,coconut,coconut_distill,reverie --seeds 0 $COMMON

# ablations of the fused objective (single-seed, reuse settings)
for FLAG in "--alpha 0" "--gamma 0"; do
  name=$(echo "$FLAG" | tr -d ' -' )
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/run.py --method reverie $COMMON \
    $FLAG --seed 0 --out "runs/ablate_${name}.json"
done

# linear-chain control (branch=0, trap-depth=0): no search -> planning gap should
# collapse (Reverie ≈ CoT ≈ Coconut), ruling out "latent just helps everywhere".
for M in cot reverie; do
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/run.py --method $M --steps 1200 \
    --hops-mix 2,3,4 --branch 0 --trap-depth 0 --n-train 12000 --n-val 300 --n-test 400 \
    --d-model 128 --layers 2 --heads 4 --max-steps 5 --batch-size 64 --lr 3e-3 \
    --seed 0 --out "runs/control_${M}.json"
done

python scripts/report.py runs/matrix.json runs/reverie_s0.json
