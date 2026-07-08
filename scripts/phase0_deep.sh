#!/usr/bin/env bash
# Fallback if the 2-layer model can't crack multi-hop reachability: a deeper,
# wider model (more effective reasoning depth) at the cost of slower steps.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="--steps 1500 --hops-mix 2,3,4 --branch 1 --trap-depth 1 \
  --n-train 15000 --n-val 300 --n-test 400 \
  --d-model 160 --layers 4 --heads 4 --max-steps 5 --batch-size 48 --lr 3e-3"
PYTHONUNBUFFERED=1 .venv/bin/python scripts/matrix.py \
  --methods nocot,cot,coconut,coconut_distill,reverie --seeds 0 $COMMON
python scripts/report.py runs/matrix.json runs/reverie_s0.json
