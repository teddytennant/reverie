#!/usr/bin/env python
"""Run the full method comparison matrix and emit a markdown results table.

Trains every method (No-CoT, CoT, Coconut, Coconut+distill, Reverie) at matched
compute across one or more seeds, aggregates test accuracy / latent-steps-used /
calibration ρ, and writes both a JSON blob and a markdown table.

    python scripts/matrix.py --steps 3000 --hops-mix 2,3,4,5 --seeds 0,1,2

Runs are subprocesses (fresh JAX state each) so they cannot leak into one
another; results are collected from each run's --out JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METHODS = ["nocot", "cot", "coconut", "coconut_distill", "reverie"]


def run_one(method, seed, common, out_path):
    cmd = [sys.executable, os.path.join(HERE, "run.py"),
           "--method", method, "--seed", str(seed), "--out", out_path] + common
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print(f"  -> {method} seed={seed} ...", flush=True)
    r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit(f"run failed: {method} seed={seed}")
    with open(out_path) as f:
        return json.load(f)


def agg(vals):
    m = statistics.mean(vals)
    s = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--hops-mix", default="2,3,4,5")
    ap.add_argument("--branch", type=int, default=2)
    ap.add_argument("--trap-depth", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--outdir", default="runs")
    args = ap.parse_args()

    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    common = ["--steps", str(args.steps), "--hops-mix", args.hops_mix,
              "--branch", str(args.branch), "--trap-depth", str(args.trap_depth),
              "--n-train", str(args.n_train), "--n-val", str(args.n_val),
              "--n-test", str(args.n_test), "--d-model", str(args.d_model),
              "--layers", str(args.layers), "--heads", str(args.heads),
              "--max-steps", str(args.max_steps), "--batch-size", str(args.batch_size),
              "--lr", str(args.lr), "--eval-every", str(max(500, args.steps))]
    os.makedirs(os.path.join(ROOT, args.outdir), exist_ok=True)

    t0 = time.time()
    results = {}
    for method in methods:
        runs = []
        for seed in seeds:
            out = os.path.join(ROOT, args.outdir, f"{method}_s{seed}.json")
            runs.append(run_one(method, seed, common, out))
        results[method] = runs

    # markdown table
    rows = []
    for method in methods:
        accs = [r["test"]["acc"] for r in results[method]]
        steps_ = [r["test"]["mean_steps"] for r in results[method]]
        rhos = [r["test"]["rho_steps_hops"] for r in results[method]]
        am, asd = agg(accs)
        sm, _ = agg(steps_)
        rm, _ = agg(rhos)
        rows.append((method, am, asd, sm, rm, results[method][0]["test"]["acc_by_hop"]))

    hopset = sorted({int(k) for _, _, _, _, _, abh in rows for k in abh})
    hdr = ["method", "acc", "±", "mean_steps", "ρ(steps,hops)"] + [f"acc@k={k}" for k in hopset]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    for method, am, asd, sm, rm, abh in rows:
        cells = [method, f"{am:.3f}", f"{asd:.3f}", f"{sm:.2f}", f"{rm:+.2f}"]
        cells += [f"{abh.get(str(k), abh.get(k, float('nan'))):.2f}" for k in hopset]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    print("\n" + table)
    blob = os.path.join(ROOT, args.outdir, "matrix.json")
    with open(blob, "w") as f:
        json.dump({"args": vars(args), "results": results, "table": table}, f, indent=2)
    with open(os.path.join(ROOT, args.outdir, "matrix_table.md"), "w") as f:
        f.write(f"# Reverie — results matrix\n\n{table}\n")
    print(f"\nwrote {blob} | total wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
