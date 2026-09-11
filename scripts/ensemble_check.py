#!/usr/bin/env python
"""Compare two directories of run.py-shaped JSONs, seed by seed.

    python scripts/ensemble_check.py --a runs/seq --b runs/ens --method reverie

Used to check that scripts/ensemble.py reproduces scripts/run.py. Prints the
per-seed test accuracy from each side and the largest absolute difference over
every scalar in the blob, so a mismatch shows up as a number instead of a
feeling.
"""

from __future__ import annotations

import argparse
import json
import os


def scalars(blob: dict) -> dict:
    """Flatten the numeric fields a downstream aggregator actually reads."""
    t = blob["test"]
    out = {"test.acc": t["acc"], "test.mean_steps": t["mean_steps"],
           "test.rho_steps_hops": t["rho_steps_hops"], "test.n": t["n"]}
    for k, v in t["acc_by_hop"].items():
        out[f"acc_by_hop[{k}]"] = v
    for k, v in t["steps_by_hop"].items():
        out[f"steps_by_hop[{k}]"] = v
    for p in blob.get("pareto") or []:
        out[f"pareto[{p['halt_bias']}].acc"] = p["acc"]
        out[f"pareto[{p['halt_bias']}].mean_steps"] = p["mean_steps"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="directory of the reference JSONs")
    ap.add_argument("--b", required=True)
    ap.add_argument("--method", default="reverie")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--pattern", default="{method}_s{seed}.json")
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        pre, suf = args.pattern.format(method=args.method, seed="\x00").split("\x00")
        seeds = sorted(int(f[len(pre):-len(suf)]) for f in os.listdir(args.a)
                       if f.startswith(pre) and f.endswith(suf))

    worst = (-1.0, "", -1)  # -1 so an exact match still records which field it was
    print(f"{'seed':>5} {'a.acc':>9} {'b.acc':>9} {'|d|acc':>10} "
          f"{'max|d| any':>11}  worst field")
    for seed in seeds:
        name = args.pattern.format(method=args.method, seed=seed)
        a = json.load(open(os.path.join(args.a, name)))
        b = json.load(open(os.path.join(args.b, name)))
        sa, sb = scalars(a), scalars(b)
        keys = sorted(set(sa) | set(sb))
        assert set(sa) == set(sb), f"key mismatch on seed {seed}: {set(sa) ^ set(sb)}"
        diffs = [(abs(sa[k] - sb[k]), k) for k in keys]
        dmax, kmax = max(diffs)
        if dmax > worst[0]:
            worst = (dmax, kmax, seed)
        print(f"{seed:>5} {sa['test.acc']:>9.4f} {sb['test.acc']:>9.4f} "
              f"{abs(sa['test.acc']-sb['test.acc']):>10.2e} {dmax:>11.2e}  {kmax}")
        if set(a) != set(b):
            print(f"  top-level key mismatch: {set(a) ^ set(b)}")
        for k in ("method", "adaptive", "seed", "params_M", "hops", "branch", "steps", "cfg"):
            if a[k] != b[k]:
                print(f"  {k}: {a[k]!r} vs {b[k]!r}")

    print(f"\nworst over all seeds: {worst[0]:.3e} on {worst[1]} (seed {worst[2]})")


if __name__ == "__main__":
    main()
