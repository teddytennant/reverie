#!/usr/bin/env python
"""Render result tables (markdown) from run JSONs for the paper.

    python scripts/report.py runs/matrix.json runs/reverie_s0.json

- From a matrix.json: the main method-comparison table.
- From any single-run JSON: per-hop accuracy + mean latent steps (calibration),
  and the ε-swept accuracy-vs-latent-steps Pareto frontier if present.
"""

from __future__ import annotations

import json
import sys


def main_table(matrix: dict) -> str:
    res = matrix["results"]
    methods = list(res.keys())
    hopset = sorted({int(k) for m in methods for r in res[m]
                     for k in r["test"]["acc_by_hop"]})
    hdr = ["method", "acc", "steps", "ρ(steps,hops)"] + [f"acc@k={k}" for k in hopset]
    out = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]
    import statistics
    for m in methods:
        accs = [r["test"]["acc"] for r in res[m]]
        am = statistics.mean(accs)
        asd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        sm = statistics.mean(r["test"]["mean_steps"] for r in res[m])
        rm = statistics.mean(r["test"]["rho_steps_hops"] for r in res[m])
        abh = res[m][0]["test"]["acc_by_hop"]
        cells = [m, f"{am:.3f}±{asd:.3f}", f"{sm:.2f}", f"{rm:+.2f}"]
        cells += [f"{abh.get(str(k), abh.get(k, float('nan'))):.2f}" for k in hopset]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def calibration_table(run: dict) -> str:
    t = run["test"]
    abh, sbh = t.get("acc_by_hop", {}), t.get("steps_by_hop", {})
    ks = sorted({int(k) for k in abh}, key=int)
    out = [f"**{run['method']}** — calibration (ρ={t['rho_steps_hops']:+.2f})",
           "| hop k | accuracy | mean latent steps |", "|---|---|---|"]
    for k in ks:
        out.append(f"| {k} | {abh.get(str(k), abh.get(k)):.2f} | "
                   f"{sbh.get(str(k), sbh.get(k, float('nan'))):.2f} |")
    return "\n".join(out)


def pareto_table(run: dict) -> str:
    p = run.get("pareto") or []
    if not p:
        return ""
    key = "halt_bias" if "halt_bias" in p[0] else "eps"
    out = [f"**{run['method']}** — single-model Pareto (dial halt)",
           f"| {key} | accuracy | mean latent steps |", "|---|---|---|"]
    for row in p:
        out.append(f"| {row[key]:+.1f} | {row['acc']:.3f} | {row['mean_steps']:.2f} |")
    return "\n".join(out)


def main():
    for path in sys.argv[1:]:
        with open(path) as f:
            blob = json.load(f)
        print(f"\n### {path}\n")
        if "results" in blob:            # matrix
            print(main_table(blob))
        else:                            # single run
            print(calibration_table(blob))
            pt = pareto_table(blob)
            if pt:
                print("\n" + pt)


if __name__ == "__main__":
    main()
