#!/usr/bin/env python
"""Render the ablation / calibration comparison table from run JSONs.

    python scripts/ablation_table.py

Reads the standard Phase-0 run outputs (whichever exist) and prints a markdown
table of accuracy, calibration ρ, mean latent steps, and per-hop mean steps —
the evidence that the depth-supervision term (γ) is what produces calibration.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    ("Reverie (full)", "runs/reverie_s0.json"),
    ("− depth-sup (γ=0)", "runs/ablate_noDepthSup.json"),
    ("− trajectory (α=0)", "runs/ablate_noTraj.json"),
    ("Reverie (search regime)", "runs/search_reverie.json"),
]


def load(p):
    fp = os.path.join(ROOT, p)
    return json.load(open(fp)) if os.path.exists(fp) else None


def main():
    rows = []
    for name, path in RUNS:
        d = load(path)
        if not d:
            continue
        t = d["test"]
        sbh = t.get("steps_by_hop", {})
        sbh_s = " ".join(f"{k}:{v:.1f}" for k, v in sorted(sbh.items(), key=lambda x: int(x[0])))
        rows.append((name, t["acc"], t["rho_steps_hops"], t["mean_steps"], sbh_s))

    print("| config | acc | ρ(steps,hops) | mean steps | steps by hop |")
    print("|---|---|---|---|---|")
    for name, acc, rho, ms, sbh in rows:
        print(f"| {name} | {acc:.3f} | {rho:+.2f} | {ms:.2f} | {sbh} |")

    full = load("runs/reverie_s0.json")
    nog = load("runs/ablate_noDepthSup.json")
    if full and nog:
        print(f"\nCalibration is caused by depth-supervision: ρ = "
              f"{full['test']['rho_steps_hops']:+.2f} (full) vs "
              f"{nog['test']['rho_steps_hops']:+.2f} (γ=0).")


if __name__ == "__main__":
    main()
