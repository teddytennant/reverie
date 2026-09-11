#!/usr/bin/env python3
"""Aggregate the opportunist sweep's run JSONs into the tables docs/multiseed.md
reports.

    python3 scripts/aggregate_opp.py --dir ~/reverie/runs/opp

Nothing here imports jax, equinox or numpy on purpose: this has to run on the
NCShare login node against runs/opp directly, and the login node has no AVX so
the real venv can't even be imported there. Stdlib only.

Every run.py-shaped JSON carries method/adaptive/seed/branch/steps/cfg/test,
but not d_model, n_layers or the learning rate; scripts/run.py never wrote
those to the blob. The opportunist's task generator (gpu-opportunist/
make-tasks.sh) encodes them in the filename instead, so parsing the filename
is not a workaround, it is the only place that information still lives. Each
regex below is lifted straight from that script's echo lines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys

# name -> (family, regex, group->field mapping is implicit via named groups)
PATTERNS = [
    ("method_comparison", re.compile(
        r"^(?P<method>nocot|cot|coconut_distill|coconut|reverie)_s(?P<seed>\d+)\.json$")),
    ("ablation_noTraj", re.compile(r"^ablate_noTraj_s(?P<seed>\d+)\.json$")),
    ("ablation_noDepthSup", re.compile(r"^ablate_noDepthSup_s(?P<seed>\d+)\.json$")),
    ("search", re.compile(
        r"^search_(?P<method>nocot|coconut|reverie)_s(?P<seed>\d+)\.json$")),
    ("search_ablation_noTraj", re.compile(r"^search_ablate_noTraj_s(?P<seed>\d+)\.json$")),
    ("capacity_fixedlr", re.compile(
        r"^cap_l(?P<layers>\d+)_d(?P<d>\d+)_(?P<method>nocot|reverie)_s(?P<seed>\d+)\.json$")),
    ("lr_wide", re.compile(
        r"^lr_d(?P<d>\d+)_(?P<tag>\d+e\d+)_(?P<method>nocot|reverie)_s(?P<seed>\d+)\.json$")),
    ("depth", re.compile(
        r"^depth_h(?P<tag>[\d-]+)_(?P<method>nocot|reverie)_s(?P<seed>\d+)\.json$")),
    ("lrw", re.compile(
        r"^lrw_l(?P<layers>\d+)_d(?P<d>\d+)_(?P<tag>\d+e\d+)_"
        r"(?P<method>nocot|reverie|coconut)_s(?P<seed>\d+)\.json$")),
    ("connect", re.compile(r"^connect_c(?P<c>\d+)_s(?P<seed>\d+)\.json$")),
]


def tag_to_lr(tag: str) -> float:
    """'5e4' -> 5e-4, '1e2' -> 1e-2. The opportunist strips the '-' with tr -d."""
    m = re.match(r"^(\d+)e(\d+)$", tag)
    mant, exp = m.groups()
    return float(f"{mant}e-{exp}")


def classify(fname: str):
    for family, pat in PATTERNS:
        m = pat.match(fname)
        if m:
            return family, m.groupdict()
    return None, None


def load_runs(dirpath: str, limit: int | None = None):
    """Yield (family, meta, blob) for every run.py-shaped JSON in dirpath.
    Files that fail to parse (truncated by a killed job) are skipped and
    counted, not raised."""
    n_seen = n_bad = n_unclassified = 0
    names = os.listdir(dirpath)
    if limit:
        names = names[:limit]
    for fname in names:
        if not fname.endswith(".json"):
            continue
        n_seen += 1
        family, meta = classify(fname)
        if family is None:
            n_unclassified += 1
            continue
        path = os.path.join(dirpath, fname)
        try:
            with open(path) as f:
                blob = json.load(f)
            if "test" not in blob or "acc" not in blob["test"]:
                raise ValueError("missing test.acc")
        except Exception:
            n_bad += 1
            continue
        yield family, meta, blob
    print(f"# scanned {n_seen} files: {n_unclassified} unclassified, "
          f"{n_bad} unreadable/incomplete", file=sys.stderr)


def mean_std(xs):
    m = statistics.fmean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, s


def welch_sigma(m1, s1, n1, m2, s2, n2):
    """abs(delta) / pooled standard error. Matches the "sigma" figures already
    in docs/multiseed.md: a two-sample z/t statistic, not a p-value."""
    se = math.sqrt((s1 * s1) / max(n1, 1) + (s2 * s2) / max(n2, 1))
    if se == 0:
        return float("inf") if m1 != m2 else 0.0
    return (m1 - m2) / se


def group_stats(rows, key_fn, acc_fn=lambda b: b["test"]["acc"]):
    """rows: list of (meta, blob). Returns {key: (mean, std, n, accs)}."""
    buckets: dict = {}
    for meta, blob in rows:
        buckets.setdefault(key_fn(meta, blob), []).append(acc_fn(blob))
    out = {}
    for k, accs in buckets.items():
        m, s = mean_std(accs)
        out[k] = (m, s, len(accs), accs)
    return out


def fmt_contrast(name, a, b):
    ma, sa, na, _ = a
    mb, sb, nb, _ = b
    sig = welch_sigma(ma, sa, na, mb, sb, nb)
    print(f"| {name} | {ma - mb:+.4f} | {abs(sig):.2f} | n={na}/{nb} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap files scanned, for a quick check")
    args = ap.parse_args()

    by_family: dict = {}
    for family, meta, blob in load_runs(args.dir, args.limit):
        by_family.setdefault(family, []).append((meta, blob))
    for fam, rows in sorted(by_family.items()):
        print(f"# {fam}: {len(rows)} runs", file=sys.stderr)

    # ---- method comparison -----------------------------------------------
    rows = by_family.get("method_comparison", [])
    if rows:
        print("\n## Method comparison\n")
        stats = group_stats(rows, lambda meta, b: meta["method"])
        print("| method | acc | n | latent steps | rho |")
        print("|---|---|---|---|---|")
        for method in ["nocot", "cot", "coconut", "coconut_distill", "reverie"]:
            if method not in stats:
                continue
            m, s, n, _ = stats[method]
            steps = statistics.fmean(b["test"]["mean_steps"] for _, b in rows
                                     if _["method"] == method)
            rho = statistics.fmean(b["test"]["rho_steps_hops"] for _, b in rows
                                   if _["method"] == method)
            print(f"| {method} | {m:.4f} ± {s:.4f} | {n} | {steps:.3f} | {rho:+.3f} |")
        print("\n| contrast | delta | sigma | n |")
        print("|---|---|---|---|")
        for a, b in [("cot", "reverie"), ("cot", "nocot"), ("reverie", "nocot"),
                    ("reverie", "coconut")]:
            if a in stats and b in stats:
                fmt_contrast(f"{a} vs {b}", stats[a], stats[b])

    # ---- ablations ----------------------------------------------------------
    full_rows = [(m, b) for m, b in rows if m["method"] == "reverie"] if rows else []
    noTraj = by_family.get("ablation_noTraj", [])
    noDepth = by_family.get("ablation_noDepthSup", [])
    if full_rows or noTraj or noDepth:
        print("\n## Objective ablations\n")
        arms = [("reverie (full)", full_rows), ("alpha=0 (no trajectory)", noTraj),
               ("gamma=0 (no depth supervision)", noDepth)]
        print("| arm | acc | n | latent steps | rho |")
        print("|---|---|---|---|---|")
        arm_stats = {}
        for label, arm_rows in arms:
            if not arm_rows:
                continue
            accs = [b["test"]["acc"] for _, b in arm_rows]
            m, s = mean_std(accs)
            steps = statistics.fmean(b["test"]["mean_steps"] for _, b in arm_rows)
            rho = statistics.fmean(b["test"]["rho_steps_hops"] for _, b in arm_rows)
            arm_stats[label] = (m, s, len(accs), accs)
            print(f"| {label} | {m:.4f} ± {s:.4f} | {len(accs)} | {steps:.3f} | {rho:+.3f} |")
        print("\n| contrast | delta | sigma |")
        print("|---|---|---|")
        base = arm_stats.get("reverie (full)")
        if base:
            for label in ["alpha=0 (no trajectory)", "gamma=0 (no depth supervision)"]:
                if label in arm_stats:
                    a = arm_stats[label]
                    sig = welch_sigma(a[0], a[1], a[2], base[0], base[1], base[2])
                    print(f"| {label} vs full | {a[0]-base[0]:+.4f} | {abs(sig):.2f} |")

    # ---- capacity: lrw grid, best lr per (layers, d_model, method) ----------
    rows = by_family.get("lrw", [])
    if rows:
        print("\n## Capacity (lrw grid): best lr per arm\n")
        cell_key = lambda meta, b: (int(meta["layers"]), int(meta["d"]), meta["method"],
                                    tag_to_lr(meta["tag"]))
        lr_stats = group_stats(rows, cell_key)
        # best (method,layers,d) by max mean acc across its 4 lrs
        best: dict = {}
        for (L, D, method, lr), (m, s, n, accs) in lr_stats.items():
            k = (L, D, method)
            if k not in best or m > best[k][0]:
                best[k] = (m, s, n, accs, lr)
        for L in sorted({k[0] for k in best}):
            print(f"\n**{L} layer{'s' if L != 1 else ''}**\n")
            print("| d_model | nocot best | reverie best | coconut best | "
                  "delta (rev-noc) | sigma |")
            print("|---|---|---|---|---|---|")
            for D in sorted({k[1] for k in best if k[0] == L}):
                cells = {method: best.get((L, D, method)) for method in
                         ("nocot", "reverie", "coconut")}
                noc, rev, coc = cells["nocot"], cells["reverie"], cells["coconut"]

                def cellstr(c):
                    if c is None:
                        return "-"
                    m, s, n, accs, lr = c
                    return f"{m:.4f} ± {s:.3f} @ {lr:g} (n={n})"

                if noc and rev:
                    sig = welch_sigma(rev[0], rev[1], rev[2], noc[0], noc[1], noc[2])
                    delta = f"{rev[0]-noc[0]:+.4f}"
                    sigs = f"{sig:+.1f}"
                else:
                    delta = sigs = "-"
                print(f"| {D} | {cellstr(noc)} | {cellstr(rev)} | {cellstr(coc)} | "
                      f"{delta} | {sigs} |")


if __name__ == "__main__":
    main()
