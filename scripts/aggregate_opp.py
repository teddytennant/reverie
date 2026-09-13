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
    ("ablation_noTrajNoDepth", re.compile(r"^ablate_noTrajNoDepth_s(?P<seed>\d+)\.json$")),
    ("fix", re.compile(
        r"^fix_l(?P<layers>\d+)_d(?P<d>\d+)_(?P<tag>\d+e\d+)_s(?P<seed>\d+)\.json$")),
    ("fixA", re.compile(
        r"^fixA_l1_d(?P<d>\d+)_(?P<tag>\d+e\d+)_s(?P<seed>\d+)\.json$")),
    ("fixG", re.compile(
        r"^fixG_l1_d(?P<d>\d+)_(?P<tag>\d+e\d+)_s(?P<seed>\d+)\.json$")),
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
    ("ksweep_easy", re.compile(r"^ksweep_easy_K(?P<k>\d+)_s(?P<seed>\d+)\.json$")),
    ("ksweep_hard", re.compile(
        r"^ksweep_hard_l(?P<layers>\d+)_d(?P<d>\d+)_K(?P<k>\d+)_s(?P<seed>\d+)\.json$")),
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
    nocot_rows = [(m, b) for m, b in rows if m["method"] == "nocot"] if rows else []
    coconut_rows = [(m, b) for m, b in rows if m["method"] == "coconut"] if rows else []
    noTraj = by_family.get("ablation_noTraj", [])
    noDepth = by_family.get("ablation_noDepthSup", [])
    noTrajNoDepth = by_family.get("ablation_noTrajNoDepth", [])
    if full_rows or noTraj or noDepth or noTrajNoDepth:
        print("\n## Objective ablations\n")
        arms = [("reverie (full)", full_rows), ("alpha=0 (no trajectory)", noTraj),
               ("gamma=0 (no depth supervision)", noDepth),
               ("alpha=0, gamma=0 (the fix)", noTrajNoDepth),
               ("nocot", nocot_rows), ("coconut", coconut_rows)]
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
            for label in ["alpha=0 (no trajectory)", "gamma=0 (no depth supervision)",
                         "alpha=0, gamma=0 (the fix)"]:
                if label in arm_stats:
                    a = arm_stats[label]
                    sig = welch_sigma(a[0], a[1], a[2], base[0], base[1], base[2])
                    print(f"| {label} vs full | {a[0]-base[0]:+.4f} | {abs(sig):.2f} |")
        fix = arm_stats.get("alpha=0, gamma=0 (the fix)")
        if fix:
            for label in ["nocot", "coconut"]:
                if label in arm_stats:
                    a = arm_stats[label]
                    sig = welch_sigma(fix[0], fix[1], fix[2], a[0], a[1], a[2])
                    print(f"| the fix vs {label} | {fix[0]-a[0]:+.4f} | {abs(sig):.2f} |")

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

    # ---- capacity: fix grid (alpha=0, gamma=0), best lr per (layers, d) -----
    fix_rows = by_family.get("fix", [])
    if fix_rows:
        print("\n## Capacity (fix grid): alpha=0, gamma=0, best lr per cell\n")
        fix_key = lambda meta, b: (int(meta["layers"]), int(meta["d"]),
                                   tag_to_lr(meta["tag"]))
        fix_lr_stats = group_stats(fix_rows, fix_key)
        fix_best: dict = {}
        for (L, D, lr), (m, s, n, accs) in fix_lr_stats.items():
            k = (L, D)
            if k not in fix_best or m > fix_best[k][0]:
                fix_best[k] = (m, s, n, accs, lr)
        # steps/rho at the winning (L, D, lr) cell, not pooled across lrs
        fix_meta_rows = {}
        for meta, blob in fix_rows:
            k = (int(meta["layers"]), int(meta["d"]), tag_to_lr(meta["tag"]))
            fix_meta_rows.setdefault(k, []).append(blob)
        # reuse lrw's own best-lr dict so the comparison is cell-for-cell,
        # not a re-derivation that could silently drift from the table above
        lrw_rows = by_family.get("lrw", [])
        lrw_best: dict = {}
        if lrw_rows:
            lrw_key = lambda meta, b: (int(meta["layers"]), int(meta["d"]), meta["method"],
                                       tag_to_lr(meta["tag"]))
            lrw_lr_stats = group_stats(lrw_rows, lrw_key)
            for (L, D, method, lr), (m, s, n, accs) in lrw_lr_stats.items():
                k = (L, D, method)
                if k not in lrw_best or m > lrw_best[k][0]:
                    lrw_best[k] = (m, s, n, accs, lr)
        for L in sorted({k[0] for k in fix_best}):
            print(f"\n**{L} layer{'s' if L != 1 else ''}**\n")
            print("| d_model | nocot best | reverie best | coconut best | "
                  "fix best | steps | rho | delta (fix-noc) | sigma | "
                  "delta (fix-rev) | sigma |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for D in sorted({k[1] for k in fix_best if k[0] == L}):
                noc = lrw_best.get((L, D, "nocot"))
                rev = lrw_best.get((L, D, "reverie"))
                coc = lrw_best.get((L, D, "coconut"))
                fix = fix_best[(L, D)]
                fm, fs, fn, faccs, flr = fix
                blobs = fix_meta_rows[(L, D, flr)]
                steps = statistics.fmean(b["test"]["mean_steps"] for b in blobs)
                rho = statistics.fmean(b["test"]["rho_steps_hops"] for b in blobs)

                def cellstr(c):
                    if c is None:
                        return "-"
                    m, s, n, accs, lr = c
                    return f"{m:.4f} ± {s:.3f} @ {lr:g} (n={n})"

                def contrast(a, b):
                    if a is None or b is None:
                        return "-", "-"
                    return f"{a[0]-b[0]:+.4f}", f"{welch_sigma(a[0], a[1], a[2], b[0], b[1], b[2]):+.1f}"

                d_noc, s_noc = contrast(fix, noc)
                d_rev, s_rev = contrast(fix, rev)
                print(f"| {D} | {cellstr(noc)} | {cellstr(rev)} | {cellstr(coc)} | "
                      f"{fm:.4f} ± {fs:.3f} @ {flr:g} (n={fn}) | {steps:.3f} | {rho:+.3f} | "
                      f"{d_noc} | {s_noc} | {d_rev} | {s_rev} |")


    # ---- capacity: fixA/fixG (one term dropped at a time), one layer only ---
    fixA_rows = by_family.get("fixA", [])
    fixG_rows = by_family.get("fixG", [])
    if fixA_rows or fixG_rows:
        print("\n## Capacity (fixA/fixG): one term dropped, one layer only\n")

        def best_per_d(rows):
            key = lambda meta, b: (int(meta["d"]), tag_to_lr(meta["tag"]))
            lr_stats = group_stats(rows, key)
            best: dict = {}
            for (D, lr), (m, s, n, accs) in lr_stats.items():
                if D not in best or m > best[D][0]:
                    best[D] = (m, s, n, accs, lr)
            meta_rows = {}
            for meta, blob in rows:
                k = (int(meta["d"]), tag_to_lr(meta["tag"]))
                meta_rows.setdefault(k, []).append(blob)
            return best, meta_rows

        a_best, a_meta = best_per_d(fixA_rows)
        g_best, g_meta = best_per_d(fixG_rows)
        lrw_rows = by_family.get("lrw", [])
        lrw_best = {}
        if lrw_rows:
            lrw_key = lambda meta, b: (int(meta["layers"]), int(meta["d"]), meta["method"],
                                       tag_to_lr(meta["tag"]))
            lrw_lr_stats = group_stats(lrw_rows, lrw_key)
            for (L, D, method, lr), (m, s, n, accs) in lrw_lr_stats.items():
                k = (L, D, method)
                if k not in lrw_best or m > lrw_best[k][0]:
                    lrw_best[k] = (m, s, n, accs, lr)

        def cellstr(c):
            if c is None:
                return "-"
            m, s, n, accs, lr = c
            return f"{m:.4f} ± {s:.3f} @ {lr:g} (n={n})"

        def contrast(a, b):
            if a is None or b is None:
                return "-", "-"
            return f"{a[0]-b[0]:+.4f}", f"{welch_sigma(a[0], a[1], a[2], b[0], b[1], b[2]):+.1f}"

        widths = sorted(set(a_best) | set(g_best))
        print("| d_model | nocot best (L1) | reverie best (L1) | alpha=0 only | "
              "steps/rho | vs reverie | gamma=0 only | steps/rho | vs reverie |")
        print("|---|---|---|---|---|---|---|---|---|")
        for D in widths:
            noc = lrw_best.get((1, D, "nocot"))
            rev = lrw_best.get((1, D, "reverie"))
            a = a_best.get(D)
            g = g_best.get(D)
            a_steps = a_rho = g_steps = g_rho = float("nan")
            if a:
                blobs = a_meta[(D, a[4])]
                a_steps = statistics.fmean(b["test"]["mean_steps"] for b in blobs)
                a_rho = statistics.fmean(b["test"]["rho_steps_hops"] for b in blobs)
            if g:
                blobs = g_meta[(D, g[4])]
                g_steps = statistics.fmean(b["test"]["mean_steps"] for b in blobs)
                g_rho = statistics.fmean(b["test"]["rho_steps_hops"] for b in blobs)
            da, sa = contrast(a, rev)
            dg, sg = contrast(g, rev)
            print(f"| {D} | {cellstr(noc)} | {cellstr(rev)} | {cellstr(a)} | "
                  f"{a_steps:.2f}/{a_rho:+.2f} | {da} ({sa}) | {cellstr(g)} | "
                  f"{g_steps:.2f}/{g_rho:+.2f} | {dg} ({sg}) |")

    # ---- K-sweep: fixed-K coconut recipe, only K varies -----------------
    easy_rows = by_family.get("ksweep_easy", [])
    if easy_rows:
        print("\n## K-sweep (easy task, branch=0/trap=0, d=128, 2 layers)\n")
        key = lambda meta, b: int(meta["k"])
        stats = group_stats(easy_rows, key)
        print("| K | acc | n |")
        print("|---|---|---|")
        for k in sorted(stats):
            m, s, n, _ = stats[k]
            print(f"| {k} | {m:.4f} ± {s:.4f} | {n} |")
        ks = sorted(stats)
        print("\n| contrast | delta | sigma |")
        print("|---|---|---|")
        for a, b in zip(ks[1:], ks[:-1]):
            ma, sa, na, _ = stats[a]
            mb, sb, nb, _ = stats[b]
            sig = welch_sigma(ma, sa, na, mb, sb, nb)
            print(f"| K={a} vs K={b} | {ma-mb:+.4f} | {abs(sig):.2f} |")

    hard_rows = by_family.get("ksweep_hard", [])
    if hard_rows:
        print("\n## K-sweep (hard task, branch=1/trap=1, 2 layers)\n")
        key = lambda meta, b: (int(meta["d"]), int(meta["k"]))
        stats = group_stats(hard_rows, key)
        widths = sorted({d for d, _ in stats})
        for D in widths:
            print(f"\n**d_model={D}**\n")
            print("| K | acc | n |")
            print("|---|---|---|")
            ks = sorted(k for d, k in stats if d == D)
            for k in ks:
                m, s, n, _ = stats[(D, k)]
                print(f"| {k} | {m:.4f} ± {s:.4f} | {n} |")
            print("\n| contrast | delta | sigma |")
            print("|---|---|---|")
            for a, b in zip(ks[1:], ks[:-1]):
                ma, sa, na, _ = stats[(D, a)]
                mb, sb, nb, _ = stats[(D, b)]
                sig = welch_sigma(ma, sa, na, mb, sb, nb)
                print(f"| K={a} vs K={b} | {ma-mb:+.4f} | {abs(sig):.2f} |")


if __name__ == "__main__":
    main()
