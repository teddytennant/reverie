#!/usr/bin/env python
"""Train many seeds at once on one device and write one run.py-shaped JSON each.

    python scripts/ensemble.py --method reverie --seeds 0,1,2,3 \
        --steps 3000 --hops-mix 2,3,4,5 --outdir runs/ens

Same flags as scripts/run.py, plus --seeds and --outdir. The output files are
`{method}_s{seed}.json` with exactly run.py's keys, so scripts/matrix.py's
aggregation and anything else reading those blobs keeps working.

Replicas have to share array shapes to be stacked, and the vocab size, prompt
width and CoT width all come from the generated data, so they can in principle
differ by seed. Seeds are grouped by that signature and each group is trained as
its own ensemble. For a fixed generator config the signature is normally
identical across seeds and there is one group.

`wall_s` in each JSON is this process's whole wall clock divided by the number of
seeds, which is the per-run cost and is measured over the same span run.py's
`wall_s` covers. The total is printed at the end.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import defaultdict

import equinox as eqx
import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reverie.data import build_vocab, global_lengths  # noqa: E402
from reverie.ensemble import (  # noqa: E402
    batch_index_stream,
    ensemble_evaluate,
    init_ensemble,
    make_chunk_runner,
    stack_splits,
    tokenize_split,
    train_fields,
)
from reverie.latent import ReverieConfig  # noqa: E402
from reverie.model import ModelConfig  # noqa: E402
from run import gen_split  # noqa: E402

EVAL_FIELDS = ("prompt_ids", "prompt_mask", "answer", "n_hops")


def add_run_args(ap):
    """run.py's flags, minus --seed/--out."""
    ap.add_argument("--method", default="reverie",
                    choices=["reverie", "coconut", "coconut_distill", "nocot", "cot"])
    ap.add_argument("--adaptive", type=int, default=-1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--lambda-prior", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--hops-mix", default="")
    ap.add_argument("--branch", type=int, default=2)
    ap.add_argument("--trap-depth", type=int, default=2)
    ap.add_argument("--connect", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)


def make_cfg(args):
    adaptive = {"reverie": True, "coconut": False, "coconut_distill": False,
                "nocot": False, "cot": False}[args.method]
    if args.adaptive in (0, 1):
        adaptive = bool(args.adaptive)
    alpha = args.alpha if args.method in ("reverie", "coconut_distill") else 0.0
    gamma = args.gamma if adaptive else 0.0
    beta = args.beta if adaptive else 0.0
    cfg = ReverieConfig(max_steps=args.max_steps, method=args.method, adaptive=adaptive,
                        alpha_traj=alpha, gamma_halt=gamma, beta_reg=beta,
                        lambda_prior=args.lambda_prior)
    return cfg, adaptive


def build_seed_data(args, seed, hops_list):
    """run.py's data section for one seed, verbatim."""
    train_insts = gen_split(args.n_train, 1000 + seed, hops_list, args.branch,
                            args.trap_depth, args.connect)
    val_insts = gen_split(args.n_val, 7000 + seed, hops_list, args.branch,
                          args.trap_depth, args.connect)
    test_insts = gen_split(args.n_test, 9000 + seed, hops_list, args.branch,
                           args.trap_depth, args.connect)
    all_insts = train_insts + val_insts + test_insts
    max_ent = max(i["n_entities"] for i in all_insts)
    vocab = build_vocab(max_concepts=max_ent + 1)
    plen, clen = global_lengths(all_insts, vocab)
    # evaluate() picks one generation width per call from the split's deepest item
    val_gen = int(max(h["n_hops"] for h in val_insts)) * 6 + 4
    test_gen = int(max(h["n_hops"] for h in test_insts)) * 6 + 4
    sig = (vocab.size, plen, clen, val_gen, test_gen)
    return dict(seed=seed, train=train_insts, val=val_insts, test=test_insts,
                vocab=vocab, plen=plen, clen=clen, val_gen=val_gen,
                test_gen=test_gen, sig=sig)


def eval_bundle(insts, vocab, cfg, plen, clen):
    """Same arrays train.evaluate builds: the answer and hop count come out of
    collate, not out of the raw instance, so any renaming stays in one place."""
    ds = tokenize_split(insts, vocab, cfg.max_steps, plen, clen, EVAL_FIELDS)
    cands = np.asarray([[vocab.concept_id(c) for c in it["candidates"]] for it in insts],
                       dtype=np.int32)
    answers = np.asarray(ds["answer"])
    hops = np.asarray(ds["n_hops"])
    return {k: ds[k] for k in ("prompt_ids", "prompt_mask")}, cands, answers, hops


def run_group(args, cfg, adaptive, group, hops_list, log=print):
    seeds = [g["seed"] for g in group]
    R = len(seeds)
    vocab = group[0]["vocab"]
    plen, clen = group[0]["plen"], group[0]["clen"]
    fields = train_fields(args.method)

    t0 = time.time()
    train_ds = stack_splits([
        tokenize_split(g["train"], vocab, cfg.max_steps, plen, clen, fields) for g in group
    ])
    val_parts = [eval_bundle(g["val"], vocab, cfg, plen, clen) for g in group]
    test_parts = [eval_bundle(g["test"], vocab, cfg, plen, clen) for g in group]
    val_ds = stack_splits([p[0] for p in val_parts])
    test_ds = stack_splits([p[0] for p in test_parts])
    val_pack = (val_ds, np.stack([p[1] for p in val_parts]),
                np.stack([p[2] for p in val_parts]), np.stack([p[3] for p in val_parts]))
    test_pack = (test_ds, np.stack([p[1] for p in test_parts]),
                 np.stack([p[2] for p in test_parts]), np.stack([p[3] for p in test_parts]))
    log(f"tokenized {R} replicas in {time.time()-t0:.1f}s")

    mcfg = ModelConfig(vocab_size=vocab.size, d_model=args.d_model,
                       n_layers=args.layers, n_heads=args.heads, max_seq_len=256)
    arr, static, optim, opt_state = init_ensemble(
        mcfg, seeds, args.lr, args.warmup, args.steps)
    nparams = sum(int(x.size) for x in jax.tree_util.tree_leaves(
        eqx.filter(arr, eqx.is_array))) // R
    log(f"model: {nparams/1e6:.2f}M params x {R} replicas | cfg={cfg}")

    # train.train permutes over len(train_insts), not over --n-train, and the two
    # part company if the generator hands back fewer instances than asked for.
    n_train = len(group[0]["train"])
    assert all(len(g["train"]) == n_train for g in group), "replicas disagree on train size"
    idx = np.stack([batch_index_stream(s, n_train, args.batch_size, args.steps)
                    for s in seeds], axis=1)          # [steps, R, B]
    run_chunk = make_chunk_runner(optim, cfg, static)

    t_train = time.time()
    # see run.py: same marks, so both paths can be sliced the same way
    log(f"PHASE train_start {t_train:.3f}")
    done = 0
    while done < args.steps:
        next_eval = (done // args.eval_every + 1) * args.eval_every
        n = min(args.chunk, args.steps - done, next_eval - done)
        arr, opt_state, (losses, aux) = run_chunk(
            arr, opt_state, train_ds, jax.numpy.asarray(idx[done : done + n]))
        losses = np.asarray(losses)
        done += n
        if done % args.eval_every == 0 or done == args.steps:
            ms = ensemble_evaluate(arr, static, val_pack[0], val_pack[1], val_pack[2],
                                   val_pack[3], cfg, group[0]["val_gen"])
            accs = " ".join(f"{m['acc']:.3f}" for m in ms)
            log(f"[{cfg.method}] step {done:5d}  loss {losses[-1].mean():7.3f}  "
                f"val_acc {accs}")
        else:
            log(f"[{cfg.method}] step {done:5d}  loss {losses[-1].mean():7.3f}")
    jax.block_until_ready(arr)
    train_wall = time.time() - t_train
    log(f"PHASE train_end {time.time():.3f}")

    test_ms = ensemble_evaluate(arr, static, test_pack[0], test_pack[1], test_pack[2],
                                test_pack[3], cfg, group[0]["test_gen"])
    paretos = [[] for _ in seeds]
    if adaptive and args.method in ("reverie", "coconut_distill"):
        for hb in [4.0, 2.0, 1.0, 0.0, -1.0, -2.0, -4.0]:
            mm = ensemble_evaluate(arr, static, test_pack[0], test_pack[1], test_pack[2],
                                   test_pack[3], cfg, group[0]["test_gen"], halt_bias=hb)
            for r in range(R):
                paretos[r].append(dict(halt_bias=hb, acc=mm[r]["acc"],
                                       mean_steps=mm[r]["mean_steps"]))
    total = time.time() - t0
    log(f"group of {R}: train {train_wall:.1f}s, total {total:.1f}s "
        f"({total/R:.1f}s per seed)")

    results = []
    for r, seed in enumerate(seeds):
        results.append(dict(
            method=args.method, adaptive=adaptive, test=test_ms[r], pareto=paretos[r],
            seed=seed, params_M=round(nparams / 1e6, 3),
            hops=hops_list if len(hops_list) > 1 else args.hops, branch=args.branch,
            steps=args.steps, wall_s=round(total / R, 1),  # main() rewrites this
            cfg=dataclasses.asdict(cfg)))
    return results, total, train_wall


def main():
    ap = argparse.ArgumentParser()
    add_run_args(ap)
    ap.add_argument("--seeds", default="0", help="comma list, e.g. 0,1,2,3")
    ap.add_argument("--outdir", default="", help="write {method}_s{seed}.json here")
    ap.add_argument("--name-pattern", default="{method}_s{seed}.json")
    ap.add_argument("--chunk", type=int, default=100,
                    help="training steps per lax.scan; bigger cuts dispatch overhead "
                         "but delays the loss printout")
    args = ap.parse_args()

    t0 = time.time()
    print(f"jax backend={jax.default_backend()} devices={jax.devices()}")
    seeds = [int(s) for s in args.seeds.split(",")]
    hops_list = [int(h) for h in args.hops_mix.split(",")] if args.hops_mix else [args.hops]
    cfg, adaptive = make_cfg(args)

    t_gen = time.time()
    per_seed = [build_seed_data(args, s, hops_list) for s in seeds]
    print(f"generated {len(seeds)} seeds of data in {time.time()-t_gen:.1f}s")

    groups = defaultdict(list)
    for g in per_seed:
        groups[g["sig"]].append(g)
    if len(groups) > 1:
        print(f"WARNING: seeds do not share one data shape, splitting into "
              f"{len(groups)} ensembles: "
              + "; ".join(f"{[x['seed'] for x in v]} -> vocab={k[0]} plen={k[1]} clen={k[2]}"
                          for k, v in groups.items()))
    g0 = per_seed[0]
    print(f"data: {len(g0['train'])} train / {len(g0['val'])} val / {len(g0['test'])} test | "
          f"hops={hops_list} | vocab {g0['vocab'].size} | "
          f"prompt_len {g0['plen']} cot_len {g0['clen']}")

    all_results, train_wall = [], 0.0
    for group in groups.values():
        res, _, tw = run_group(args, cfg, adaptive, group, hops_list)
        all_results += res
        train_wall += tw

    all_results.sort(key=lambda r: seeds.index(r["seed"]))
    dt = time.time() - t0
    # run.py's wall_s is measured from the top of main and so counts data
    # generation. Overwrite the per-group number with the same thing here, or
    # the two JSONs report different quantities under the same key.
    for r in all_results:
        r["wall_s"] = round(dt / len(seeds), 1)
        print(f"seed {r['seed']}: acc {r['test']['acc']:.4f}  "
              f"mean_steps {r['test']['mean_steps']:.2f}  "
              f"rho {r['test']['rho_steps_hops']:+.3f}")
    print(f"wall {dt:.1f}s total ({dt/len(seeds):.1f}s per seed, "
          f"{len(seeds)/dt:.4f} runs/s), of which training {train_wall:.1f}s")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        for r in all_results:
            path = os.path.join(args.outdir,
                                args.name_pattern.format(method=args.method, seed=r["seed"]))
            with open(path, "w") as f:
                json.dump(r, f, indent=2)
        print(f"wrote {len(all_results)} files to {args.outdir}")


if __name__ == "__main__":
    main()
