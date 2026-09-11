#!/usr/bin/env python
"""Replica-steps/s and peak memory for the vmapped ensemble against R.

    python scripts/bench_scale.py --replicas 1,2,4,8,16,32 --steps 60

Trains nothing useful. For each R it runs one chunk to pay the compile, then
times an identical chunk and reports (R * steps) / wall, which is what turns
into runs/s once you fix a step budget. Both chunks are the same length so the
timed one is a cache hit. The sequential baseline runs the same steps one model
at a time through train.make_train_step, in this process, against the same data
and the same compile cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reverie.data import build_vocab, collate, global_lengths  # noqa: E402
from reverie.ensemble import (  # noqa: E402
    batch_index_stream,
    init_ensemble,
    make_chunk_runner,
    stack_splits,
    tokenize_split,
    train_fields,
)
from reverie.latent import ReverieConfig, ReverieModel  # noqa: E402
from reverie.model import ModelConfig  # noqa: E402
from reverie.train import _to_jax, make_optimizer, make_train_step  # noqa: E402
from run import gen_split  # noqa: E402


def peak_gb():
    st = jax.local_devices()[0].memory_stats() or {}
    return st.get("peak_bytes_in_use", 0) / 1e9


def reset_peak() -> bool:
    d = jax.local_devices()[0]
    try:
        d.reset_memory_stats()
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", default="1,2,4,8,16,32")
    ap.add_argument("--steps", type=int, default=60, help="timed steps, a multiple of --chunk")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--hops-mix", default="2,3,4,5")
    ap.add_argument("--branch", type=int, default=2)
    ap.add_argument("--trap-depth", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--chunk", type=int, default=0, help="scan length; 0 = --steps")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    Rs = [int(r) for r in args.replicas.split(",")]
    hops = [int(h) for h in args.hops_mix.split(",")]
    chunk = args.chunk or args.steps
    assert args.steps % chunk == 0, "--steps must be a multiple of --chunk"
    # one warm chunk in front of the timed ones, and a schedule long enough to
    # cover both. The schedule shape does not affect throughput, it just has to
    # be legal: optax rejects a cosine decay of zero steps.
    total = chunk + args.steps
    warmup = max(1, total // 10)
    print(f"jax {jax.__version__} backend={jax.default_backend()} {jax.devices()}")
    print(f"memory-stat reset available: {reset_peak()}")

    # one seed's data, reused for every replica. Replicas normally get their own
    # splits; here only the shapes and the arithmetic matter, and sharing keeps
    # the host side out of the measurement.
    insts = gen_split(args.n_train, 1000, hops, args.branch, args.trap_depth, 0)
    vocab = build_vocab(max_concepts=max(i["n_entities"] for i in insts) + 1)
    plen, clen = global_lengths(insts, vocab)
    cfg = ReverieConfig(max_steps=args.max_steps, method="reverie")
    mcfg = ModelConfig(vocab_size=vocab.size, d_model=args.d_model, n_layers=args.layers,
                       n_heads=args.heads, max_seq_len=256)
    one = tokenize_split(insts, vocab, cfg.max_steps, plen, clen, train_fields("reverie"))
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(ReverieModel(mcfg, key=jax.random.PRNGKey(0)), eqx.is_array)))
    print(f"model {nparams/1e6:.3f}M params | vocab {vocab.size} | prompt_len {plen} "
          f"| batch {args.batch_size} | K {args.max_steps} | n_train {args.n_train}")

    rows = []

    # ---- sequential: one model at a time, train.train's inner loop
    reset_peak()
    model = ReverieModel(mcfg, key=jax.random.PRNGKey(0))
    optim = make_optimizer(args.lr, warmup, total)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    step_fn = make_train_step(optim, cfg)
    idx1 = batch_index_stream(0, args.n_train, args.batch_size, total)
    batches = [_to_jax(collate([insts[j] for j in idx1[t]], vocab, cfg.max_steps,
                               prompt_len=plen, cot_len=clen))
               for t in range(total)]
    tc = time.time()
    model, opt_state, loss, _ = step_fn(model, opt_state, batches[0])
    jax.block_until_ready(loss)
    seq_compile = time.time() - tc
    for t in range(1, chunk):
        model, opt_state, loss, _ = step_fn(model, opt_state, batches[t])
    jax.block_until_ready(loss)
    t0 = time.time()
    for t in range(chunk, total):
        model, opt_state, loss, _ = step_fn(model, opt_state, batches[t])
    jax.block_until_ready(loss)
    seq_wall = time.time() - t0
    seq_rps = args.steps / seq_wall
    rows.append(dict(mode="sequential", R=1, steps=args.steps, wall_s=round(seq_wall, 3),
                     replica_steps_per_s=round(seq_rps, 2), peak_gb=round(peak_gb(), 2),
                     compile_s=round(seq_compile, 2)))
    print(f"sequential 1 model: compile {seq_compile:6.1f}s  {seq_wall:7.2f}s for "
          f"{args.steps} steps -> {seq_rps:8.1f} replica-steps/s, peak {peak_gb():.2f} GB",
          flush=True)
    del model, opt_state, batches

    # ---- ensemble at each R
    for R in Rs:
        reset_peak()
        try:
            ds = stack_splits([one] * R)
            arr, static, optim, opt_state = init_ensemble(
                mcfg, list(range(R)), args.lr, warmup, total)
            run_chunk = make_chunk_runner(optim, cfg, static)
            idx = np.stack([batch_index_stream(s, args.n_train, args.batch_size, total)
                            for s in range(R)], axis=1)
            tc = time.time()
            arr, opt_state, out = run_chunk(arr, opt_state, ds, jnp.asarray(idx[:chunk]))
            jax.block_until_ready(out)
            compile_s = time.time() - tc

            t0 = time.time()
            for done in range(chunk, total, chunk):
                arr, opt_state, out = run_chunk(arr, opt_state, ds,
                                                jnp.asarray(idx[done : done + chunk]))
            jax.block_until_ready(out)
            wall = time.time() - t0
            rps = R * args.steps / wall
            rows.append(dict(mode="ensemble", R=R, steps=args.steps, wall_s=round(wall, 3),
                             replica_steps_per_s=round(rps, 2), peak_gb=round(peak_gb(), 2),
                             compile_s=round(compile_s, 2),
                             speedup_vs_sequential=round(rps / seq_rps, 2)))
            print(f"ensemble R={R:3d}: compile {compile_s:6.1f}s  {wall:7.2f}s for "
                  f"{args.steps} steps -> {rps:8.1f} replica-steps/s "
                  f"({rps/seq_rps:5.2f}x), peak {peak_gb():.2f} GB", flush=True)
            del arr, opt_state, ds, run_chunk, out
        except Exception as e:  # running out of device memory is how this ends
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            rows.append(dict(mode="ensemble", R=R, error=msg))
            print(f"ensemble R={R:3d}: FAILED {msg}", flush=True)
            break

    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), params=nparams, prompt_len=plen, rows=rows), f,
                      indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
