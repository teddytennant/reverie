#!/usr/bin/env python
"""Train + evaluate one method on self-generated ProsQA. Self-contained.

    python scripts/run.py --method reverie --steps 3000 --hops 4

Generates train/val/test with distinct seeds (hold-out by fresh seed, no row
leakage), builds the vocab from the data, trains, evaluates on test, and writes
a JSON metrics blob.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import jax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reverie.data import build_vocab, global_lengths  # noqa: E402
from reverie.latent import ReverieConfig, ReverieModel  # noqa: E402
from reverie.model import ModelConfig  # noqa: E402
from reverie.train import evaluate, train  # noqa: E402

BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data-gen", "target", "release", "reverie-datagen")


def gen(n, seed, hops, branch, trap_depth, connect=0):
    out = subprocess.run(
        [BIN, "--n", str(n), "--seed", str(seed), "--hops", str(hops),
         "--branch", str(branch), "--trap-depth", str(trap_depth),
         "--connect", str(connect)],
        capture_output=True, text=True, check=True,
    ).stdout
    return [json.loads(line) for line in out.strip().split("\n")]


def gen_split(n, seed, hops_list, branch, trap_depth, connect=0):
    """Generate n instances; if hops_list has >1 value, mix them evenly
    (heterogeneous reasoning depth -> the calibration / depth-variance story)."""
    per = max(1, n // len(hops_list))
    insts = []
    for j, h in enumerate(hops_list):
        insts += gen(per, seed + 100 * j, h, branch, trap_depth, connect)
    import random
    random.Random(seed).shuffle(insts)
    return insts[:n] if len(insts) >= n else insts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="reverie",
                    choices=["reverie", "coconut", "coconut_distill", "nocot", "cot"])
    ap.add_argument("--adaptive", type=int, default=-1,
                    help="-1=default per method; 0/1 override")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.01)
    ap.add_argument("--lambda-prior", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--hops-mix", default="",
                    help="comma list e.g. '2,3,4,5' for heterogeneous depth; overrides --hops")
    ap.add_argument("--branch", type=int, default=2)
    ap.add_argument("--trap-depth", type=int, default=2)
    ap.add_argument("--connect", type=int, default=0,
                    help="decoy->source cross-edges: weakly-connect the graph to kill the "
                         "component-membership shortcut (forces directed-reachability reasoning)")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    print(f"jax backend={jax.default_backend()} devices={jax.devices()}")
    hops_list = [int(h) for h in args.hops_mix.split(",")] if args.hops_mix else [args.hops]
    train_insts = gen_split(args.n_train, 1000 + args.seed, hops_list, args.branch, args.trap_depth, args.connect)
    val_insts = gen_split(args.n_val, 7000 + args.seed, hops_list, args.branch, args.trap_depth, args.connect)
    test_insts = gen_split(args.n_test, 9000 + args.seed, hops_list, args.branch, args.trap_depth, args.connect)

    max_ent = max(i["n_entities"] for i in train_insts + val_insts + test_insts)
    vocab = build_vocab(max_concepts=max_ent + 1)
    plen, clen = global_lengths(train_insts + val_insts + test_insts, vocab)
    print(f"data: {len(train_insts)} train / {len(val_insts)} val / {len(test_insts)} test | "
          f"hops={hops_list} | vocab {vocab.size} | max_entities {max_ent} | "
          f"prompt_len {plen} cot_len {clen}")

    adaptive = {"reverie": True, "coconut": False, "coconut_distill": False,
                "nocot": False, "cot": False}[args.method]
    if args.adaptive in (0, 1):
        adaptive = bool(args.adaptive)
    # coconut = fixed depth, answer only (no traj/halt); coconut_distill adds traj
    alpha = args.alpha if args.method in ("reverie", "coconut_distill") else 0.0
    cfg = ReverieConfig(max_steps=args.max_steps, method=args.method, adaptive=adaptive,
                        alpha_traj=alpha, gamma_halt=args.gamma, beta_reg=args.beta,
                        lambda_prior=args.lambda_prior)

    mcfg = ModelConfig(vocab_size=vocab.size, d_model=args.d_model, n_layers=args.layers,
                       n_heads=args.heads, max_seq_len=256)
    model = ReverieModel(mcfg, key=jax.random.PRNGKey(args.seed))
    nparams = sum(x.size for x in jax.tree_util.tree_leaves(
        __import__("equinox").filter(model, __import__("equinox").is_array)))
    print(f"model: {nparams/1e6:.2f}M params | cfg={cfg}")

    model, hist, (plen, clen) = train(
        model, train_insts, val_insts, vocab, cfg, steps=args.steps,
        batch_size=args.batch_size, peak_lr=args.lr, warmup=args.warmup,
        seed=args.seed, eval_every=args.eval_every, prompt_len=plen, cot_len=clen)
    test_m = evaluate(model, test_insts, vocab, cfg, prompt_len=plen, cot_len=clen)

    # single-model accuracy-vs-latent-compute Pareto frontier: dial the halting
    # threshold ε on the ONE trained adaptive model (no retraining).
    pareto = []
    if adaptive and args.method in ("reverie", "coconut_distill"):
        for hb in [4.0, 2.0, 1.0, 0.0, -1.0, -2.0, -4.0]:  # >0 halts earlier
            mm = evaluate(model, test_insts, vocab, cfg, halt_bias=hb,
                          prompt_len=plen, cot_len=clen)
            pareto.append(dict(halt_bias=hb, acc=mm["acc"], mean_steps=mm["mean_steps"]))

    dt = time.time() - t0
    result = dict(method=args.method, adaptive=adaptive, test=test_m, pareto=pareto,
                  seed=args.seed, params_M=round(nparams / 1e6, 3),
                  hops=hops_list if len(hops_list) > 1 else args.hops, branch=args.branch,
                  steps=args.steps, wall_s=round(dt, 1), cfg=dataclasses_dict(cfg))
    print("TEST:", json.dumps(test_m, indent=2))
    if pareto:
        print("PARETO:", json.dumps(pareto))
    print(f"wall {dt:.1f}s")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print("wrote", args.out)


def dataclasses_dict(cfg):
    import dataclasses
    return dataclasses.asdict(cfg)


if __name__ == "__main__":
    main()
