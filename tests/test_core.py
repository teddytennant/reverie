"""Data labels, model shapes, halting math, gradients."""

import json
import os
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from reverie.data import build_vocab, collate, concept_name, render
from reverie.latent import (
    ReverieConfig,
    ReverieModel,
    batch_loss,
    halting_distribution,
    latent_unroll,
)
from reverie.model import ModelConfig, Transformer

BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data-gen", "target", "release", "reverie-datagen")


def _gen(n=16, seed=3, hops=4, branch=2):
    if not os.path.exists(BIN):
        pytest.skip("rust generator not built")
    out = subprocess.run([BIN, "--n", str(n), "--seed", str(seed), "--hops", str(hops),
                          "--branch", str(branch)], capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.strip().split("\n")]


# ---- data ----
def test_concept_name_matches_rust():
    # these are the strings data-gen/src/main.rs::concept_name produces
    assert [concept_name(i) for i in range(4)] == ["ba", "ca", "da", "fa"]
    assert concept_name(20) == "be" and concept_name(21) == "ce"
    names = [concept_name(i) for i in range(500)]
    assert len(set(names)) == 500


def test_labels_are_correct():
    """The answer has to be reachable from source and the decoy must not be."""
    insts = _gen(n=64, hops=5, branch=3)

    def reachable(edges, src, dst):
        adj = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)
        seen, stack = {src}, [src]
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            for w in adj.get(u, []):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        return dst == src

    for it in insts:
        src, ans = it["source"], it["answer"]
        c0, c1 = it["candidates"]
        decoy = c1 if ans == c0 else c0
        assert reachable(it["edges"], src, ans), "answer must be reachable"
        assert not reachable(it["edges"], src, decoy), "decoy must be unreachable"
        # gold path is a real chain source -> ... -> answer
        gp = it["gold_path"]
        assert gp[0] == src and gp[-1] == ans
        eset = {tuple(e) for e in it["edges"]}
        for u, v in zip(gp[:-1], gp[1:]):
            assert (u, v) in eset


def test_collate_shapes_and_leftpad():
    insts = _gen(n=8, hops=3)
    vocab = build_vocab(max_concepts=200)
    b = collate(insts, vocab, max_steps=6)
    B = len(insts)
    assert b.prompt_ids.shape[0] == B and b.answer.shape == (B,)
    assert b.path_targets.shape == (B, 6)
    # left-padding means every row ends on a real token, whatever its length
    assert np.all(b.prompt_mask[:, -1] == 1.0)
    assert np.all(b.path_len == np.minimum(b.n_hops, 6))


# ---- model ----
def test_transformer_forward_and_feedback():
    cfg = ModelConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=4)
    m = Transformer(cfg, key=jax.random.PRNGKey(0))
    x = m.embed(jnp.arange(7))
    logits, hidden = m(x)
    assert logits.shape == (7, 50) and hidden.shape == (7, 32)
    # the latent step: append the last hidden as the next input embedding
    x2 = jnp.concatenate([x, hidden[-1:]], 0)
    l2, h2 = m(x2)
    assert l2.shape == (8, 50)


def test_init_logits_start_near_chance():
    """Tied embeddings at Equinox's N(0, 1) default make step-0 CE ~130 nats."""
    cfg = ModelConfig(vocab_size=24, d_model=128, n_layers=2, n_heads=4)
    m = Transformer(cfg, key=jax.random.PRNGKey(0))
    logits, _ = m(m.embed(jnp.arange(16)))
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.zeros((16,), jnp.int32))
    assert float(jnp.mean(ce)) < 3 * np.log(24)


# ---- halting ----
def test_halting_sums_to_one():
    for _ in range(5):
        lam = jax.random.uniform(jax.random.PRNGKey(_), (7,))
        p = halting_distribution(lam)
        assert abs(float(p.sum()) - 1.0) < 1e-4
        assert jnp.all(p >= 0)


def test_latent_unroll_shape():
    vocab = build_vocab(max_concepts=64)
    mcfg = ModelConfig(vocab_size=vocab.size, d_model=32, n_layers=2, n_heads=4)
    model = ReverieModel(mcfg, key=jax.random.PRNGKey(0))
    Sp, K, d = 10, 6, 32
    embeds = jax.random.normal(jax.random.PRNGKey(1), (Sp, d))
    valid = jnp.ones((Sp,))
    Y = latent_unroll(model, embeds, valid, K)
    assert Y.shape == (K + 1, d)          # y_0 .. y_K
    Y0 = latent_unroll(model, embeds, valid, 0)
    assert Y0.shape == (1, d)


# ---- gradients ----
def test_gradient_flows_through_latent_chain():
    insts = _gen(n=8, hops=4)
    vocab = build_vocab(max_concepts=200)
    b = collate(insts, vocab, max_steps=6)
    batch = {k: jnp.asarray(getattr(b, k)) for k in
             ["prompt_ids", "prompt_mask", "answer", "path_targets", "path_len",
              "n_hops", "cot_ids", "cot_mask", "cot_loss_mask"]}
    mcfg = ModelConfig(vocab_size=vocab.size, d_model=32, n_layers=2, n_heads=4)
    model = ReverieModel(mcfg, key=jax.random.PRNGKey(0))
    cfg = ReverieConfig(max_steps=6, method="reverie")
    (loss, _), grads = eqx.filter_value_and_grad(
        lambda m: batch_loss(m, batch, cfg), has_aux=True)(model)
    # no gradient into the halt head would mean halting is not learned at all
    hg = np.linalg.norm(np.asarray(grads.halt_head.weight))
    assert hg > 0, "halt head got no gradient"
    # and gradient reaches the embeddings through the whole thought chain
    eg = np.linalg.norm(np.asarray(grads.transformer.tok_embed.weight))
    assert eg > 0 and np.isfinite(loss)


def test_all_methods_run():
    insts = _gen(n=8, hops=3)
    vocab = build_vocab(max_concepts=200)
    b = collate(insts, vocab, max_steps=6)
    batch = {k: jnp.asarray(getattr(b, k)) for k in
             ["prompt_ids", "prompt_mask", "answer", "path_targets", "path_len",
              "n_hops", "cot_ids", "cot_mask", "cot_loss_mask"]}
    mcfg = ModelConfig(vocab_size=vocab.size, d_model=32, n_layers=2, n_heads=4)
    model = ReverieModel(mcfg, key=jax.random.PRNGKey(0))
    for method in ["reverie", "coconut", "coconut_distill", "nocot", "cot"]:
        adaptive = method == "reverie"
        alpha = 1.0 if method in ("reverie", "coconut_distill") else 0.0
        cfg = ReverieConfig(max_steps=6, method=method, adaptive=adaptive, alpha_traj=alpha)
        loss, aux = batch_loss(model, batch, cfg)
        assert np.isfinite(float(loss)), f"{method} produced non-finite loss"


# ---- vmapped ensemble ----
def test_ensemble_matches_sequential_training():
    """The stacked ensemble has to take the same updates as one model at a time.

    Two replicas, the same data, different seeds, ten steps. Sequential uses
    train.make_train_step; the ensemble uses one vmapped scan. The only thing
    that differs is the order XLA reduces in.

    Ten steps is deliberate. On CPU the two paths come out bit-identical, but on
    a GPU the reduction order differs by ~1e-5 relative at step 1 and training
    amplifies it: by step 500 the weights are ~10% apart. Do not raise the step
    count here expecting this tolerance to hold.
    """
    from reverie.data import global_lengths
    from reverie.ensemble import (
        batch_index_stream,
        init_ensemble,
        make_chunk_runner,
        stack_splits,
        tokenize_split,
        train_fields,
    )
    from reverie.train import _to_jax, make_optimizer, make_train_step

    seeds, steps, bs = [0, 3], 10, 8
    insts = _gen(n=64, hops=3)
    vocab = build_vocab(max_concepts=200)
    plen, clen = global_lengths(insts, vocab)
    cfg = ReverieConfig(max_steps=4, method="reverie")
    mcfg = ModelConfig(vocab_size=vocab.size, d_model=32, n_layers=2, n_heads=4)
    idx = np.stack([batch_index_stream(s, len(insts), bs, steps) for s in seeds], axis=1)

    seq = []
    for r, s in enumerate(seeds):
        model = ReverieModel(mcfg, key=jax.random.PRNGKey(s))
        optim = make_optimizer(1e-3, 5, steps)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        train_step = make_train_step(optim, cfg)
        for t in range(steps):
            b = collate([insts[j] for j in idx[t, r]], vocab, cfg.max_steps,
                        prompt_len=plen, cot_len=clen)
            model, opt_state, _, _ = train_step(model, opt_state, _to_jax(b))
        seq.append(eqx.filter(model, eqx.is_array))

    fields = train_fields(cfg.method)
    ds = stack_splits([tokenize_split(insts, vocab, cfg.max_steps, plen, clen, fields)
                       for _ in seeds])
    arr, static, optim, opt_state = init_ensemble(mcfg, seeds, 1e-3, 5, steps)
    run_chunk = make_chunk_runner(optim, cfg, static)
    arr, opt_state, _ = run_chunk(arr, opt_state, ds, jnp.asarray(idx))

    for r in range(len(seeds)):
        got = jax.tree_util.tree_map(lambda x: x[r], arr)
        d = max(float(jnp.max(jnp.abs(a - b))) for a, b in
                zip(jax.tree_util.tree_leaves(seq[r]), jax.tree_util.tree_leaves(got)))
        assert d < 1e-4, f"replica {r} drifted from its sequential run by {d}"


def test_ensemble_init_is_the_standalone_init():
    from reverie.ensemble import build_replicas, unstack_model

    mcfg = ModelConfig(vocab_size=32, d_model=32, n_layers=2, n_heads=4)
    seeds = [0, 7, 11]
    arr, static = build_replicas(mcfg, seeds)
    for i, s in enumerate(seeds):
        want = ReverieModel(mcfg, key=jax.random.PRNGKey(s))
        got = unstack_model(arr, static, i)
        for a, b in zip(jax.tree_util.tree_leaves(eqx.filter(want, eqx.is_array)),
                        jax.tree_util.tree_leaves(eqx.filter(got, eqx.is_array))):
            assert jnp.array_equal(a, b)
