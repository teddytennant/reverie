"""Train R seeds at once as a vmapped ensemble.

A 0.43M-parameter model at batch 64 cannot fill a modern GPU. Running R of them
as separate processes just gives the card R streams of tiny kernel launches to
interleave, and the card sits idle between them. Instead, stack R replicas along
a leading axis and vmap the loss, the gradient and the optimizer update over it.
Every launch is then R times wider and there is one dispatch stream, not R.

The science has to come out the same, so every per-replica input is exactly what
the sequential run would have produced:

  init      replica r is built from jax.random.PRNGKey(seed_r) and the leaves
            are stacked, so its weights are bit-identical to the standalone run.
  data      replica r gets its own generated splits (the seed drives the
            generator too), tokenized once up front instead of per step.
  batching  np.random.default_rng(seed_r) drives the same permutation and the
            same reshuffle-on-wrap as train.train.
  updates   one optax optimizer, vmapped, so each replica has its own moments,
            its own global-norm clip and the same schedule.

What is not identical is the floating-point reduction order. XLA picks different
kernels for a [R, B, ...] contraction than for [B, ...], so losses agree to
around float32 epsilon at step 1 and then drift the way any two runs of the same
model on different hardware drift. tests/test_core.py compares the weights after
ten steps, and scripts/ensemble_check.py diffs two directories of finished runs.

The step loop is a lax.scan over a chunk of steps with the tokenized dataset
resident on device and only batch indices flowing in. That removes the per-step
Python dispatch, which for a model this small is a large part of the wall clock.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from reverie.data import ANS, Vocab, collate
from reverie.latent import ReverieConfig, ReverieModel, batch_loss
from reverie.model import ModelConfig
from reverie.train import (
    _cot_generate,
    _predict_latent,
    _predict_nocot,
    make_optimizer,
    metrics_from_preds,
)

# fields batch_loss reads, per method
_LATENT_FIELDS = ("prompt_ids", "prompt_mask", "answer", "path_targets",
                  "path_len", "n_hops")
_NOCOT_FIELDS = ("prompt_ids", "prompt_mask", "answer")
_COT_FIELDS = ("prompt_ids", "prompt_mask", "cot_ids", "cot_mask", "cot_loss_mask")


def train_fields(method: str) -> tuple[str, ...]:
    if method == "cot":
        return _COT_FIELDS
    if method == "nocot":
        return _NOCOT_FIELDS
    return _LATENT_FIELDS


def stack_models(models: list) -> tuple:
    """Stack a list of identically-shaped Equinox modules into (arrays, static).

    ``arrays`` is a pytree whose leaves carry a leading replica axis; ``static``
    is the shared non-array skeleton (configs, None-valued tied heads). Combine
    one slice with it to get a plain model back.
    """
    parts = [eqx.partition(m, eqx.is_array) for m in models]
    arrs = [a for a, _ in parts]
    static = parts[0][1]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *arrs)
    return stacked, static


def unstack_model(arrays, static, i: int):
    return eqx.combine(jax.tree_util.tree_map(lambda x: x[i], arrays), static)


def build_replicas(mcfg: ModelConfig, seeds: list[int]) -> tuple:
    """One ReverieModel per seed, from the same key the standalone run uses."""
    models = [ReverieModel(mcfg, key=jax.random.PRNGKey(s)) for s in seeds]
    return stack_models(models)


# ---- data ---------------------------------------------------------------------
def tokenize_split(insts: list[dict], vocab: Vocab, max_steps: int,
                   prompt_len: int, cot_len: int, fields) -> dict:
    """Tokenize a whole split once. collate is row-independent given fixed
    padding widths, so row j here equals what collate would emit for instance j
    inside any batch."""
    b = collate(insts, vocab, max_steps, prompt_len=prompt_len, cot_len=cot_len)
    return {k: jnp.asarray(getattr(b, k)) for k in fields}


def stack_splits(per_seed: list[dict]) -> dict:
    return {k: jnp.stack([d[k] for d in per_seed]) for k in per_seed[0]}


def batch_index_stream(seed: int, n_train: int, batch_size: int, steps: int) -> np.ndarray:
    """The [steps, batch] index sequence train.train walks for this seed."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_train)
    ptr = 0
    out = np.empty((steps, batch_size), np.int32)
    for t in range(steps):
        if ptr + batch_size > len(order):
            order = rng.permutation(n_train)
            ptr = 0
        out[t] = order[ptr : ptr + batch_size]
        ptr += batch_size
    return out


# ---- training -----------------------------------------------------------------
def _gather(ds: dict, idx) -> dict:
    """ds fields are [R, N, ...], idx is [R, B] -> [R, B, ...]."""
    take = lambda d, i: d[i]
    return {k: jax.vmap(take)(v, idx) for k, v in ds.items()}


def make_chunk_runner(optim, cfg: ReverieConfig, static):
    """A jitted lax.scan over `chunk` training steps for all replicas at once."""

    def one_replica(arr_i, opt_i, batch_i):
        model = eqx.combine(arr_i, static)

        def lf(m):
            return batch_loss(m, batch_i, cfg)

        (loss, aux), grads = eqx.filter_value_and_grad(lf, has_aux=True)(model)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_i = optim.update(grads, opt_i, params)
        model = eqx.apply_updates(model, updates)
        return eqx.filter(model, eqx.is_array), opt_i, loss, aux

    @partial(jax.jit, donate_argnums=(0, 1))
    def run_chunk(arr, opt_state, ds, idx_chunk):
        # ds stays outside the scan carry. Threading the whole tokenized dataset
        # through the carry makes XLA copy it once per iteration.
        def step(carry, idx_t):
            arr, opt_state = carry
            batch = _gather(ds, idx_t)
            arr, opt_state, loss, aux = jax.vmap(one_replica)(arr, opt_state, batch)
            return (arr, opt_state), (loss, aux)

        (arr, opt_state), out = jax.lax.scan(step, (arr, opt_state), idx_chunk)
        return arr, opt_state, out

    return run_chunk


def init_ensemble(mcfg: ModelConfig, seeds: list[int], peak_lr: float,
                  warmup: int, total_steps: int, wd: float = 0.1):
    arr, static = build_replicas(mcfg, seeds)
    optim = make_optimizer(peak_lr, warmup, total_steps, wd)
    opt_state = jax.vmap(optim.init)(eqx.filter(arr, eqx.is_inexact_array))
    return arr, static, optim, opt_state


# ---- evaluation ---------------------------------------------------------------
def _ens(fn, static):
    """Lift a single-model function to the replica axis."""

    def wrapped(arr, *args):
        def one(a, *xs):
            return fn(eqx.combine(a, static), *xs)

        return jax.vmap(one)(arr, *args)

    return wrapped


def ensemble_evaluate(arr, static, eval_ds: dict, cands, answers, hops,
                      cfg: ReverieConfig, cot_gen_len: int,
                      eps: float = 0.1, batch_size: int = 128,
                      halt_bias: float = 0.0) -> list[dict]:
    """train.evaluate for every replica at once. Returns one metrics dict per
    replica, in seed order.

    ``eval_ds`` fields are [R, N, ...]; ``cands`` [R, N, 2]; ``answers``/``hops``
    are numpy [R, N]. The chunking over N matches train.evaluate so the same
    shapes get compiled.
    """
    R, N = cands.shape[0], cands.shape[1]
    K = cfg.max_steps
    preds = np.zeros((R, N), np.int32)
    steps = np.zeros((R, N), np.int32)

    # built once rather than per chunk, so the vmap is traced once per call
    # instead of once per chunk
    if cfg.method == "nocot":
        predict = _ens(_predict_nocot, static)
    elif cfg.method == "cot":
        predict = _ens(lambda m, p, q, c: _cot_generate(m, p, q, c, cot_gen_len), static)
    else:
        predict = _ens(
            lambda m, p, q, c: _predict_latent(m, p, q, c, K, cfg.adaptive, eps, halt_bias),
            static,
        )

    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        out = predict(arr, eval_ds["prompt_ids"][:, start:stop],
                      eval_ds["prompt_mask"][:, start:stop], cands[:, start:stop])
        if cfg.method == "nocot":
            preds[:, start:stop] = np.asarray(out)
        elif cfg.method == "cot":
            gen, candp = np.asarray(out[0]), np.asarray(out[1])
            for r in range(R):
                for i, row in enumerate(gen[r]):
                    w = np.where(row == ANS)[0]
                    # answer = candidate preferred right after the generated <ans>
                    preds[r, start + i] = (candp[r, i, w[0] + 1]
                                           if len(w) and w[0] + 1 < cot_gen_len else -1)
            steps[:, start:stop] = cot_gen_len
        else:
            preds[:, start:stop] = np.asarray(out[0])
            steps[:, start:stop] = np.asarray(out[1])

    return [metrics_from_preds(preds[r], answers[r], steps[r], hops[r])
            for r in range(R)]
