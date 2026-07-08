"""Training + evaluation harness (Optax, Equinox).

One code path trains every method (Reverie, ablations, Coconut reimpl, No-CoT,
CoT) via ``ReverieConfig.method``. Evaluation runs each method under its *own*
inference procedure so the comparison is fair:

  * reverie / coconut : adaptive (or fixed) latent depth, answer = tied-head
    argmax at the halted depth; also reports realized latent steps.
  * nocot             : answer from the prompt's last hidden.
  * cot               : greedy autoregressive generation of the reasoning chain,
    then read the answer concept token (no teacher forcing — real CoT).
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from reverie.data import ANS, Vocab, collate, global_lengths
from reverie.latent import (
    ReverieConfig,
    ReverieModel,
    batch_loss,
    halting_distribution,
    latent_unroll,
)


def make_optimizer(peak_lr: float, warmup: int, total: int, wd: float = 0.1):
    sched = optax.warmup_cosine_decay_schedule(
        0.0, peak_lr, warmup_steps=warmup, decay_steps=total, end_value=peak_lr * 0.1
    )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(sched, b1=0.9, b2=0.95, eps=1e-8, weight_decay=wd),
    )


def make_train_step(optim, cfg: ReverieConfig):
    @eqx.filter_jit(donate="all")
    def train_step(model, opt_state, batch):
        def lf(m):
            loss, aux = batch_loss(m, batch, cfg)
            return loss, aux

        (loss, aux), grads = eqx.filter_value_and_grad(lf, has_aux=True)(model)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optim.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    return train_step


def _to_jax(batch) -> dict:
    keys = ["prompt_ids", "prompt_mask", "answer", "path_targets", "path_len",
            "n_hops", "cot_ids", "cot_mask", "cot_loss_mask"]
    return {k: jnp.asarray(getattr(batch, k)) for k in keys}


# ---- inference ---------------------------------------------------------------
@eqx.filter_jit
def _predict_latent(model, prompt_ids, prompt_mask, K, adaptive, eps):
    """Per-batch latent inference. Returns (pred_token [B], steps_used [B])."""

    def one(pid, pmask):
        embeds = model.embed(pid)
        Y = latent_unroll(model, embeds, pmask, K)          # [K+1, d]
        logits = model.project(Y)                            # [K+1, V]
        lam = jax.nn.sigmoid(jax.vmap(model.halt_head)(Y)[:, 0])
        p = halting_distribution(lam)
        if adaptive:
            cum = jnp.cumsum(p)
            nstar = jnp.argmax(cum >= (1.0 - eps))           # first depth crossing budget
        else:
            nstar = jnp.array(K)
        pred = jnp.argmax(logits[nstar])
        return pred, nstar

    return jax.vmap(one)(prompt_ids, prompt_mask)


@eqx.filter_jit
def _predict_nocot(model, prompt_ids, prompt_mask):
    def one(pid, pmask):
        embeds = model.embed(pid)
        Y = latent_unroll(model, embeds, pmask, 0)
        return jnp.argmax(model.project(Y)[0])

    return jax.vmap(one)(prompt_ids, prompt_mask)


@eqx.filter_jit
def _cot_generate(model, prompt_ids, prompt_mask, gen_len):
    """Greedy-decode gen_len tokens after the (left-padded) prompt, right-extending."""
    B, Sp = prompt_ids.shape

    def one(pid, pmask):
        ids = jnp.concatenate([pid, jnp.zeros((gen_len,), jnp.int32)])
        valid = jnp.concatenate([pmask, jnp.zeros((gen_len,))])
        L = Sp + gen_len
        positions = jnp.arange(L)
        causal = jnp.arange(L)[:, None] >= jnp.arange(L)[None, :]

        def step(carry, t):
            ids, valid = carry
            allowed = causal & (valid[None, :] > 0)
            mask = jnp.where(allowed, 0.0, -1e30)
            H = model.transformer.backbone(model.embed(ids), positions, mask)
            nxt = jnp.argmax(model.project(H)[Sp + t - 1])   # token after last valid pos
            ids = ids.at[Sp + t].set(nxt)
            valid = valid.at[Sp + t].set(1.0)
            return (ids, valid), nxt

        (ids, _), gen = jax.lax.scan(step, (ids, valid), jnp.arange(gen_len))
        return gen

    return jax.vmap(one)(prompt_ids, prompt_mask)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(model, insts, vocab: Vocab, cfg: ReverieConfig,
             eps: float = 0.1, batch_size: int = 128,
             prompt_len: int | None = None, cot_len: int | None = None) -> dict:
    K = cfg.max_steps
    # fixed gen length for the whole eval call (compile CoT generation once)
    cot_gen_len = int(max(h["n_hops"] for h in insts)) * 6 + 4
    preds, steps, answers, hops = [], [], [], []
    for start in range(0, len(insts), batch_size):
        chunk = insts[start : start + batch_size]
        b = collate(chunk, vocab, max_steps=K, prompt_len=prompt_len, cot_len=cot_len)
        pid = jnp.asarray(b.prompt_ids)
        pmask = jnp.asarray(b.prompt_mask)
        if cfg.method == "nocot":
            pr = np.asarray(_predict_nocot(model, pid, pmask))
            st = np.zeros(len(chunk), np.int32)
        elif cfg.method == "cot":
            gen_len = cot_gen_len
            gen = np.asarray(_cot_generate(model, pid, pmask, gen_len))  # [b, gen_len]
            pr = np.zeros(len(chunk), np.int32)
            for i, row in enumerate(gen):
                where = np.where(row == ANS)[0]
                pr[i] = row[where[0] + 1] if len(where) and where[0] + 1 < len(row) else -1
            st = np.full(len(chunk), gen_len, np.int32)
        else:
            pr, st = _predict_latent(model, pid, pmask, K, cfg.adaptive, eps)
            pr, st = np.asarray(pr), np.asarray(st)
        preds.extend(pr.tolist())
        steps.extend(st.tolist())
        answers.extend(b.answer.tolist())
        hops.extend(b.n_hops.tolist())

    preds, answers, steps, hops = map(np.asarray, (preds, answers, steps, hops))
    correct = (preds == answers)
    acc_by_hop = {int(k): float(correct[hops == k].mean())
                  for k in sorted(set(hops.tolist())) if (hops == k).any()}
    return dict(
        acc=float(correct.mean()),
        mean_steps=float(steps.mean()),
        rho_steps_hops=_spearman(steps, hops),
        acc_by_hop=acc_by_hop,
        n=len(preds),
    )


# ---- training loop -----------------------------------------------------------
def train(model, train_insts, val_insts, vocab, cfg: ReverieConfig,
          steps: int, batch_size: int, peak_lr: float, warmup: int,
          seed: int = 0, eval_every: int = 500, log=print,
          prompt_len: int | None = None, cot_len: int | None = None) -> tuple:
    # fix padding shapes globally -> the model compiles once, not per batch
    if prompt_len is None or cot_len is None:
        sp, sc = global_lengths(train_insts + val_insts, vocab)
        prompt_len = prompt_len or sp
        cot_len = cot_len or sc
    optim = make_optimizer(peak_lr, warmup, steps)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optim, cfg)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_insts))
    ptr = 0
    history = []
    for step in range(1, steps + 1):
        if ptr + batch_size > len(order):
            order = rng.permutation(len(train_insts))
            ptr = 0
        idx = order[ptr : ptr + batch_size]
        ptr += batch_size
        batch = _to_jax(collate([train_insts[j] for j in idx], vocab, cfg.max_steps,
                                prompt_len=prompt_len, cot_len=cot_len))
        model, opt_state, loss, aux = train_step(model, opt_state, batch)
        if step % eval_every == 0 or step == steps:
            m = evaluate(model, val_insts, vocab, cfg,
                         prompt_len=prompt_len, cot_len=cot_len)
            history.append((step, float(loss), m))
            log(f"[{cfg.method}] step {step:5d}  loss {float(loss):7.3f}  "
                f"val_acc {m['acc']:.3f}  mean_steps {m['mean_steps']:.2f}  "
                f"rho {m['rho_steps_hops']:+.2f}")
    return model, history, (prompt_len, cot_len)
