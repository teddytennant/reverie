"""Latent reasoning: continuous thoughts, learned halting, and the loss.

Given a left-padded prompt of length ``Sp`` we append ``K`` thought slots.
Thought ``t``, at column ``Sp+t``, takes its input embedding from the last-layer
hidden at column ``Sp+t-1``, fed back with no projection, as in Coconut.

The part that is easy to miss: attention is causal, so once thoughts
``0..m-1`` are filled, the hidden at column ``Sp+m-1`` can never change again.
One length-``K`` ``lax.scan`` therefore hands back every intermediate read-out,
no re-running the stack per depth.

The answer is a single concept token, so the answer distribution after ``m``
thoughts is the tied LM head applied to ``y_m``, and all ``K+1`` depths fall out
of one batched matmul.

Training is single-stage and differentiable, no RL and no curriculum. Four
terms, each toggleable for ablations:

    L = Σ_m p_m · CE(answer, W y_m)                     # PonderNet expected loss
      + α · Σ_{i=1}^{k} CE(path[i], W y_i)              # trajectory distillation
      + γ · (−log p_k)                                  # halt at teacher depth k
      + β · KL(p ‖ Geometric(λ_prior))                  # anti-collapse compute prior (training)

p_m is the PonderNet halting distribution over depth m∈{0..K}; k = n_hops is
the teacher's reasoning depth for that instance. Turning flags off recovers
curriculum-free Coconut (α=0, non-adaptive) and trajectory-distilled
fixed-depth Coconut (α>0, non-adaptive).
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from reverie.model import ModelConfig, Transformer

_NEG_INF = -1e30
_EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class ReverieConfig:
    max_steps: int = 6            # K: number of latent thought slots
    method: str = "reverie"       # reverie | coconut | coconut_distill
    adaptive: bool = True         # learned halting distribution vs fixed depth K
    alpha_traj: float = 1.0       # trajectory-distillation weight
    gamma_halt: float = 1.0       # depth-supervision weight (halt at n_hops)
    beta_reg: float = 0.01        # KL-to-geometric-prior weight (training anti-collapse)
    lambda_prior: float = 0.2     # geometric prior halt rate; untruncated E[depth]=(1-λ)/λ on {0,1,...}


class ReverieModel(eqx.Module):
    """Transformer backbone + a scalar halting head."""

    transformer: Transformer
    halt_head: eqx.nn.Linear

    def __init__(self, model_cfg: ModelConfig, *, key: PRNGKeyArray):
        kt, kh = jax.random.split(key)
        self.transformer = Transformer(model_cfg, key=kt)
        self.halt_head = eqx.nn.Linear(model_cfg.d_model, 1, use_bias=True, key=kh)

    # ---- convenience passthroughs
    def embed(self, ids):
        return self.transformer.embed(ids)

    def project(self, hidden):
        return self.transformer.project(hidden)


def latent_unroll(
    model: ReverieModel,
    prompt_embeds: Float[Array, "sp d"],
    prompt_valid: Float[Array, " sp"],
    K: int,
) -> Float[Array, "k1 d"]:
    """Fill K continuous thoughts; return the read-outs y_0..y_K, shape [K+1, d].

    y_0 reads out the prompt alone, y_m after m thoughts. One example at a time;
    vmap over the batch.
    """
    Sp, d = prompt_embeds.shape
    L = Sp + K
    positions = jnp.arange(L)
    valid = jnp.concatenate([prompt_valid, jnp.ones((K,))], axis=0)
    # Causal, and pad columns blocked outright: the prompt is left-padded, so
    # without the validity term the thought slots would attend to <pad>.
    causal = jnp.arange(L)[:, None] >= jnp.arange(L)[None, :]
    allowed = causal & (valid[None, :] > 0)
    add_mask = jnp.where(allowed, 0.0, _NEG_INF)

    E0 = jnp.concatenate([prompt_embeds, jnp.zeros((K, d))], axis=0)

    if K == 0:
        H = model.transformer.backbone(E0, positions, add_mask)
        return H[Sp - 1][None, :]

    def step(E, i):
        H = model.transformer.backbone(E, positions, add_mask)
        cur = H[Sp + i - 1]                      # y_i (final by causality)
        E = E.at[Sp + i].set(cur)                # thought i's input embedding
        return E, cur

    # Each step writes one thought and emits the read-out that is now frozen,
    # so the scan yields y_0..y_{K-1}; y_K needs one more pass.
    E_full, ys = jax.lax.scan(step, E0, jnp.arange(K))   # ys: [K, d]
    H_final = model.transformer.backbone(E_full, positions, add_mask)
    yK = H_final[Sp + K - 1]
    return jnp.concatenate([ys, yK[None, :]], axis=0)     # [K+1, d]


def halting_distribution(lam: Float[Array, " k1"]) -> Float[Array, " k1"]:
    """PonderNet halting distribution p_m = λ_m Π_{j<m}(1-λ_j).

    λ at the deepest slot is pinned to 1 so the remaining mass lands there and p
    sums to 1 without a separate normalization.
    """
    lam = jnp.clip(lam, _EPS, 1.0 - _EPS)
    lam = lam.at[-1].set(1.0)
    one_minus = 1.0 - lam
    prefix = jnp.concatenate([jnp.ones((1,)), jnp.cumprod(one_minus)[:-1]])
    return lam * prefix


def geometric_prior(n: int, lam_p: float) -> Float[Array, " n"]:
    """Geom(λ) over {0..n-1}, truncated and renormalized.

    0-indexed because depth 0 is a real option here: answer off the prompt with
    no thoughts at all.
    """
    m = jnp.arange(n)
    g = lam_p * (1.0 - lam_p) ** m
    return g / g.sum()


# ---- per-example loss cores --------------------------------------------------
def _answer_ce(logits: Float[Array, "k1 v"], answer: int) -> Float[Array, " k1"]:
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, jnp.full((logits.shape[0],), answer)
    )


def reverie_example_loss(model, prompt_embeds, prompt_valid, answer,
                         path_targets, path_len, n_hops, cfg: ReverieConfig):
    """Full Reverie objective for one example. Returns (loss, aux dict)."""
    K = cfg.max_steps
    Y = latent_unroll(model, prompt_embeds, prompt_valid, K)   # [K+1, d]
    logits = model.project(Y)                                  # [K+1, V]
    ce = _answer_ce(logits, answer)                            # [K+1]

    lam = jax.nn.sigmoid(jax.vmap(model.halt_head)(Y)[:, 0])   # [K+1]
    p = halting_distribution(lam)                              # [K+1]

    if cfg.adaptive:
        l_task = jnp.sum(p * ce)
    else:
        l_task = ce[K]                                          # fixed depth K

    # trajectory distillation: read-out y_i should decode to path node i, i=1..k
    if cfg.alpha_traj > 0.0:
        idx = jnp.arange(1, K + 1)                             # y_1..y_K
        tgt = jnp.where(idx <= path_len, path_targets[jnp.minimum(idx - 1, K - 1)], 0)
        step_ce = optax.softmax_cross_entropy_with_integer_labels(logits[1:], tgt)
        mask = (idx <= path_len).astype(jnp.float32)
        l_traj = jnp.sum(step_ce * mask) / jnp.maximum(mask.sum(), 1.0)
    else:
        l_traj = jnp.zeros(())

    # depth supervision: halt exactly at the teacher's per-instance depth
    if cfg.adaptive and cfg.gamma_halt > 0.0:
        k = jnp.clip(n_hops, 0, K)
        l_halt = -jnp.log(p[k] + _EPS)
    else:
        l_halt = jnp.zeros(())

    # anti-collapse compute prior (training); inference Pareto dial is halt_bias
    if cfg.adaptive and cfg.beta_reg > 0.0:
        g = geometric_prior(K + 1, cfg.lambda_prior)
        l_reg = jnp.sum(p * (jnp.log(p + _EPS) - jnp.log(g + _EPS)))
    else:
        l_reg = jnp.zeros(())

    loss = l_task + cfg.alpha_traj * l_traj + cfg.gamma_halt * l_halt + cfg.beta_reg * l_reg
    expected_depth = jnp.sum(p * jnp.arange(K + 1))
    # adaptive: MAP halt depth readout; non-adaptive: full-depth K (matches l_task)
    if cfg.adaptive:
        nstar = jnp.argmax(p)
        pred = jnp.argmax(logits[nstar])
    else:
        pred = jnp.argmax(logits[K])
    aux = dict(l_task=l_task, l_traj=l_traj, l_halt=l_halt, l_reg=l_reg,
               expected_depth=expected_depth, correct=(pred == answer).astype(jnp.float32))
    return loss, aux


# ---- batched loss ------------------------------------------------------------
def batch_loss(model: ReverieModel, batch: dict, cfg: ReverieConfig):
    """Mean loss over a batch (dict of jnp arrays)."""
    embeds = jax.vmap(model.embed)(batch["prompt_ids"])         # [B, Sp, d]
    valid = batch["prompt_mask"]

    f = lambda e, v, a, pt, pl, nh: reverie_example_loss(
        model, e, v, a, pt, pl, nh, cfg)
    losses, auxes = jax.vmap(f)(
        embeds, valid, batch["answer"], batch["path_targets"],
        batch["path_len"], batch["n_hops"])

    loss = jnp.mean(losses)
    aux = {k: jnp.mean(v) for k, v in auxes.items()}
    return loss, aux
