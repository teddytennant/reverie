"""Decoder-only transformer core (Equinox), built for latent reasoning.

Two design choices distinguish this from a vanilla LM implementation, both
required by continuous-latent-space reasoning:

1. ``__call__`` consumes **input embeddings** ``[T, D]`` directly rather than
   token ids. This lets a caller build a sequence that mixes ordinary token
   embeddings with *continuous thought* vectors (the last-layer hidden state
   fed back as the next input), which is the whole point of the method.

2. ``__call__`` returns both ``logits`` and the **final-layer hidden states**
   ``[T, D]`` (post final norm — the exact tensor the LM head reads). That
   hidden state at the last position is what gets recycled as the next latent
   input.

The module operates on a single (un-batched) sequence; batch with ``jax.vmap``.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_ff: int | None = None  # None -> ~8/3 * d_model, rounded to a multiple of 64
    max_seq_len: int = 512
    rope_base: float = 10_000.0
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads

    @property
    def ffn_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        raw = int(8 * self.d_model / 3)
        return ((raw + 63) // 64) * 64


class RMSNorm(eqx.Module):
    weight: Float[Array, " d"]
    eps: float = eqx.field(static=True)

    def __init__(self, dim: int, eps: float = 1e-5):
        self.weight = jnp.ones((dim,))
        self.eps = eps

    def __call__(self, x: Float[Array, "... d"]) -> Float[Array, "... d"]:
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + self.eps)
        return normed * self.weight


def _rope_cos_sin(
    positions: Int[Array, " t"], head_dim: int, base: float
) -> tuple[Float[Array, "t d"], Float[Array, "t d"]]:
    """Rotary position embedding tables for the given absolute positions."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]  # [T, half]
    # duplicate to full head_dim (interleave-free "GPT-NeoX" style)
    angles = jnp.concatenate([angles, angles], axis=-1)  # [T, head_dim]
    return jnp.cos(angles), jnp.sin(angles)


def _rotate_half(x: Float[Array, "... d"]) -> Float[Array, "... d"]:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rope(
    x: Float[Array, "t h d"],
    cos: Float[Array, "t d"],
    sin: Float[Array, "t d"],
) -> Float[Array, "t h d"]:
    cos = cos[:, None, :]  # [T, 1, head_dim]
    sin = sin[:, None, :]
    return x * cos + _rotate_half(x) * sin


class Attention(eqx.Module):
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, cfg: ModelConfig, *, key: PRNGKeyArray):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.wq = eqx.nn.Linear(d, d, use_bias=False, key=k1)
        self.wk = eqx.nn.Linear(d, d, use_bias=False, key=k2)
        self.wv = eqx.nn.Linear(d, d, use_bias=False, key=k3)
        self.wo = eqx.nn.Linear(d, d, use_bias=False, key=k4)

    def __call__(
        self,
        x: Float[Array, "t d"],
        cos: Float[Array, "t hd"],
        sin: Float[Array, "t hd"],
        mask: Float[Array, "t t"],
    ) -> Float[Array, "t d"]:
        t = x.shape[0]
        q = jax.vmap(self.wq)(x).reshape(t, self.n_heads, self.head_dim)
        k = jax.vmap(self.wk)(x).reshape(t, self.n_heads, self.head_dim)
        v = jax.vmap(self.wv)(x).reshape(t, self.n_heads, self.head_dim)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # [H, T, T] attention scores
        scores = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(self.head_dim)
        scores = scores + mask[None, :, :]
        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hts,shd->thd", attn, v).reshape(t, self.n_heads * self.head_dim)
        return jax.vmap(self.wo)(out)


class SwiGLU(eqx.Module):
    w_gate: eqx.nn.Linear
    w_up: eqx.nn.Linear
    w_down: eqx.nn.Linear

    def __init__(self, cfg: ModelConfig, *, key: PRNGKeyArray):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w_gate = eqx.nn.Linear(cfg.d_model, cfg.ffn_dim, use_bias=False, key=k1)
        self.w_up = eqx.nn.Linear(cfg.d_model, cfg.ffn_dim, use_bias=False, key=k2)
        self.w_down = eqx.nn.Linear(cfg.ffn_dim, cfg.d_model, use_bias=False, key=k3)

    def __call__(self, x: Float[Array, "t d"]) -> Float[Array, "t d"]:
        gate = jax.nn.silu(jax.vmap(self.w_gate)(x))
        up = jax.vmap(self.w_up)(x)
        return jax.vmap(self.w_down)(gate * up)


class Block(eqx.Module):
    attn_norm: RMSNorm
    attn: Attention
    ffn_norm: RMSNorm
    ffn: SwiGLU

    def __init__(self, cfg: ModelConfig, *, key: PRNGKeyArray):
        ka, kf = jax.random.split(key, 2)
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, key=ka)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg, key=kf)

    def __call__(self, x, cos, sin, mask):
        x = x + self.attn(jax.vmap(self.attn_norm)(x), cos, sin, mask)
        x = x + self.ffn(jax.vmap(self.ffn_norm)(x))
        return x


class Transformer(eqx.Module):
    """Decoder-only transformer over input embeddings.

    Returns ``(logits, hidden)`` where ``hidden`` is the post-final-norm
    last-layer state (the tensor the LM head consumes). Feed ``hidden[-1]`` back
    as the next input embedding to reason in latent space.
    """

    tok_embed: eqx.nn.Embedding
    blocks: list[Block]
    final_norm: RMSNorm
    lm_head: eqx.nn.Linear | None
    cfg: ModelConfig = eqx.field(static=True)

    def __init__(self, cfg: ModelConfig, *, key: PRNGKeyArray):
        ke, kh, *kb = jax.random.split(key, cfg.n_layers + 2)
        self.cfg = cfg
        self.tok_embed = eqx.nn.Embedding(cfg.vocab_size, cfg.d_model, key=ke)
        self.blocks = [Block(cfg, key=kb[i]) for i in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = (
            None
            if cfg.tie_embeddings
            else eqx.nn.Linear(cfg.d_model, cfg.vocab_size, use_bias=False, key=kh)
        )

    def embed(self, ids: Int[Array, " t"]) -> Float[Array, "t d"]:
        return jax.vmap(self.tok_embed)(ids)

    def project(self, hidden: Float[Array, "t d"]) -> Float[Array, "t v"]:
        if self.lm_head is None:  # weight tying
            return hidden @ self.tok_embed.weight.T
        return jax.vmap(self.lm_head)(hidden)

    def backbone(
        self,
        x: Float[Array, "t d"],
        positions: Int[Array, " t"],
        attn_mask: Float[Array, "t t"] | None = None,
    ) -> Float[Array, "t d"]:
        """Run the stack over input embeddings.

        ``attn_mask`` is an *additive* [T, T] bias (0 = attend, -inf = block).
        If None, a plain causal mask is used. Latent reasoning passes a
        causal-AND-validity mask so padding positions are never attended.
        """
        t = x.shape[0]
        cos, sin = _rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_base)
        if attn_mask is None:
            attn_mask = jnp.where(
                jnp.arange(t)[:, None] >= jnp.arange(t)[None, :], 0.0, -jnp.inf
            )
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask)
        return jax.vmap(self.final_norm)(x)

    def __call__(
        self,
        x: Float[Array, "t d"],
        positions: Int[Array, " t"] | None = None,
        attn_mask: Float[Array, "t t"] | None = None,
    ) -> tuple[Float[Array, "t v"], Float[Array, "t d"]]:
        if positions is None:
            positions = jnp.arange(x.shape[0])
        hidden = self.backbone(x, positions, attn_mask)
        return self.project(hidden), hidden
