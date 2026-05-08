from __future__ import annotations

import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
import netket as nk


def rotate_half(x: jax.Array) -> jax.Array:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return jnp.stack((-x_odd, x_even), axis=-1).reshape(x.shape)


def apply_rope_axis(
    x: jax.Array,
    positions: jax.Array,
    *,
    base: float = 10_000.0,
) -> jax.Array:
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}.")

    freq_dtype = jnp.float32
    inv_freq = 1.0 / (
        base ** (jnp.arange(0, head_dim, 2, dtype=freq_dtype) / head_dim)
    )
    freqs = positions.astype(freq_dtype)[:, None] * inv_freq[None, :]

    cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1).astype(x.dtype)
    sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1).astype(x.dtype)

    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + rotate_half(x) * sin


def apply_2d_axial_rope(
    x: jax.Array,
    rows: jax.Array,
    cols: jax.Array,
    *,
    base: float = 10_000.0,
) -> jax.Array:
    head_dim = x.shape[-1]
    if head_dim % 4 != 0:
        raise ValueError(
            f"2D axial RoPE requires head_dim divisible by 4, got {head_dim}."
        )

    x_row, x_col = jnp.split(x, 2, axis=-1)
    x_row = apply_rope_axis(x_row, rows, base=base)
    x_col = apply_rope_axis(x_col, cols, base=base)
    return jnp.concatenate([x_row, x_col], axis=-1)


def lattice_coordinates_2d(
    num_sites: int,
    lattice_shape: Tuple[int, int],
) -> tuple[jax.Array, jax.Array]:
    n_rows, n_cols = lattice_shape
    if n_rows * n_cols != num_sites:
        raise ValueError(
            f"lattice_shape={lattice_shape} does not match num_sites={num_sites}."
        )

    site_ids = jnp.arange(num_sites, dtype=jnp.int32)
    rows = site_ids // n_cols
    cols = site_ids % n_cols
    return rows, cols


class CausalSelfAttention2DRoPE(nn.Module):
    width: int
    num_heads: int
    rope_base: float = 10_000.0
    dropout: float = 0.0
    use_qkv_bias: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        rows: jax.Array,
        cols: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        batch_size, num_sites, channels = x.shape
        if channels != self.width:
            raise ValueError(f"Expected width={self.width}, got channels={channels}.")
        if self.width % self.num_heads != 0:
            raise ValueError(
                f"width={self.width} must be divisible by num_heads={self.num_heads}."
            )

        head_dim = self.width // self.num_heads
        qkv = nn.Dense(
            3 * self.width,
            use_bias=self.use_qkv_bias,
            name="qkv",
        )(x)
        qkv = qkv.reshape(batch_size, num_sites, 3, self.num_heads, head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)

        q = apply_2d_axial_rope(q, rows, cols, base=self.rope_base)
        k = apply_2d_axial_rope(k, rows, cols, base=self.rope_base)

        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k) * (head_dim ** -0.5)
        causal_mask = jnp.tril(jnp.ones((num_sites, num_sites), dtype=bool))
        attn_logits = jnp.where(
            causal_mask[None, None, :, :],
            attn_logits,
            jnp.finfo(attn_logits.dtype).min,
        )

        attn = jax.nn.softmax(attn_logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout, name="attn_dropout")(
            attn,
            deterministic=not train,
        )

        y = jnp.einsum("bhts,bshd->bthd", attn, v)
        y = y.reshape(batch_size, num_sites, self.width)
        y = nn.Dense(self.width, name="out_proj")(y)
        y = nn.Dropout(rate=self.dropout, name="out_dropout")(
            y,
            deterministic=not train,
        )
        return y


class CausalTransformerBlock2D(nn.Module):
    width: int
    num_heads: int
    mlp_dim: int
    rope_base: float = 10_000.0
    dropout: float = 0.0
    residual_scale: float = 1.0

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        rows: jax.Array,
        cols: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        h = nn.LayerNorm(name="attn_ln")(x)
        h = CausalSelfAttention2DRoPE(
            width=self.width,
            num_heads=self.num_heads,
            rope_base=self.rope_base,
            dropout=self.dropout,
            name="attn",
        )(h, rows=rows, cols=cols, train=train)
        x = x + self.residual_scale * h

        h = nn.LayerNorm(name="mlp_ln")(x)
        h = nn.Dense(self.mlp_dim, name="mlp_fc1")(h)
        h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout, name="mlp_dropout_1")(
            h,
            deterministic=not train,
        )
        h = nn.Dense(self.width, name="mlp_fc2")(h)
        h = nn.Dropout(rate=self.dropout, name="mlp_dropout_2")(
            h,
            deterministic=not train,
        )
        return x + self.residual_scale * h


class ComplexRoPETransformer2DAR(nk.models.AbstractARNN):
    hilbert: object
    lattice_shape: Tuple[int, int]

    embed_dim: int = 64
    width: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_dim: Optional[int] = None
    head_hidden: int = 256

    use_site_embedding: bool = True
    rope_base: float = 10_000.0
    dropout: float = 0.0
    residual_scale: float = 1.0

    machine_pow: int = 2
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    @nn.compact
    def conditionals_log_psi(
        self,
        inputs: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        batch_size, num_sites = inputs.shape
        local_size = self.hilbert.local_size
        if local_size != 2:
            raise ValueError(f"Expected local_size=2, got {local_size}.")

        spin_tokens = (inputs > 0).astype(jnp.int32)
        bos_token = jnp.full((batch_size, 1), 2, dtype=jnp.int32)
        prev_tokens = jnp.concatenate([bos_token, spin_tokens[:, :-1]], axis=1)

        x = nn.Embed(
            num_embeddings=3,
            features=self.embed_dim,
            name="tok_emb",
        )(prev_tokens)

        if self.use_site_embedding:
            site_ids = jnp.arange(num_sites, dtype=jnp.int32)[None, :]
            x = x + nn.Embed(
                num_embeddings=num_sites,
                features=self.embed_dim,
                name="site_emb",
            )(site_ids)

        x = nn.Dense(self.width, name="input_proj")(x)
        x = nn.tanh(x)
        x = nn.LayerNorm(name="input_ln")(x)
        x = nn.Dropout(rate=self.dropout, name="input_dropout")(
            x,
            deterministic=not train,
        )

        rows, cols = lattice_coordinates_2d(num_sites, self.lattice_shape)
        mlp_dim = self.mlp_dim or 4 * self.width

        for layer_idx in range(self.depth):
            x = CausalTransformerBlock2D(
                width=self.width,
                num_heads=self.num_heads,
                mlp_dim=mlp_dim,
                rope_base=self.rope_base,
                dropout=self.dropout,
                residual_scale=self.residual_scale,
                name=f"block_{layer_idx}",
            )(x, rows=rows, cols=cols, train=train)

        x = nn.LayerNorm(name="final_ln")(x)

        x_amp = nn.Dense(self.head_hidden, name="amp_fc1")(x)
        x_amp = nn.gelu(x_amp)
        x_amp = nn.Dense(self.head_hidden, name="amp_fc2")(x_amp)
        x_amp = nn.gelu(x_amp)
        amp_logits = nn.Dense(local_size, name="amp_head")(x_amp)
        amp_logpsi = jax.nn.log_softmax(amp_logits, axis=-1) / float(self.machine_pow)

        x_phase = nn.Dense(self.head_hidden, name="phase_fc1")(x)
        x_phase = nn.gelu(x_phase)
        x_phase = nn.Dense(self.head_hidden, name="phase_fc2")(x_phase)
        x_phase = nn.gelu(x_phase)
        phase_raw = nn.Dense(
            local_size,
            name="phase_head",
            kernel_init=nn.initializers.normal(stddev=self.phase_init_std),
            bias_init=nn.initializers.zeros_init(),
        )(x_phase)
        phase_raw = phase_raw - jnp.mean(phase_raw, axis=-1, keepdims=True)
        phase = self.phase_scale * jnp.tanh(phase_raw)

        return amp_logpsi + 1j * phase
