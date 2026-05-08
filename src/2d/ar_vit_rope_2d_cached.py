from __future__ import annotations
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
import netket as nk


# -----------------------------------------------------------------------------
# RoPE utilities
# -----------------------------------------------------------------------------


def rotate_half(x: jax.Array) -> jax.Array:
    """Pairs even/odd channels and applies a 90-degree rotation."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return jnp.stack((-x_odd, x_even), axis=-1).reshape(x.shape)


def apply_rope_axis(
    x: jax.Array,
    positions: jax.Array,
    *,
    base: float = 10_000.0,
) -> jax.Array:
    """
    Applies 1D rotary embeddings along a single coordinate axis.

    Args:
        x: Array of shape (B, T, H, D_axis).
        positions: Integer/float array of shape (T,).
        base: RoPE base.
    """
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
    """
    Applies axial 2D RoPE by splitting each attention head into row and column halves.

    Args:
        x: Array of shape (B, T, H, D).
        rows: Row coordinates of shape (T,).
        cols: Column coordinates of shape (T,).
    """
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
    """Returns row/column coordinates for row-major site ordering."""
    n_rows, n_cols = lattice_shape
    if n_rows * n_cols != num_sites:
        raise ValueError(
            f"lattice_shape={lattice_shape} does not match num_sites={num_sites}."
        )

    site_ids = jnp.arange(num_sites, dtype=jnp.int32)
    rows = site_ids // n_cols
    cols = site_ids % n_cols
    return rows, cols


# -----------------------------------------------------------------------------
# Transformer blocks
# -----------------------------------------------------------------------------


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
        """Full causal self-attention path used by NetKet/full scoring."""
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

    @nn.compact
    def decode(
        self,
        x: jax.Array,
        *,
        row_t: jax.Array,
        col_t: jax.Array,
        t: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        One-token cached attention step.

        Args:
            x: Current hidden state, shape (B, 1, width).
            row_t: Current row coordinate, shape (1,).
            col_t: Current column coordinate, shape (1,).
            t: Current site index, scalar int.
            k_cache: Layer key cache, shape (B, N, H, Dh).
            v_cache: Layer value cache, shape (B, N, H, Dh).

        Returns:
            y: Current output, shape (B, 1, width).
            k_cache: Updated key cache.
            v_cache: Updated value cache.
        """
        batch_size, one, channels = x.shape
        if one != 1:
            raise ValueError(f"decode expects one token, got sequence length {one}.")
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
        qkv = qkv.reshape(batch_size, 1, 3, self.num_heads, head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)

        q = apply_2d_axial_rope(q, row_t, col_t, base=self.rope_base)
        k = apply_2d_axial_rope(k, row_t, col_t, base=self.rope_base)

        # Cache has shape (B, N, H, Dh). Insert at position t.
        k = k.astype(k_cache.dtype)
        v = v.astype(v_cache.dtype)
        # k_cache = jax.lax.dynamic_update_slice(k_cache, k, (0, t, 0, 0))
        # v_cache = jax.lax.dynamic_update_slice(v_cache, v, (0, t, 0, 0))
        t = t.astype(jnp.int32)
        zero = jnp.asarray(0, dtype=t.dtype)

        k_cache = jax.lax.dynamic_update_slice(
            k_cache,
            k,
            (zero, t, zero, zero),
        )
        v_cache = jax.lax.dynamic_update_slice(
            v_cache,
            v,
            (zero, t, zero, zero),
        )

        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k_cache)
        attn_logits = attn_logits * (head_dim ** -0.5)

        max_len = k_cache.shape[1]
        prefix_mask = jnp.arange(max_len, dtype=t.dtype) <= t
        attn_logits = jnp.where(
            prefix_mask[None, None, None, :],
            attn_logits,
            jnp.finfo(attn_logits.dtype).min,
        )

        attn = jax.nn.softmax(attn_logits, axis=-1)
        attn = nn.Dropout(rate=self.dropout, name="attn_dropout")(
            attn,
            deterministic=not train,
        )

        y = jnp.einsum("bhts,bshd->bthd", attn, v_cache)
        y = y.reshape(batch_size, 1, self.width)
        y = nn.Dense(self.width, name="out_proj")(y)
        y = nn.Dropout(rate=self.dropout, name="out_dropout")(
            y,
            deterministic=not train,
        )
        return y, k_cache, v_cache


class CausalTransformerBlock2D(nn.Module):
    width: int
    num_heads: int
    mlp_dim: int
    rope_base: float = 10_000.0
    dropout: float = 0.0
    residual_scale: float = 1.0
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
        h = nn.LayerNorm(name="attn_ln")(x)
        h = CausalSelfAttention2DRoPE(
            width=self.width,
            num_heads=self.num_heads,
            rope_base=self.rope_base,
            dropout=self.dropout,
            use_qkv_bias=self.use_qkv_bias,
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

    @nn.compact
    def decode(
        self,
        x: jax.Array,
        *,
        row_t: jax.Array,
        col_t: jax.Array,
        t: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        h = nn.LayerNorm(name="attn_ln")(x)
        h, k_cache, v_cache = CausalSelfAttention2DRoPE(
            width=self.width,
            num_heads=self.num_heads,
            rope_base=self.rope_base,
            dropout=self.dropout,
            use_qkv_bias=self.use_qkv_bias,
            name="attn",
        ).decode(
            h,
            row_t=row_t,
            col_t=col_t,
            t=t,
            k_cache=k_cache,
            v_cache=v_cache,
            train=train,
        )
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
        x = x + self.residual_scale * h
        return x, k_cache, v_cache


# -----------------------------------------------------------------------------
# NetKet-compatible autoregressive model
# -----------------------------------------------------------------------------


class ComplexRoPETransformer2DAR(nk.models.AbstractARNN):
    """
    Complex autoregressive Transformer NQS with 2D axial RoPE.

    NetKet compatibility:
        - `conditionals_log_psi(inputs, train=False)` returns shape (B, N, local_size).
        - The model subclasses `nk.models.AbstractARNN`.

    Cached inference:
        - `decode_step(prev_token, t, cache, train=False)` returns one conditional
          distribution of shape (B, local_size) and an updated KV cache.
    """

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
    use_qkv_bias: bool = True

    machine_pow: int = 2
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    @nn.compact
    def conditionals_log_psi(
        self,
        inputs: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        """
        Full parallel conditional log-psi path used by NetKet.

        Args:
            inputs: Spin configurations with values in {-1,+1}, shape (N,) or (B,N).
            train: Whether to enable dropout.

        Returns:
            Conditional log psi, shape (B,N,local_size), complex-valued.
        """
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        batch_size, num_sites = inputs.shape
        local_size = self.hilbert.local_size
        if local_size != 2:
            raise ValueError(f"Expected local_size=2, got {local_size}.")

        spin_tokens = (inputs > 0).astype(jnp.int32)  # -1 -> 0, +1 -> 1
        bos_token = jnp.full((batch_size, 1), 2, dtype=jnp.int32)
        prev_tokens = jnp.concatenate([bos_token, spin_tokens[:, :-1]], axis=1)

        x = self._embed_tokens(prev_tokens, num_sites=num_sites, train=train)

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
                use_qkv_bias=self.use_qkv_bias,
                name=f"block_{layer_idx}",
            )(x, rows=rows, cols=cols, train=train)

        x = nn.LayerNorm(name="final_ln")(x)
        return self._heads(x, local_size=local_size)

    @nn.compact
    def decode_step(
        self,
        prev_token: jax.Array,
        t: jax.Array,
        cache: dict[str, jax.Array],
        train: bool = False,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """
        One cached autoregressive step.

        Args:
            prev_token: Previous token, shape (B,), with values:
                0 for spin -1, 1 for spin +1, 2 for BOS.
            t: Current site index, scalar int in [0, N).
            cache: Dict with keys "k" and "v" of shape
                (depth, B, N, num_heads, head_dim).
            train: Usually False. Dropout during cached sampling is normally not useful.

        Returns:
            cond_logpsi: Shape (B, local_size), complex-valued conditional log psi.
            new_cache: Updated KV cache.
        """
        if prev_token.ndim != 1:
            raise ValueError(f"prev_token must have shape (B,), got {prev_token.shape}.")

        num_sites = self.lattice_shape[0] * self.lattice_shape[1]
        local_size = self.hilbert.local_size
        if local_size != 2:
            raise ValueError(f"Expected local_size=2, got {local_size}.")

        x = self._embed_decode_token(
            prev_token,
            t=t,
            num_sites=num_sites,
            train=train,
        )

        rows, cols = lattice_coordinates_2d(num_sites, self.lattice_shape)
        row_t = jax.lax.dynamic_slice(rows, (t,), (1,))
        col_t = jax.lax.dynamic_slice(cols, (t,), (1,))

        mlp_dim = self.mlp_dim or 4 * self.width
        k_all = cache["k"]
        v_all = cache["v"]

        for layer_idx in range(self.depth):
            x, k_layer, v_layer = CausalTransformerBlock2D(
                width=self.width,
                num_heads=self.num_heads,
                mlp_dim=mlp_dim,
                rope_base=self.rope_base,
                dropout=self.dropout,
                residual_scale=self.residual_scale,
                use_qkv_bias=self.use_qkv_bias,
                name=f"block_{layer_idx}",
            ).decode(
                x,
                row_t=row_t,
                col_t=col_t,
                t=t,
                k_cache=k_all[layer_idx],
                v_cache=v_all[layer_idx],
                train=train,
            )
            k_all = k_all.at[layer_idx].set(k_layer)
            v_all = v_all.at[layer_idx].set(v_layer)

        x = nn.LayerNorm(name="final_ln")(x)
        cond_logpsi = self._heads(x, local_size=local_size)[:, 0, :]

        return cond_logpsi, {"k": k_all, "v": v_all}

    def _embed_tokens(
        self,
        tokens: jax.Array,
        *,
        num_sites: int,
        train: bool,
    ) -> jax.Array:
        """Shared token/site embedding stem."""
        batch_size, seq_len = tokens.shape

        x = nn.Embed(
            num_embeddings=3,
            features=self.embed_dim,
            name="tok_emb",
        )(tokens)

        if self.use_site_embedding:
            # For full path, seq_len == N and positions are 0..N-1.
            # For decode path, seq_len == 1, but the caller has already provided
            # token at the current position. We infer the current site from the
            # requested slice by requiring decode_step to pass a single token and
            # then overwrite below using a dynamic site id.
            #
            # To keep this helper simple, full path uses arange(seq_len). The
            # decode path must call this only with seq_len=1 and then site id 0
            # would be wrong. Therefore decode_step does not use this helper's
            # default site embedding when `seq_len == 1`; see explicit branch.
            site_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
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
        return x

    def _embed_decode_token(
        self,
        token: jax.Array,
        *,
        t: jax.Array,
        num_sites: int,
        train: bool,
    ) -> jax.Array:
        """Embedding stem for one decode token at site t."""
        x = nn.Embed(
            num_embeddings=3,
            features=self.embed_dim,
            name="tok_emb",
        )(token[:, None])

        if self.use_site_embedding:
            site_id = jnp.broadcast_to(t.astype(jnp.int32), (1, 1))
            x = x + nn.Embed(
                num_embeddings=num_sites,
                features=self.embed_dim,
                name="site_emb",
            )(site_id)

        x = nn.Dense(self.width, name="input_proj")(x)
        x = nn.tanh(x)
        x = nn.LayerNorm(name="input_ln")(x)
        x = nn.Dropout(rate=self.dropout, name="input_dropout")(
            x,
            deterministic=not train,
        )
        return x

    def _heads(self, x: jax.Array, *, local_size: int) -> jax.Array:
        """Shared amplitude and phase heads."""
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


# -----------------------------------------------------------------------------
# Cache helpers and cached sampling
# -----------------------------------------------------------------------------


def make_kv_cache(
    model: ComplexRoPETransformer2DAR,
    *,
    batch_size: int,
    num_sites: Optional[int] = None,
    dtype=jnp.float32,
) -> dict[str, jax.Array]:
    """Creates an empty KV cache for `model.decode_step`."""
    if num_sites is None:
        num_sites = model.lattice_shape[0] * model.lattice_shape[1]

    if model.width % model.num_heads != 0:
        raise ValueError(
            f"width={model.width} must be divisible by num_heads={model.num_heads}."
        )

    head_dim = model.width // model.num_heads
    if head_dim % 4 != 0:
        raise ValueError(
            f"2D axial RoPE requires head_dim divisible by 4, got {head_dim}."
        )

    shape = (model.depth, batch_size, num_sites, model.num_heads, head_dim)
    return {
        "k": jnp.zeros(shape, dtype=dtype),
        "v": jnp.zeros(shape, dtype=dtype),
    }


def sample_cached_apply(
    model: ComplexRoPETransformer2DAR,
    variables: dict,
    key: jax.Array,
    *,
    batch_size: int,
    cache_dtype=jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """
    Cached autoregressive sampler.

    Args:
        model: `ComplexRoPETransformer2DAR` instance.
        variables: Flax variables, usually {"params": params}.
        key: PRNG key.
        batch_size: Number of samples.
        cache_dtype: dtype of K/V cache.

    Returns:
        samples: Spin samples in {-1,+1}, shape (B,N).
        logpsi: Sum of selected conditional log-psi values, shape (B,).
    """
    num_sites = model.lattice_shape[0] * model.lattice_shape[1]
    cache = make_kv_cache(
        model,
        batch_size=batch_size,
        num_sites=num_sites,
        dtype=cache_dtype,
    )

    samples0 = jnp.zeros((batch_size, num_sites), dtype=jnp.int8)
    logpsi0 = jnp.zeros((batch_size,), dtype=jnp.complex64)

    # Token convention: 0 -> spin -1, 1 -> spin +1, 2 -> BOS.
    prev_token0 = jnp.full((batch_size,), 2, dtype=jnp.int32)

    keys = jax.random.split(key, num_sites)
    ts = jnp.arange(num_sites, dtype=jnp.int32)

    def body(carry, xs):
        cache, prev_token, samples, logpsi = carry
        t, key_t = xs

        cond_logpsi, cache = model.apply(
            variables,
            prev_token,
            t,
            cache,
            train=False,
            method=model.decode_step,
        )

        # amp_logpsi = log_softmax(logits) / machine_pow.
        # Therefore machine_pow * Re(cond_logpsi) are categorical log-probs.
        cat_log_probs = float(model.machine_pow) * jnp.real(cond_logpsi)
        token = jax.random.categorical(key_t, cat_log_probs, axis=-1).astype(jnp.int32)
        spin = (2 * token - 1).astype(jnp.int8)

        samples = samples.at[:, t].set(spin)
        chosen = cond_logpsi[jnp.arange(batch_size), token]
        logpsi = logpsi + chosen

        return (cache, token, samples, logpsi), None

    (_, _, samples, logpsi), _ = jax.lax.scan(
        body,
        (cache, prev_token0, samples0, logpsi0),
        (ts, keys),
    )
    return samples, logpsi


def cached_conditionals_for_inputs(
    model: ComplexRoPETransformer2DAR,
    variables: dict,
    inputs: jax.Array,
    *,
    cache_dtype=jnp.float32,
) -> jax.Array:
    """
    Computes all conditionals for fixed input configurations using decode_step.

    This is mainly a correctness/debug helper. For known full configurations,
    `conditionals_log_psi` is usually faster because it uses full parallel attention.

    Args:
        inputs: Spin configurations in {-1,+1}, shape (N,) or (B,N).

    Returns:
        conds: Shape (B,N,local_size).
    """
    if inputs.ndim == 1:
        inputs = inputs[None, :]
    batch_size, num_sites = inputs.shape

    cache = make_kv_cache(
        model,
        batch_size=batch_size,
        num_sites=num_sites,
        dtype=cache_dtype,
    )

    spin_tokens = (inputs > 0).astype(jnp.int32)
    bos = jnp.full((batch_size,), 2, dtype=jnp.int32)
    prev_tokens = jnp.concatenate([bos[:, None], spin_tokens[:, :-1]], axis=1)

    ts = jnp.arange(num_sites, dtype=jnp.int32)

    def body(cache, t):
        prev_token = prev_tokens[:, t]
        cond, cache = model.apply(
            variables,
            prev_token,
            t,
            cache,
            train=False,
            method=model.decode_step,
        )
        return cache, cond

    _, conds_t = jax.lax.scan(body, cache, ts)
    return jnp.swapaxes(conds_t, 0, 1)  # (B,N,local_size)
