from __future__ import annotations
import math
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
import netket as nk

PATCH_SIZE = 2
PATCH_AREA = PATCH_SIZE * PATCH_SIZE
PATCH_VOCAB_SIZE = 2 ** PATCH_AREA


def _require_patch_size(patch_size: int) -> int:
    if int(patch_size) != PATCH_SIZE:
        raise ValueError("Only 2x2 patching is supported in this simplified module.")
    return PATCH_SIZE


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


def magnetization_to_targets(
    num_sites: int,
    target_magnetization: int = 0,
) -> tuple[int, int]:
    if (num_sites + target_magnetization) % 2 != 0:
        raise ValueError(
            f"num_sites={num_sites} and target_magnetization={target_magnetization} "
            "have incompatible parity."
        )
    target_up = (num_sites + target_magnetization) // 2
    target_down = num_sites - target_up
    if target_up < 0 or target_down < 0:
        raise ValueError(
            f"Invalid target_magnetization={target_magnetization} for N={num_sites}."
        )
    return int(target_down), int(target_up)


def mask_log_probs_from_feasible_actions(
    log_probs: jax.Array,
    feasible: jax.Array,
) -> jax.Array:
    neg_inf = jnp.asarray(-jnp.inf, dtype=log_probs.dtype)
    masked = jnp.where(feasible, log_probs, neg_inf)
    return jax.nn.log_softmax(masked, axis=-1)


def apply_feasible_mask_to_cond_logpsi(
    cond_logpsi: jax.Array,
    feasible: jax.Array,
    *,
    machine_pow: int,
) -> jax.Array:
    cat_log_probs = float(machine_pow) * jnp.real(cond_logpsi)
    masked_cat_log_probs = mask_log_probs_from_feasible_actions(cat_log_probs, feasible)
    return masked_cat_log_probs / float(machine_pow) + 1j * jnp.imag(cond_logpsi)


def patch_vocab_size(patch_size: int = PATCH_SIZE) -> int:
    patch_size = _require_patch_size(patch_size)
    return 2 ** (patch_size * patch_size)


def patch_token_table(patch_size: int = PATCH_SIZE) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    patch_area = patch_size * patch_size
    vocab = patch_vocab_size(patch_size)
    bits = (
        (jnp.arange(vocab, dtype=jnp.int32)[:, None]
         >> jnp.arange(patch_area, dtype=jnp.int32)[None, :])
        & 1
    )
    return bits.astype(jnp.int32)


def patch_up_count_table(patch_size: int = PATCH_SIZE) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    return jnp.sum(patch_token_table(patch_size), axis=-1).astype(jnp.int32)


def num_patch_tokens(lattice_shape: Tuple[int, int], patch_size: int = PATCH_SIZE) -> int:
    patch_size = _require_patch_size(patch_size)
    n_rows, n_cols = lattice_shape
    if n_rows % patch_size != 0 or n_cols % patch_size != 0:
        raise ValueError(
            f"lattice_shape={lattice_shape} must be divisible by patch_size={patch_size}."
        )
    return (n_rows // patch_size) * (n_cols // patch_size)


def patch_lattice_shape(lattice_shape: Tuple[int, int], patch_size: int = PATCH_SIZE) -> Tuple[int, int]:
    patch_size = _require_patch_size(patch_size)
    n_rows, n_cols = lattice_shape
    if n_rows % patch_size != 0 or n_cols % patch_size != 0:
        raise ValueError(
            f"lattice_shape={lattice_shape} must be divisible by patch_size={patch_size}."
        )
    return (n_rows // patch_size, n_cols // patch_size)


def configs_to_patch_tokens(
    configs: jax.Array,
    *,
    lattice_shape: Optional[Tuple[int, int]] = None,
    L: Optional[int] = None,
    patch_size: int = 2,
) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    if configs.ndim == 1:
        configs = configs[None, :]
    configs = configs.astype(jnp.int32)

    batch_size, n_sites = configs.shape
    del batch_size
    if lattice_shape is None:
        if L is None:
            L_float = math.sqrt(n_sites)
            L_int = int(round(L_float))
            if L_int * L_int != n_sites:
                raise ValueError(
                    "configs_to_patch_tokens needs lattice_shape or L for non-square inputs."
                )
            lattice_shape = (L_int, L_int)
        else:
            lattice_shape = (int(L), int(L))

    n_rows, n_cols = lattice_shape
    if n_rows * n_cols != n_sites:
        raise ValueError(
            f"lattice_shape={lattice_shape} does not match configs with N={n_sites}."
        )
    if n_rows % patch_size != 0 or n_cols % patch_size != 0:
        raise ValueError(
            f"lattice_shape={lattice_shape} must be divisible by patch_size={patch_size}."
        )

    bits = (configs > 0).astype(jnp.int32).reshape((-1, n_rows, n_cols))
    p = patch_size
    pr = n_rows // p
    pc = n_cols // p
    patches = bits.reshape((-1, pr, p, pc, p)).transpose(0, 1, 3, 2, 4)
    patches = patches.reshape((-1, pr * pc, p * p))
    powers = (1 << jnp.arange(p * p, dtype=jnp.int32))
    return jnp.sum(patches * powers[None, None, :], axis=-1).astype(jnp.int32)


def patch_tokens_to_configs(
    tokens: jax.Array,
    *,
    lattice_shape: Optional[Tuple[int, int]] = None,
    L: Optional[int] = None,
    patch_size: int = 2,
) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    if tokens.ndim == 1:
        tokens = tokens[None, :]
    tokens = tokens.astype(jnp.int32)

    batch_size, n_tokens = tokens.shape
    del batch_size
    if lattice_shape is None:
        if L is None:
            Lp_float = math.sqrt(n_tokens)
            Lp = int(round(Lp_float))
            if Lp * Lp != n_tokens:
                raise ValueError(
                    "patch_tokens_to_configs needs lattice_shape or L for non-square patch grids."
                )
            lattice_shape = (Lp * patch_size, Lp * patch_size)
        else:
            lattice_shape = (int(L), int(L))

    n_rows, n_cols = lattice_shape
    p = patch_size
    if n_rows % p != 0 or n_cols % p != 0:
        raise ValueError(
            f"lattice_shape={lattice_shape} must be divisible by patch_size={patch_size}."
        )
    pr = n_rows // p
    pc = n_cols // p
    if pr * pc != n_tokens:
        raise ValueError(
            f"patch token count {n_tokens} does not match patch lattice {(pr, pc)}."
        )

    table = patch_token_table(p)  # (vocab, p*p)
    patches = table[tokens]      # (B, T, p*p)
    patches = patches.reshape((-1, pr, pc, p, p))
    bits = patches.transpose(0, 1, 3, 2, 4).reshape((-1, n_rows, n_cols))
    return (2 * bits - 1).astype(jnp.int32).reshape((-1, n_rows * n_cols))


def feasible_patch_mask_from_prefix_up_count(
    n_up: jax.Array,
    t_patch: jax.Array,
    *,
    num_sites: int,
    num_patches: int,
    patch_size: int = 2,
    target_magnetization: int = 0,
) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    _, target_up = magnetization_to_targets(num_sites, target_magnetization)
    n_up = n_up.astype(jnp.int32)
    t_patch = t_patch.astype(jnp.int32)

    up_counts = patch_up_count_table(patch_size)
    up_counts = up_counts.reshape((1,) * n_up.ndim + (up_counts.shape[0],))
    n_up_after = n_up[..., None] + up_counts

    remaining_patches = (num_patches - (t_patch + 1))[..., None]
    patch_area = patch_size * patch_size
    max_future_up = remaining_patches * patch_area

    can_not_exceed = n_up_after <= jnp.asarray(target_up, dtype=jnp.int32)
    can_still_reach = n_up_after + max_future_up >= jnp.asarray(target_up, dtype=jnp.int32)
    return can_not_exceed & can_still_reach


def feasible_patch_mask_from_tokens(
    patch_tokens: jax.Array,
    *,
    num_sites: int,
    patch_size: int = 2,
    target_magnetization: int = 0,
) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    batch_size, n_tokens = patch_tokens.shape
    del batch_size
    up_counts = patch_up_count_table(patch_size)[patch_tokens]
    n_up_before = jnp.cumsum(up_counts, axis=1) - up_counts
    t_patch = jnp.arange(n_tokens, dtype=jnp.int32)[None, :]
    return feasible_patch_mask_from_prefix_up_count(
        n_up_before,
        t_patch,
        num_sites=num_sites,
        num_patches=n_tokens,
        patch_size=patch_size,
        target_magnetization=target_magnetization,
    )


def prefix_count_features_from_site_prefix(
    n_up_before: jax.Array,
    t_sites: jax.Array,
    *,
    num_sites: int,
    target_magnetization: int = 0,
) -> jax.Array:
    target_down, target_up = magnetization_to_targets(
        num_sites,
        target_magnetization,
    )
    dtype = jnp.float32
    n_up = n_up_before.astype(dtype)
    t = t_sites.astype(dtype)
    n_down = t - n_up

    denom = jnp.asarray(max(num_sites, 1), dtype=dtype)
    target_up_f = jnp.asarray(target_up, dtype=dtype)
    target_down_f = jnp.asarray(target_down, dtype=dtype)
    target_mag_f = jnp.asarray(target_magnetization, dtype=dtype)

    sites_to_go = jnp.asarray(num_sites, dtype=dtype) - t
    up_to_go = target_up_f - n_up
    down_to_go = target_down_f - n_down
    prefix_magnetization = n_up - n_down

    zeros = jnp.zeros_like(n_up + t)
    features = jnp.stack(
        [
            n_up / denom,
            n_down / denom,
            (sites_to_go + zeros) / denom,
            up_to_go / denom,
            down_to_go / denom,
            prefix_magnetization / denom,
            (target_mag_f + zeros) / denom,
        ],
        axis=-1,
    )
    return features


def prefix_count_features_from_patch_tokens(
    patch_tokens: jax.Array,
    *,
    num_sites: int,
    patch_size: int = 2,
    target_magnetization: int = 0,
) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    batch_size, n_tokens = patch_tokens.shape
    del batch_size
    up_counts = patch_up_count_table(patch_size)[patch_tokens]
    n_up_before = jnp.cumsum(up_counts, axis=1) - up_counts
    patch_area = int(patch_size * patch_size)
    t_sites = patch_area * jnp.arange(n_tokens, dtype=jnp.int32)[None, :]
    return prefix_count_features_from_site_prefix(
        n_up_before,
        t_sites,
        num_sites=num_sites,
        target_magnetization=target_magnetization,
    )


def prefix_count_features_from_patch_prefix(
    n_up_before: jax.Array,
    t_patch: jax.Array,
    *,
    num_sites: int,
    patch_size: int = 2,
    target_magnetization: int = 0,
) -> jax.Array:
    patch_size = _require_patch_size(patch_size)
    patch_area = int(patch_size * patch_size)
    t_sites = t_patch.astype(jnp.int32) * patch_area
    return prefix_count_features_from_site_prefix(
        n_up_before,
        t_sites,
        num_sites=num_sites,
        target_magnetization=target_magnetization,
    )


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

        k = k.astype(k_cache.dtype)
        v = v.astype(v_cache.dtype)
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


class ComplexPatchRoPETransformer2DAR(nn.Module):

    hilbert: object
    lattice_shape: Tuple[int, int]
    embed_dim: int = 64
    width: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_dim: Optional[int] = None
    head_hidden: int = 256

    use_site_embedding: bool = True
    rope_base: float = 100.0
    dropout: float = 0.0
    residual_scale: float = 1.0
    use_qkv_bias: bool = True
    use_input_tanh: bool = False

    machine_pow: int = 2
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    enforce_zero_magnetization: bool = True
    target_magnetization: int = 0

    use_prefix_features: bool = True
    prefix_feature_dim: int = 16

    @property
    def is_patch_ar(self) -> bool:
        return True

    @property
    def patch_vocab(self) -> int:
        return patch_vocab_size(PATCH_SIZE)

    @property
    def num_tokens(self) -> int:
        return num_patch_tokens(self.lattice_shape, PATCH_SIZE)

    @nn.compact
    def conditionals_log_psi(
        self,
        inputs: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        batch_size, num_sites = inputs.shape
        expected_sites = self.lattice_shape[0] * self.lattice_shape[1]
        if num_sites != expected_sites:
            raise ValueError(f"Expected N={expected_sites}, got N={num_sites}.")

        local_size = patch_vocab_size(PATCH_SIZE)
        patch_tokens = configs_to_patch_tokens(
            inputs,
            lattice_shape=self.lattice_shape,
            patch_size=PATCH_SIZE,
        )
        num_tokens_ = patch_tokens.shape[1]

        bos_token = jnp.full((batch_size, 1), local_size, dtype=jnp.int32)
        prev_tokens = jnp.concatenate([bos_token, patch_tokens[:, :-1]], axis=1)

        prefix_features = None
        if self.use_prefix_features:
            prefix_features = prefix_count_features_from_patch_tokens(
                patch_tokens,
                num_sites=num_sites,
                patch_size=PATCH_SIZE,
                target_magnetization=self.target_magnetization,
            )

        x = self._embed_tokens(
            prev_tokens,
            num_tokens=num_tokens_,
            vocab_size=local_size + 1,
            prefix_features=prefix_features,
            train=train,
        )

        p_lattice_shape = patch_lattice_shape(self.lattice_shape, PATCH_SIZE)
        rows, cols = lattice_coordinates_2d(num_tokens_, p_lattice_shape)
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

        feasible = None
        if self.enforce_zero_magnetization:
            feasible = feasible_patch_mask_from_tokens(
                patch_tokens,
                num_sites=num_sites,
                patch_size=PATCH_SIZE,
                target_magnetization=self.target_magnetization,
            )
        return self._heads(x, local_size=local_size, feasible_mask=feasible)

    @nn.compact
    def decode_step(
        self,
        prev_token: jax.Array,
        t: jax.Array,
        cache: dict[str, jax.Array],
        n_up_before: Optional[jax.Array] = None,
        train: bool = False,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if prev_token.ndim != 1:
            raise ValueError(f"prev_token must have shape (B,), got {prev_token.shape}.")

        local_size = patch_vocab_size(PATCH_SIZE)
        num_tokens_ = num_patch_tokens(self.lattice_shape, PATCH_SIZE)

        num_sites = self.lattice_shape[0] * self.lattice_shape[1]
        prefix_features = None
        if self.use_prefix_features and n_up_before is not None:
            prefix_features = prefix_count_features_from_patch_prefix(
                n_up_before,
                t,
                num_sites=num_sites,
                patch_size=PATCH_SIZE,
                target_magnetization=self.target_magnetization,
            )[:, None, :]

        x = self._embed_decode_token(
            prev_token,
            t=t,
            num_tokens=num_tokens_,
            vocab_size=local_size + 1,
            prefix_features=prefix_features,
            train=train,
        )

        p_lattice_shape = patch_lattice_shape(self.lattice_shape, PATCH_SIZE)
        rows, cols = lattice_coordinates_2d(num_tokens_, p_lattice_shape)
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
        num_tokens: int,
        vocab_size: int,
        prefix_features: Optional[jax.Array] = None,
        train: bool,
    ) -> jax.Array:
        _, seq_len = tokens.shape
        x = nn.Embed(
            num_embeddings=vocab_size,
            features=self.embed_dim,
            name="tok_emb",
        )(tokens)

        if self.use_site_embedding:
            site_ids = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
            x = x + nn.Embed(
                num_embeddings=num_tokens,
                features=self.embed_dim,
                name="site_emb",
            )(site_ids)

        if self.use_prefix_features and prefix_features is not None:
            x = x + self._prefix_features_to_embedding(prefix_features, dtype=x.dtype)

        x = nn.Dense(self.width, name="input_proj")(x)
        if self.use_input_tanh:
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
        num_tokens: int,
        vocab_size: int,
        prefix_features: Optional[jax.Array] = None,
        train: bool,
    ) -> jax.Array:
        x = nn.Embed(
            num_embeddings=vocab_size,
            features=self.embed_dim,
            name="tok_emb",
        )(token[:, None])

        if self.use_site_embedding:
            site_id = jnp.broadcast_to(t.astype(jnp.int32), (1, 1))
            x = x + nn.Embed(
                num_embeddings=num_tokens,
                features=self.embed_dim,
                name="site_emb",
            )(site_id)

        if self.use_prefix_features and prefix_features is not None:
            x = x + self._prefix_features_to_embedding(prefix_features, dtype=x.dtype)

        x = nn.Dense(self.width, name="input_proj")(x)
        if self.use_input_tanh:
            x = nn.tanh(x)
        x = nn.LayerNorm(name="input_ln")(x)
        x = nn.Dropout(rate=self.dropout, name="input_dropout")(
            x,
            deterministic=not train,
        )
        return x

    def _prefix_features_to_embedding(
        self,
        prefix_features: jax.Array,
        *,
        dtype,
    ) -> jax.Array:
        h = prefix_features.astype(jnp.float32)
        h = nn.Dense(self.prefix_feature_dim, name="prefix_stats_fc1")(h)
        h = nn.tanh(h)
        h = nn.Dense(self.embed_dim, name="prefix_stats_fc2")(h)
        return h.astype(dtype)

    def _heads(
        self,
        x: jax.Array,
        *,
        local_size: int,
        feasible_mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        x_amp = nn.Dense(self.head_hidden, name="amp_fc1")(x)
        x_amp = nn.gelu(x_amp)
        x_amp = nn.Dense(self.head_hidden, name="amp_fc2")(x_amp)
        x_amp = nn.gelu(x_amp)
        amp_logits = nn.Dense(local_size, name="amp_head")(x_amp)
        if feasible_mask is not None:
            neg_inf = jnp.asarray(-jnp.inf, dtype=amp_logits.dtype)
            amp_logits = jnp.where(feasible_mask, amp_logits, neg_inf)
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


def make_kv_cache(
    model: ComplexPatchRoPETransformer2DAR,
    *,
    batch_size: int,
    num_tokens: Optional[int] = None,
    dtype=jnp.float32,
) -> dict[str, jax.Array]:
    if num_tokens is None:
        num_tokens = num_patch_tokens(model.lattice_shape, PATCH_SIZE)
    if model.width % model.num_heads != 0:
        raise ValueError(f"width={model.width} must be divisible by num_heads={model.num_heads}.")
    head_dim = model.width // model.num_heads
    if head_dim % 4 != 0:
        raise ValueError(f"2D axial RoPE requires head_dim divisible by 4, got {head_dim}.")
    shape = (model.depth, batch_size, num_tokens, model.num_heads, head_dim)
    return {"k": jnp.zeros(shape, dtype=dtype), "v": jnp.zeros(shape, dtype=dtype)}


def sample_cached_apply(
    model: ComplexPatchRoPETransformer2DAR,
    variables: dict,
    key: jax.Array,
    *,
    batch_size: int,
    cache_dtype=jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    num_sites = model.lattice_shape[0] * model.lattice_shape[1]
    num_tokens_ = num_patch_tokens(model.lattice_shape, PATCH_SIZE)
    local_size = patch_vocab_size(PATCH_SIZE)
    cache = make_kv_cache(model, batch_size=batch_size, num_tokens=num_tokens_, dtype=cache_dtype)

    patch_tokens0 = jnp.zeros((batch_size, num_tokens_), dtype=jnp.int32)
    logpsi0 = jnp.zeros((batch_size,), dtype=jnp.complex64)
    n_up0 = jnp.zeros((batch_size,), dtype=jnp.int32)
    prev_token0 = jnp.full((batch_size,), local_size, dtype=jnp.int32)
    keys = jax.random.split(key, num_tokens_)
    ts = jnp.arange(num_tokens_, dtype=jnp.int32)
    up_count_table = patch_up_count_table(PATCH_SIZE)

    def body(carry, xs):
        cache, prev_token, patch_tokens, logpsi, n_up = carry
        t, key_t = xs
        cond_logpsi, cache = model.apply(
            variables,
            prev_token,
            t,
            cache,
            n_up_before=n_up,
            train=False,
            method=model.decode_step,
        )
        cat_log_probs = float(model.machine_pow) * jnp.real(cond_logpsi)
        if model.enforce_zero_magnetization:
            feasible = feasible_patch_mask_from_prefix_up_count(
                n_up,
                t,
                num_sites=num_sites,
                num_patches=num_tokens_,
                patch_size=PATCH_SIZE,
                target_magnetization=model.target_magnetization,
            )
            cat_log_probs = mask_log_probs_from_feasible_actions(cat_log_probs, feasible)

        token = jax.random.categorical(key_t, cat_log_probs, axis=-1).astype(jnp.int32)
        patch_tokens = patch_tokens.at[:, t].set(token)
        batch_idx = jnp.arange(batch_size)
        chosen_amp = cat_log_probs[batch_idx, token] / float(model.machine_pow)
        chosen_phase = jnp.imag(cond_logpsi)[batch_idx, token]
        logpsi = logpsi + (chosen_amp + 1j * chosen_phase)
        n_up = n_up + up_count_table[token]
        return (cache, token, patch_tokens, logpsi, n_up), None

    (_, _, patch_tokens, logpsi, _), _ = jax.lax.scan(
        body,
        (cache, prev_token0, patch_tokens0, logpsi0, n_up0),
        (ts, keys),
    )
    samples = patch_tokens_to_configs(
        patch_tokens,
        lattice_shape=model.lattice_shape,
        patch_size=PATCH_SIZE,
    )
    return samples.astype(jnp.int8), logpsi


def cached_conditionals_for_inputs(
    model: ComplexPatchRoPETransformer2DAR,
    variables: dict,
    inputs: jax.Array,
    *,
    cache_dtype=jnp.float32,
) -> jax.Array:
    if inputs.ndim == 1:
        inputs = inputs[None, :]
    batch_size, num_sites = inputs.shape
    num_tokens_ = num_patch_tokens(model.lattice_shape, PATCH_SIZE)
    local_size = patch_vocab_size(PATCH_SIZE)
    cache = make_kv_cache(model, batch_size=batch_size, num_tokens=num_tokens_, dtype=cache_dtype)

    patch_tokens = configs_to_patch_tokens(
        inputs,
        lattice_shape=model.lattice_shape,
        patch_size=PATCH_SIZE,
    )
    bos = jnp.full((batch_size,), local_size, dtype=jnp.int32)
    prev_tokens = jnp.concatenate([bos[:, None], patch_tokens[:, :-1]], axis=1)
    up_counts = patch_up_count_table(PATCH_SIZE)[patch_tokens]
    n_up_before = jnp.cumsum(up_counts, axis=1) - up_counts
    ts = jnp.arange(num_tokens_, dtype=jnp.int32)

    def body(cache, t):
        cond, cache = model.apply(
            variables,
            prev_tokens[:, t],
            t,
            cache,
            n_up_before=n_up_before[:, t],
            train=False,
            method=model.decode_step,
        )
        if model.enforce_zero_magnetization:
            feasible = feasible_patch_mask_from_prefix_up_count(
                n_up_before[:, t],
                t,
                num_sites=num_sites,
                num_patches=num_tokens_,
                patch_size=PATCH_SIZE,
                target_magnetization=model.target_magnetization,
            )
            cond = apply_feasible_mask_to_cond_logpsi(
                cond,
                feasible,
                machine_pow=model.machine_pow,
            )
        return cache, cond

    _, conds_t = jax.lax.scan(body, cache, ts)
    return jnp.swapaxes(conds_t, 0, 1)
