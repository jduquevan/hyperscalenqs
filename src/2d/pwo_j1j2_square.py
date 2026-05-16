from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import netket as nk
import optax
import orbax.checkpoint as ocp
from flax.training import train_state

from .vit_rope import (
    ComplexPatchRoPETransformer2DAR,
    ComplexRoPETransformer2DAR,
    configs_to_patch_tokens,
    magnetization_to_targets,
    sample_cached_apply,
)


@dataclass
class Args:
    seed: int = 42
    jax_platform_name: str = "gpu"

    L: int = 10
    pbc: bool = True
    J1: float = 1.0
    J2: float = 0.5
    total_sz: float = 0.0
    enforce_zero_magnetization: bool = True

    n_samples: int = 1024
    machine_pow: int = 2

    use_patch_ar: bool = True
    patch_size: int = 2
    use_input_tanh: bool = False

    use_prefix_features: bool = True
    prefix_feature_dim: int = 16

    width: int = 96
    depth: int = 8
    num_heads: int = 6
    embed_dim: int = 64
    mlp_dim: Optional[int] = None
    head_hidden: int = 192
    use_site_embedding: bool = True
    # 100 is usually a better starting point than the LLM default 10_000 for a 5x5
    # patch lattice / 10x10 spin lattice.
    rope_base: float = 100.0
    dropout: float = 0.0
    residual_scale: float = 1.0
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    # Learning-rate schedule.
    #
    # Supported values:
    #   - "warmup_const_decay" (default): warm up, hold, then drop once.
    #   - "constant": fixed LR = peak_lr.
    #   - "cosine_onecycle" / "onecycle" / "cosine": original Optax one-cycle schedule.
    #   - "cosine_decay": warmup followed by cosine decay from peak_lr to end_lr.
    lr_schedule_type: str = "constant"
    peak_lr: float = 0.0001
    warmup_outer_steps: int = 25
    decay_start_outer_steps: int = 300
    decay_factor: float = 0.3
    end_lr: float = 1e-6

    pct_start: float = 0.15
    div_factor: float = 10.0
    final_div_factor: float = 200.0
    transition_steps: int = 10_000
    n_iter: int = 1_000_000
    optimizer: str = "adam"
    sgd_momentum: float = 0.0

    ppo_epochs: int = 4
    ppo_clip_eps: float = 1e-3
    normalize_advantage: bool = True

    phase_loss_type: str = "delta_clip"
    phase_coef: float = 1.0
    phase_delta_clip: float = 0.3
    phase_ratio_tau: float = 1e-3
    phase_clip_eps: float = 1e-2
    center_imag_advantage: bool = True
    normalize_imag_advantage: bool = True
    phase_delta_l2_coef: float = 0.0

    use_phase_jacobian_baseline: bool = False
    phase_jacobian_baseline_eps: float = 1e-8

    eval_every: int = 300
    eval_n_samples: int = 32768
    # eval_n_samples: int = 262_144
    eval_batch_size: int = 1024
    reference_energy: Optional[float] = None

    local_energy_conn_chunk_size: int = 16
    debug_local_energy_progress: bool = False
    debug_local_energy_print_every_chunks: int = 1
    kv_cache_dtype: str = "float32"
    validate_cached_sampler: bool = False

    log_every: int = 1
    log_gradient_info: bool = False
    time_optimization_steps: bool = False

    save_checkpoint_every: int = 0
    output_directory: Optional[str] = "."

    wandb_project: str = "hyperscalenqs-j1j2-square"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_directory: Optional[str] = "."
    wandb_tags: Optional[str] = "pwo,j1j2_square,patch_ar_2x2,rope_2d,kv_cache,zero_magnetization,prefix_stats"


ConfigStore.instance().store(name="config", node=Args)


class TrainState(train_state.TrainState):
    rng: jax.Array


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scale(tree, scalar):
    return jax.tree_util.tree_map(lambda x: scalar * x, tree)


def tree_l2_sq(tree):
    return sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree))


def cache_dtype_from_string(name: str):
    name = name.lower()
    if name == "float32":
        return jnp.float32
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float16":
        return jnp.float16
    raise ValueError(f"Unknown kv_cache_dtype={name}")


def make_warmup_const_decay_schedule(
    *,
    peak_lr: float,
    warmup_steps: int,
    decay_start_steps: int,
    decay_factor: float,
):
    warmup_steps = max(int(warmup_steps), 0)
    decay_start_steps = max(int(decay_start_steps), warmup_steps)
    decay_factor = float(decay_factor)

    def schedule(step):
        step = jnp.asarray(step)
        peak = jnp.asarray(peak_lr, dtype=jnp.float32)
        if warmup_steps > 0:
            warmup = peak * (step + 1) / float(warmup_steps)
            lr = jnp.where(step < warmup_steps, warmup, peak)
        else:
            lr = peak
        decayed = peak * decay_factor
        lr = jnp.where(step >= decay_start_steps, decayed, lr)
        return lr

    return schedule


def make_tx(cfg: Args):
    schedule_type = cfg.lr_schedule_type.lower()

    warmup_steps = int(cfg.warmup_outer_steps) * int(cfg.ppo_epochs)
    decay_start_steps = int(cfg.decay_start_outer_steps) * int(cfg.ppo_epochs)

    if schedule_type == "warmup_const_decay":
        lr_schedule = make_warmup_const_decay_schedule(
            peak_lr=cfg.peak_lr,
            warmup_steps=warmup_steps,
            decay_start_steps=decay_start_steps,
            decay_factor=cfg.decay_factor,
        )

    elif schedule_type == "constant":
        lr_schedule = optax.constant_schedule(cfg.peak_lr)

    elif schedule_type in {"cosine", "cosine_onecycle", "onecycle"}:
        lr_schedule = optax.schedules.cosine_onecycle_schedule(
            transition_steps=cfg.transition_steps,
            peak_value=cfg.peak_lr,
            pct_start=cfg.pct_start,
            div_factor=cfg.div_factor,
            final_div_factor=cfg.final_div_factor,
        )

    elif schedule_type == "cosine_decay":
        init_value = cfg.peak_lr / cfg.div_factor
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=cfg.peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=cfg.transition_steps,
            end_value=cfg.end_lr,
        )

    else:
        raise ValueError(
            f"Unknown lr_schedule_type={cfg.lr_schedule_type}. "
            "Choose one of: warmup_const_decay, constant, cosine_onecycle, cosine_decay."
        )

    opt = cfg.optimizer.lower()
    if opt == "adam":
        tx = optax.adam(lr_schedule)
    elif opt == "sgd":
        tx = optax.sgd(lr_schedule, momentum=cfg.sgd_momentum)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return tx, lr_schedule


def target_magnetization_from_total_sz(total_sz: float) -> int:
    """Converts NetKet total_sz=sum_i S_i^z to M=sum_i s_i for s_i in {-1,+1}."""
    target = 2.0 * float(total_sz)
    rounded = int(round(target))
    if abs(target - rounded) > 1e-6:
        raise ValueError(
            f"total_sz={total_sz} gives non-integer target magnetization {target}."
        )
    return rounded


def log_num_fixed_magnetization_configs(n_sites: int, target_magnetization: int) -> float:
    """Returns log of the number of spin configurations with sum_i s_i = target_magnetization."""
    if abs(target_magnetization) > n_sites:
        raise ValueError(
            f"target_magnetization={target_magnetization} is outside [-N, N] for N={n_sites}."
        )
    if (n_sites + target_magnetization) % 2 != 0:
        raise ValueError(
            f"target_magnetization={target_magnetization} has incompatible parity for N={n_sites}."
        )
    target_up = (n_sites + target_magnetization) // 2
    target_down = n_sites - target_up
    return (
        math.lgamma(n_sites + 1)
        - math.lgamma(target_up + 1)
        - math.lgamma(target_down + 1)
    )


def max_entropy_for_hilbert(n_sites: int, target_magnetization: int, enforce_fixed_magnetization: bool) -> float:
    """Maximum entropy of the sampling support in nats."""
    if enforce_fixed_magnetization:
        return log_num_fixed_magnetization_configs(n_sites, target_magnetization)
    return n_sites * math.log(2.0)


def make_fixed_magnetization_configs(
    batch_size: int,
    n_sites: int,
    target_magnetization: int = 0,
) -> jax.Array:
    """Creates simple valid spin configurations with fixed M=sum_i s_i."""
    target_down, target_up = magnetization_to_targets(n_sites, target_magnetization)
    one = jnp.concatenate([
        -jnp.ones((target_down,), dtype=jnp.int32),
        jnp.ones((target_up,), dtype=jnp.int32),
    ])
    return jnp.broadcast_to(one[None, :], (batch_size, n_sites))


def build_square_j1j2(cfg: Args):
    n_sites = cfg.L * cfg.L
    if cfg.enforce_zero_magnetization:
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites, total_sz=cfg.total_sz)
    else:
        hi = nk.hilbert.Spin(s=1 / 2, N=n_sites)
    graph = nk.graph.Hypercube(
        length=cfg.L,
        n_dim=2,
        pbc=cfg.pbc,
        max_neighbor_order=2,
    )
    hamiltonian = nk.operator.Heisenberg(
        hilbert=hi,
        graph=graph,
        J=[cfg.J1, cfg.J2],
        sign_rule=[False, False],
    )
    return hi, graph, hamiltonian


def make_model(cfg: Args, hilbert):
    common_kwargs = dict(
        hilbert=hilbert,
        lattice_shape=(cfg.L, cfg.L),
        embed_dim=cfg.embed_dim,
        width=cfg.width,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_dim=cfg.mlp_dim,
        head_hidden=cfg.head_hidden,
        use_site_embedding=cfg.use_site_embedding,
        rope_base=cfg.rope_base,
        dropout=cfg.dropout,
        residual_scale=cfg.residual_scale,
        machine_pow=cfg.machine_pow,
        phase_scale=cfg.phase_scale,
        phase_init_std=cfg.phase_init_std,
        enforce_zero_magnetization=cfg.enforce_zero_magnetization,
        target_magnetization=target_magnetization_from_total_sz(cfg.total_sz),
        use_prefix_features=cfg.use_prefix_features,
        prefix_feature_dim=cfg.prefix_feature_dim,
    )
    if cfg.use_patch_ar:
        return ComplexPatchRoPETransformer2DAR(
            **common_kwargs,
            patch_size=cfg.patch_size,
            use_input_tanh=cfg.use_input_tanh,
        )
    return ComplexRoPETransformer2DAR(**common_kwargs)




def apply_logpsi_explicit(model, params, configs):
    """Evaluate log psi by explicitly gathering the selected AR token log-psi.

    For the original site AR model this gathers one binary token per spin. For
    v2 patch AR, the physical spin configuration is first converted into patch
    tokens and then one categorical token is gathered per patch. In both cases
    the returned value is log psi for the physical spin configuration.
    """
    if configs.ndim == 1:
        configs = configs[None, :]
    configs = configs.astype(jnp.int32)
    cond_logpsi = model.apply(
        {"params": params},
        configs,
        train=False,
        method=model.conditionals_log_psi,
    )

    if bool(getattr(model, "is_patch_ar", False)):
        patch_tokens = configs_to_patch_tokens(
            configs,
            lattice_shape=model.lattice_shape,
            patch_size=model.patch_size,
        )
        selected = jnp.take_along_axis(
            cond_logpsi,
            patch_tokens[..., None],
            axis=-1,
        )[..., 0]
        return jnp.sum(selected, axis=-1)

    spin_tokens = (configs > 0).astype(jnp.int32)
    selected = jnp.take_along_axis(
        cond_logpsi,
        spin_tokens[..., None],
        axis=-1,
    )[..., 0]
    return jnp.sum(selected, axis=-1)

def make_local_energy_fn(
    model,
    H,
    conn_chunk_size: int = 1,
    *,
    debug_progress: bool = False,
    debug_print_every_chunks: int = 1,
):
    """
    Local energy with aggressive NaN diagnostics for fixed-magnetization masked AR models.

    Important details:
      * Padded zero-matrix-element connected states are never scored by the model.
        They are replaced by the original valid configuration before model.apply.
      * Ratio computation is masked before exp, so zero-mel entries cannot create 0 * NaN.
      * Debug prints are intentionally labelled [LE ...] so you can verify this file is running.
    """
    if conn_chunk_size < 1:
        raise ValueError("conn_chunk_size must be >= 1")
    if debug_print_every_chunks < 1:
        raise ValueError("debug_print_every_chunks must be >= 1")

    target_magnetization = int(getattr(model, "target_magnetization", 0))
    enforce_fixed_magnetization = bool(
        getattr(model, "enforce_zero_magnetization", False)
    )

    def local_energy(params, configs):
        configs = configs.astype(jnp.int32)

        x_primes, mels = H.get_conn_padded(configs)
        logpsi_x = apply_logpsi_explicit(model, params, configs)

        compute_dtype = logpsi_x.dtype
        mels = mels.astype(compute_dtype)

        B, K, N = x_primes.shape
        C = conn_chunk_size

        if debug_progress:
            mag_configs = jnp.sum(configs, axis=-1)
            target = jnp.asarray(target_magnetization, dtype=mag_configs.dtype)
            jax.debug.print(
                "[LE input] B={} K={} N={} C={} target_M={} enforce_fixed_M={} "
                "configs_finite={} logpsi_x_finite={} "
                "M_min={} M_max={} mean_abs_Merr={:.6e}",
                B,
                K,
                N,
                C,
                target_magnetization,
                enforce_fixed_magnetization,
                jnp.all(jnp.isfinite(configs)),
                jnp.all(jnp.isfinite(logpsi_x)),
                jnp.min(mag_configs),
                jnp.max(mag_configs),
                jnp.mean(jnp.abs(mag_configs - target)),
                ordered=True,
            )
            jax.debug.print(
                "[LE logpsi_x] finite={} nan_count={} "
                "real_min={:.6e} real_max={:.6e} imag_min={:.6e} imag_max={:.6e}",
                jnp.all(jnp.isfinite(logpsi_x)),
                jnp.sum(jnp.isnan(jnp.real(logpsi_x)))
                + jnp.sum(jnp.isnan(jnp.imag(logpsi_x))),
                jnp.nanmin(jnp.real(logpsi_x)),
                jnp.nanmax(jnp.real(logpsi_x)),
                jnp.nanmin(jnp.imag(logpsi_x)),
                jnp.nanmax(jnp.imag(logpsi_x)),
                ordered=True,
            )

            xprime_mag = jnp.sum(x_primes, axis=-1)
            nonzero_conn = mels != 0
            target = jnp.asarray(target_magnetization, dtype=xprime_mag.dtype)
            jax.debug.print(
                "[LE conn] x_primes_finite={} mels_finite={} nonzero_mels={} "
                "xprime_M_min={} xprime_M_max={} bad_nonzero_target_M={}",
                jnp.all(jnp.isfinite(x_primes)),
                jnp.all(jnp.isfinite(mels)),
                jnp.sum(nonzero_conn),
                jnp.min(xprime_mag),
                jnp.max(xprime_mag),
                jnp.sum(nonzero_conn & (xprime_mag != target)),
                ordered=True,
            )

        pad_k = (-K) % C
        if pad_k > 0:
            x_primes = jnp.pad(
                x_primes,
                ((0, 0), (0, pad_k), (0, 0)),
                mode="constant",
                constant_values=1,
            )
            mels = jnp.pad(
                mels,
                ((0, 0), (0, pad_k)),
                mode="constant",
                constant_values=0,
            )

        K_pad = K + pad_k
        n_chunks = K_pad // C

        x_chunks = x_primes.reshape(B, n_chunks, C, N)
        m_chunks = mels.reshape(B, n_chunks, C)

        x_chunks = jnp.swapaxes(x_chunks, 0, 1)
        m_chunks = jnp.swapaxes(m_chunks, 0, 1)
        chunk_ids = jnp.arange(n_chunks, dtype=jnp.int32)

        if debug_progress:
            jax.debug.print(
                "[local_energy] start: B={} K={} C={} n_chunks={} effective_batch={}",
                B,
                K,
                C,
                n_chunks,
                B * C,
                ordered=True,
            )

        def scan_chunk(e_acc, xs):
            chunk_idx, x_chunk, mel_chunk = xs

            x_flat = x_chunk.reshape(B * C, N)
            mel_flat = mel_chunk.reshape(B * C)
            nonzero_flat = mel_flat != 0

            # Do not evaluate padded zero-mel states under the masked AR model.
            # Some padded states can be impossible under the fixed-M constraint.
            # Replacing them by the original valid config prevents logpsi NaNs from
            # irrelevant zero-mel entries.
            x_ref_flat = jnp.repeat(configs[:, None, :], C, axis=1).reshape(B * C, N)
            x_flat_safe = jnp.where(nonzero_flat[:, None], x_flat, x_ref_flat)

            if debug_progress:
                should_print_pre = (chunk_idx % debug_print_every_chunks == 0) | (
                    chunk_idx == n_chunks - 1
                )

                def do_pre_print(_):
                    mag_flat = jnp.sum(x_flat, axis=-1)
                    mag_safe = jnp.sum(x_flat_safe, axis=-1)
                    target = jnp.asarray(target_magnetization, dtype=mag_flat.dtype)
                    jax.debug.print(
                        "[LE chunk {}/{} pre] nonzero={} zero_mel={} "
                        "raw_M_min={} raw_M_max={} safe_M_min={} safe_M_max={} "
                        "bad_nonzero_target_M={} mel_abs_max={:.6e}",
                        chunk_idx + 1,
                        n_chunks,
                        jnp.sum(nonzero_flat),
                        jnp.sum(~nonzero_flat),
                        jnp.min(mag_flat),
                        jnp.max(mag_flat),
                        jnp.min(mag_safe),
                        jnp.max(mag_safe),
                        jnp.sum(nonzero_flat & (mag_flat != target)),
                        jnp.max(jnp.abs(mel_flat)),
                        ordered=True,
                    )
                    return jnp.int32(0)

                def no_pre_print(_):
                    return jnp.int32(0)

                _ = jax.lax.cond(should_print_pre, do_pre_print, no_pre_print, operand=None)

            logpsi_xp = apply_logpsi_explicit(model, params, x_flat_safe)
            logpsi_xp = logpsi_xp.astype(compute_dtype).reshape(B, C)

            nonzero_mel = mel_chunk != 0
            log_ratio_raw = logpsi_xp - logpsi_x[:, None]
            log_ratio = jnp.where(nonzero_mel, log_ratio_raw, jnp.zeros_like(log_ratio_raw))

            # Clip only the real part. The imaginary part is a phase difference.
            log_ratio = (
                jnp.clip(jnp.real(log_ratio), -30.0, 30.0)
                + 1j * jnp.imag(log_ratio)
            ).astype(compute_dtype)

            ratio = jnp.exp(log_ratio)
            contrib = jnp.where(nonzero_mel, mel_chunk * ratio, jnp.zeros_like(ratio))
            e_acc = e_acc + jnp.sum(contrib, axis=1)

            if debug_progress:
                should_print = (chunk_idx % debug_print_every_chunks == 0) | (
                    chunk_idx == n_chunks - 1
                )

                def do_print(_):
                    jax.debug.print(
                        "[LE chunk {}/{} finite] logpsi_xp={} log_ratio={} ratio={} contrib={} e_acc={} "
                        "nan_logpsi_xp={} nan_ratio={} nan_contrib={} nan_e_acc={}",
                        chunk_idx + 1,
                        n_chunks,
                        jnp.all(jnp.isfinite(logpsi_xp)),
                        jnp.all(jnp.isfinite(log_ratio)),
                        jnp.all(jnp.isfinite(ratio)),
                        jnp.all(jnp.isfinite(contrib)),
                        jnp.all(jnp.isfinite(e_acc)),
                        jnp.sum(jnp.isnan(jnp.real(logpsi_xp)))
                        + jnp.sum(jnp.isnan(jnp.imag(logpsi_xp))),
                        jnp.sum(jnp.isnan(jnp.real(ratio)))
                        + jnp.sum(jnp.isnan(jnp.imag(ratio))),
                        jnp.sum(jnp.isnan(jnp.real(contrib)))
                        + jnp.sum(jnp.isnan(jnp.imag(contrib))),
                        jnp.sum(jnp.isnan(jnp.real(e_acc)))
                        + jnp.sum(jnp.isnan(jnp.imag(e_acc))),
                        ordered=True,
                    )
                    jax.debug.print(
                        "[local_energy] chunk {}/{} done, partial mean Re(E)={:.6f}, mean |E|={:.6f}",
                        chunk_idx + 1,
                        n_chunks,
                        jnp.mean(jnp.real(e_acc)),
                        jnp.mean(jnp.abs(e_acc)),
                        ordered=True,
                    )
                    return jnp.int32(0)

                def no_print(_):
                    return jnp.int32(0)

                _ = jax.lax.cond(should_print, do_print, no_print, operand=None)

            return e_acc, None

        e0 = jnp.zeros(logpsi_x.shape, dtype=compute_dtype)
        e_loc, _ = jax.lax.scan(scan_chunk, e0, (chunk_ids, x_chunks, m_chunks))

        if debug_progress:
            jax.debug.print(
                "[LE done] e_loc_finite={} nan_e_loc={} mean_Re={:.6f} std_Re={:.6f} mean_abs={:.6f}",
                jnp.all(jnp.isfinite(e_loc)),
                jnp.sum(jnp.isnan(jnp.real(e_loc)))
                + jnp.sum(jnp.isnan(jnp.imag(e_loc))),
                jnp.mean(jnp.real(e_loc)),
                jnp.std(jnp.real(e_loc)),
                jnp.mean(jnp.abs(e_loc)),
                ordered=True,
            )

        return e_loc

    return local_energy


@hydra.main(version_base="1.3", config_name="config")
def main(cfg: Args) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(jax.default_backend())

    n_sites = cfg.L * cfg.L
    if cfg.width % cfg.num_heads != 0:
        raise ValueError("width must be divisible by num_heads")
    if (cfg.width // cfg.num_heads) % 4 != 0:
        raise ValueError("head_dim must be divisible by 4 for 2D axial RoPE")
    if cfg.use_patch_ar and (cfg.L % cfg.patch_size != 0):
        raise ValueError("For patch AR, L must be divisible by patch_size")

    cache_dtype = cache_dtype_from_string(cfg.kv_cache_dtype)

    output_dir = Path(cfg.output_directory)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    if cfg.wandb_mode != "disabled":
        Path(cfg.wandb_directory).mkdir(parents=True, exist_ok=True)
        tags = None
        if cfg.wandb_tags is not None:
            tags = [tag.strip() for tag in cfg.wandb_tags.split(",") if tag.strip()]
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=tags,
            config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
            dir=cfg.wandb_directory,
            mode=cfg.wandb_mode,
        )

    hi, graph, h_nk = build_square_j1j2(cfg)
    h_jax = h_nk.to_jax_operator() if hasattr(h_nk, "to_jax_operator") else h_nk

    target_magnetization = target_magnetization_from_total_sz(cfg.total_sz)
    entropy_max = max_entropy_for_hilbert(
        n_sites,
        target_magnetization,
        cfg.enforce_zero_magnetization,
    )
    if cfg.enforce_zero_magnetization:
        dummy_configs = make_fixed_magnetization_configs(
            2,
            n_sites,
            target_magnetization=target_magnetization,
        )
    else:
        dummy_configs = jnp.ones((2, n_sites), dtype=jnp.int32)
    dummy_xp, dummy_mels = h_jax.get_conn_padded(dummy_configs)
    jax.block_until_ready(dummy_xp)
    jax.block_until_ready(dummy_mels)
    print(f"Local-energy padded connectivity K={dummy_xp.shape[1]}")

    model = make_model(cfg, hi)

    init_key = jax.random.PRNGKey(cfg.seed)
    if cfg.enforce_zero_magnetization:
        dummy_input = make_fixed_magnetization_configs(
            1,
            n_sites,
            target_magnetization=target_magnetization,
        )
    else:
        dummy_input = jnp.ones((1, n_sites), dtype=jnp.int32)
    variables0 = model.init(
        init_key,
        dummy_input,
        train=False,
        method=model.conditionals_log_psi,
    )
    params0 = variables0["params"]

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params0))
    ar_tokens = (cfg.L // cfg.patch_size) ** 2 if cfg.use_patch_ar else n_sites
    model_name = "ComplexPatchRoPETransformer2DAR" if cfg.use_patch_ar else "ComplexRoPETransformer2DAR"
    print(
        f"Model: {model_name} cached  L={cfg.L}  N={n_sites}  "
        f"AR_tokens={ar_tokens}  patch_size={cfg.patch_size if cfg.use_patch_ar else 1}  "
        f"J2/J1={cfg.J2 / cfg.J1:.3f}  "
        f"target_M={target_magnetization}  "
        f"zero_mag_mask={cfg.enforce_zero_magnetization}  "
        f"prefix_features={cfg.use_prefix_features}  "
        f"prefix_feature_dim={cfg.prefix_feature_dim}  "
        f"Parameters: {n_params:,}"
    )
    print(
        f"Max support entropy: {entropy_max:.6f} nats  "
        f"({entropy_max / n_sites:.6f} nats/site)"
    )

    if cfg.validate_cached_sampler:
        test_key = jax.random.PRNGKey(cfg.seed + 123)
        test_configs, _ = sample_cached_apply(
            model,
            {"params": params0},
            test_key,
            batch_size=8,
            cache_dtype=cache_dtype,
        )
        conds_full = model.apply(
            {"params": params0},
            test_configs.astype(jnp.int32),
            method=model.conditionals_log_psi,
        )
        from .vit_rope_v2 import cached_conditionals_for_inputs

        conds_cached = cached_conditionals_for_inputs(
            model,
            {"params": params0},
            test_configs.astype(jnp.int32),
            cache_dtype=cache_dtype,
        )
        max_diff = jnp.max(jnp.abs(conds_full - conds_cached))
        print(f"cached/full conditional max diff: {float(max_diff):.6e}")

    tx, lr_schedule = make_tx(cfg)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params0,
        tx=tx,
        rng=jax.random.PRNGKey(cfg.seed + 10_000),
    )

    local_energy = make_local_energy_fn(
        model,
        h_jax,
        conn_chunk_size=cfg.local_energy_conn_chunk_size,
        debug_progress=cfg.debug_local_energy_progress,
        debug_print_every_chunks=cfg.debug_local_energy_print_every_chunks,
    )

    def collect_batch(state):
        old_params = state.params
        rng, sample_key = jax.random.split(state.rng)

        configs, old_logpsi = sample_cached_apply(
            model,
            {"params": old_params},
            sample_key,
            batch_size=cfg.n_samples,
            cache_dtype=cache_dtype,
        )

        configs = jax.lax.stop_gradient(configs.astype(jnp.int32))
        old_logpsi = jax.lax.stop_gradient(old_logpsi)

        old_logp = cfg.machine_pow * jnp.real(old_logpsi)
        old_phase = jnp.imag(old_logpsi)
        e_loc = jax.lax.stop_gradient(local_energy(old_params, configs))

        state = state.replace(rng=rng)
        return state, (configs, old_logp, old_phase, e_loc)

    def loss_terms_on_batch(params, batch):
        configs, old_logp, old_phase, e_loc = batch

        new_logpsi = apply_logpsi_explicit(model, params, configs)
        new_logp = cfg.machine_pow * jnp.real(new_logpsi)
        new_phase = jnp.imag(new_logpsi)

        e_real = jnp.real(e_loc)
        adv_real = e_real - jnp.mean(e_real)
        if cfg.normalize_advantage:
            adv_real = adv_real / (jnp.std(adv_real) + 1e-8)

        log_ratio = jnp.clip(new_logp - old_logp, -20.0, 20.0)
        ratio_real = jnp.exp(log_ratio)
        clipped_ratio_real = jnp.clip(
            ratio_real,
            1.0 - cfg.ppo_clip_eps,
            1.0 + cfg.ppo_clip_eps,
        )
        loss_real = jnp.mean(
            jnp.maximum(
                ratio_real * adv_real,
                clipped_ratio_real * adv_real,
            )
        )

        e_imag = jnp.imag(e_loc)
        magnetization = jnp.sum(configs, axis=-1)
        mag_deviation = magnetization - jnp.asarray(
            target_magnetization,
            dtype=magnetization.dtype,
        )

        # On-policy sample entropy estimate: H[p_theta] = E[-log p_theta(s)].
        # old_logp is the cached sampling log-probability from the frozen batch policy.
        sample_surprisal = -old_logp
        entropy_mean = jnp.mean(sample_surprisal)
        entropy_std = jnp.std(sample_surprisal)
        entropy_max_jnp = jnp.asarray(entropy_max, dtype=entropy_mean.dtype)
        entropy_frac_max = jnp.where(
            entropy_max_jnp > 0.0,
            entropy_mean / entropy_max_jnp,
            jnp.asarray(float("nan"), dtype=entropy_mean.dtype),
        )

        adv_phase = e_imag
        if cfg.center_imag_advantage:
            adv_phase = adv_phase - jnp.mean(adv_phase)
        if cfg.normalize_imag_advantage:
            adv_phase = adv_phase / (jnp.std(adv_phase) + 1e-8)

        phase_delta = jnp.atan2(
            jnp.sin(new_phase - old_phase),
            jnp.cos(new_phase - old_phase),
        )
        is_ratio = jax.lax.stop_gradient(ratio_real)

        if cfg.phase_loss_type == "delta_clip":
            clipped_phase_delta = jnp.clip(
                phase_delta,
                -cfg.phase_delta_clip,
                cfg.phase_delta_clip,
            )
            loss_phase = jnp.mean(
                jnp.maximum(
                    is_ratio * phase_delta * adv_phase,
                    is_ratio * clipped_phase_delta * adv_phase,
                )
            )
            if cfg.phase_delta_l2_coef > 0.0:
                loss_phase = loss_phase + cfg.phase_delta_l2_coef * jnp.mean(
                    phase_delta**2
                )
            phase_stat = phase_delta

        elif cfg.phase_loss_type == "ratio":
            denom = jnp.where(
                jnp.abs(old_phase) >= cfg.phase_ratio_tau,
                old_phase,
                jnp.where(old_phase >= 0.0, cfg.phase_ratio_tau, -cfg.phase_ratio_tau),
            )
            ratio_phase = new_phase / denom
            clipped_ratio_phase = jnp.clip(
                ratio_phase,
                1.0 - cfg.phase_clip_eps,
                1.0 + cfg.phase_clip_eps,
            )
            loss_phase = jnp.mean(
                jnp.maximum(
                    is_ratio * ratio_phase * adv_phase,
                    is_ratio * clipped_ratio_phase * adv_phase,
                )
            )
            phase_stat = ratio_phase

        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")

        total_loss = loss_real + cfg.phase_coef * loss_phase
        metrics = {
            "loss": total_loss,
            "loss_real": loss_real,
            "loss_phase": loss_phase,
            "E_mean_real": jnp.mean(e_real),
            "E_std_real": jnp.std(e_real),
            "E_mean_imag": jnp.mean(e_imag),
            "E_std_imag": jnp.std(e_imag),
            "entropy": entropy_mean,
            "entropy_std": entropy_std,
            "entropy_per_site": entropy_mean / n_sites,
            "entropy_frac_max": entropy_frac_max,
            "entropy_max": entropy_max_jnp,
            "ratio_mean": jnp.mean(ratio_real),
            "clip_frac": jnp.mean(jnp.abs(ratio_real - 1.0) > cfg.ppo_clip_eps),
            "approx_kl": jnp.mean(old_logp - new_logp),
            "phase_old_mean": jnp.mean(old_phase),
            "phase_new_mean": jnp.mean(new_phase),
            "phase_delta_mean": jnp.mean(phase_delta),
            "phase_delta_rms": jnp.sqrt(jnp.mean(phase_delta**2)),
            "phase_stat_mean": jnp.mean(phase_stat),
            "phase_stat_abs_mean": jnp.mean(jnp.abs(phase_stat)),
            "magnetization_mean": jnp.mean(magnetization.astype(jnp.float32)),
            "magnetization_abs_mean": jnp.mean(jnp.abs(magnetization).astype(jnp.float32)),
            "magnetization_deviation_abs_mean": jnp.mean(jnp.abs(mag_deviation).astype(jnp.float32)),
            "magnetization_deviation_max_abs": jnp.max(jnp.abs(mag_deviation).astype(jnp.float32)),
            "magnetization_target_frac": jnp.mean(mag_deviation == 0),
        }
        if cfg.reference_energy is not None:
            reference = jnp.asarray(cfg.reference_energy, dtype=e_real.dtype)
            metrics["reference_energy"] = reference
            metrics["rel_error_reference"] = jnp.abs(
                (metrics["E_mean_real"] - reference) / reference
            )
        return {"real": loss_real, "phase": loss_phase}, metrics

    def phase_backward_weights(params, batch):
        configs, old_logp, old_phase, e_loc = batch
        new_logpsi = apply_logpsi_explicit(model, params, configs)
        new_logp = cfg.machine_pow * jnp.real(new_logpsi)
        new_phase = jnp.imag(new_logpsi)

        log_ratio = jnp.clip(new_logp - old_logp, -20.0, 20.0)
        is_ratio = jax.lax.stop_gradient(jnp.exp(log_ratio))

        e_imag = jnp.imag(e_loc)
        adv_phase = e_imag
        if cfg.center_imag_advantage:
            adv_phase = adv_phase - jnp.mean(adv_phase)
        if cfg.normalize_imag_advantage:
            adv_phase = adv_phase / (jnp.std(adv_phase) + 1e-8)

        batch_size = adv_phase.shape[0]

        if cfg.phase_loss_type == "delta_clip":
            phase_delta = jnp.atan2(
                jnp.sin(new_phase - old_phase),
                jnp.cos(new_phase - old_phase),
            )
            clipped_phase_delta = jnp.clip(
                phase_delta,
                -cfg.phase_delta_clip,
                cfg.phase_delta_clip,
            )
            active = (phase_delta * adv_phase >= clipped_phase_delta * adv_phase).astype(
                new_phase.dtype
            )
            w = is_ratio * active * adv_phase / batch_size
            if cfg.phase_delta_l2_coef > 0.0:
                w = w + (2.0 * cfg.phase_delta_l2_coef / batch_size) * phase_delta

        elif cfg.phase_loss_type == "ratio":
            denom = jnp.where(
                jnp.abs(old_phase) >= cfg.phase_ratio_tau,
                old_phase,
                jnp.where(old_phase >= 0.0, cfg.phase_ratio_tau, -cfg.phase_ratio_tau),
            )
            ratio_phase = new_phase / denom
            clipped_ratio_phase = jnp.clip(
                ratio_phase,
                1.0 - cfg.phase_clip_eps,
                1.0 + cfg.phase_clip_eps,
            )
            active = (ratio_phase * adv_phase >= clipped_ratio_phase * adv_phase).astype(
                new_phase.dtype
            )
            w = is_ratio * active * (adv_phase / denom) / batch_size

        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")

        return jax.lax.stop_gradient(w.astype(new_phase.dtype))

    def phase_linearized_objective(params, batch, weights):
        configs, _, _, _ = batch
        new_logpsi = apply_logpsi_explicit(model, params, configs)
        new_phase = jnp.imag(new_logpsi)
        return jnp.sum(jax.lax.stop_gradient(weights) * new_phase)

    def phase_objective(params, batch):
        weights = phase_backward_weights(params, batch)
        return phase_linearized_objective(params, batch, weights)

    def real_objective(params, batch):
        losses, _ = loss_terms_on_batch(params, batch)
        return losses["real"]

    def apply_phase_jacobian_baseline(params, batch, grads_phase):
        weights = phase_backward_weights(params, batch)
        weights_sq = jax.lax.stop_gradient(weights**2)

        c_num = jax.grad(phase_linearized_objective)(params, batch, weights_sq)
        c_den = jnp.sum(weights_sq) + cfg.phase_jacobian_baseline_eps
        c_tree = jax.tree_util.tree_map(
            lambda x, g: jax.lax.stop_gradient((x / c_den).astype(g.dtype)),
            c_num,
            grads_phase,
        )
        return tree_add(
            grads_phase,
            tree_scale(c_tree, -jax.lax.stop_gradient(jnp.sum(weights))),
        )

    @jax.jit
    def eval_sample_chunk(params, key):
        configs, logpsi = sample_cached_apply(
            model,
            {"params": params},
            key,
            batch_size=cfg.eval_batch_size,
            cache_dtype=cache_dtype,
        )
        configs = configs.astype(jnp.int32)
        sample_logp = cfg.machine_pow * jnp.real(logpsi)
        sample_surprisal = -sample_logp
        e_loc = local_energy(params, configs)
        e_real = jnp.real(e_loc)
        e_imag = jnp.imag(e_loc)
        magnetization = jnp.sum(configs, axis=-1)
        mag_deviation = magnetization - jnp.asarray(
            target_magnetization,
            dtype=magnetization.dtype,
        )
        return (
            jnp.sum(e_real),
            jnp.sum(e_real**2),
            jnp.sum(e_imag),
            jnp.sum(jnp.abs(e_imag)),
            jnp.sum(sample_surprisal),
            jnp.sum(sample_surprisal**2),
            jnp.sum(magnetization),
            jnp.sum(jnp.abs(magnetization)),
            jnp.sum(jnp.abs(mag_deviation)),
            jnp.max(jnp.abs(mag_deviation)),
            jnp.sum(mag_deviation == 0),
            e_real.shape[0],
        )

    def evaluate(state, eval_key):
        total_real = 0.0
        total_real_sq = 0.0
        total_imag = 0.0
        total_abs_imag = 0.0
        total_entropy = 0.0
        total_entropy_sq = 0.0
        total_mag = 0.0
        total_abs_mag = 0.0
        total_abs_mag_deviation = 0.0
        max_abs_mag_deviation = 0.0
        total_target_mag = 0
        total_n = 0

        n_chunks = (cfg.eval_n_samples + cfg.eval_batch_size - 1) // cfg.eval_batch_size
        keys = jax.random.split(eval_key, n_chunks)

        for key in keys:
            (
                real_sum,
                real_sq_sum,
                imag_sum,
                abs_imag_sum,
                entropy_sum,
                entropy_sq_sum,
                mag_sum,
                abs_mag_sum,
                abs_mag_dev_sum,
                max_abs_mag_dev,
                target_mag_sum,
                chunk_n,
            ) = eval_sample_chunk(
                state.params,
                key,
            )
            total_real += float(real_sum)
            total_real_sq += float(real_sq_sum)
            total_imag += float(imag_sum)
            total_abs_imag += float(abs_imag_sum)
            total_entropy += float(entropy_sum)
            total_entropy_sq += float(entropy_sq_sum)
            total_mag += float(mag_sum)
            total_abs_mag += float(abs_mag_sum)
            total_abs_mag_deviation += float(abs_mag_dev_sum)
            max_abs_mag_deviation = max(max_abs_mag_deviation, float(max_abs_mag_dev))
            total_target_mag += int(target_mag_sum)
            total_n += int(chunk_n)

        energy_real = total_real / total_n
        variance_real = max(total_real_sq / total_n - energy_real**2, 0.0)
        energy_imag = total_imag / total_n
        abs_imag_mean = total_abs_imag / total_n
        entropy_mean = total_entropy / total_n
        entropy_var = max(total_entropy_sq / total_n - entropy_mean**2, 0.0)
        entropy_std = math.sqrt(entropy_var)
        entropy_frac_max = entropy_mean / entropy_max if entropy_max > 0.0 else float("nan")
        denom = max(energy_real * energy_real, 1e-12)

        out = {
            "eval_E_mean_real": energy_real,
            "eval_E_mean_imag": energy_imag,
            "eval_E_var_real": variance_real,
            "eval_abs_E_imag_mean": abs_imag_mean,
            "eval_entropy": entropy_mean,
            "eval_entropy_std": entropy_std,
            "eval_entropy_per_site": entropy_mean / n_sites,
            "eval_entropy_frac_max": entropy_frac_max,
            "eval_entropy_max": entropy_max,
            "eval_V_score": float(n_sites * variance_real / denom),
            "eval_magnetization_mean": total_mag / total_n,
            "eval_magnetization_abs_mean": total_abs_mag / total_n,
            "eval_magnetization_deviation_abs_mean": total_abs_mag_deviation / total_n,
            "eval_magnetization_deviation_max_abs": max_abs_mag_deviation,
            "eval_magnetization_target_frac": total_target_mag / total_n,
            "eval_n_samples_used": total_n,
        }
        if cfg.reference_energy is not None:
            out["eval_reference_energy"] = float(cfg.reference_energy)
            out["eval_rel_error_reference"] = float(
                abs((energy_real - cfg.reference_energy) / cfg.reference_energy)
            )
        return out

    @partial(jax.jit, donate_argnums=(0,))
    def train_iter(state):
        state, batch = collect_batch(state)

        def total_objective(params, batch):
            losses, metrics = loss_terms_on_batch(params, batch)
            return losses["real"] + cfg.phase_coef * losses["phase"], (losses, metrics)

        def epoch_step(state, _):
            if cfg.use_phase_jacobian_baseline:
                losses, metrics = loss_terms_on_batch(state.params, batch)
                grads_real = jax.grad(real_objective)(state.params, batch)
                grads_phase = jax.grad(phase_objective)(state.params, batch)
                grads_phase = apply_phase_jacobian_baseline(
                    state.params,
                    batch,
                    grads_phase,
                )
                grads = tree_add(grads_real, tree_scale(grads_phase, cfg.phase_coef))
                grad_norm_real = optax.global_norm(grads_real)
                grad_norm_phase = optax.global_norm(grads_phase)
            else:
                (_, (losses, metrics)), grads = jax.value_and_grad(
                    total_objective,
                    argnums=0,
                    has_aux=True,
                )(state.params, batch)
                grad_norm_real = jnp.array(float("nan"), dtype=metrics["loss"].dtype)
                grad_norm_phase = jnp.array(float("nan"), dtype=metrics["loss"].dtype)

            grad_norm = optax.global_norm(grads)
            state = state.apply_gradients(grads=grads)
            metrics = {
                **metrics,
                "grad_norm": grad_norm,
                "grad_norm_real": grad_norm_real,
                "grad_norm_phase": grad_norm_phase,
                "param_norm": optax.global_norm(state.params),
            }
            return state, metrics

        state, metrics_hist = jax.lax.scan(
            epoch_step,
            state,
            xs=None,
            length=cfg.ppo_epochs,
        )
        metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics_hist)
        return state, metrics, batch

    timing_last_wall = None
    timing_last_iter = None
    timing_total_elapsed = 0.0
    timing_total_outer_steps = 0
    time_per_outer_iter = float("nan")
    time_per_inner_ppo_step = float("nan")
    time_per_outer_iter_recent = float("nan")
    time_per_inner_ppo_step_recent = float("nan")

    eval_base_key = jax.random.PRNGKey(cfg.seed + 20_000)

    for it in range(cfg.n_iter):
        state, metrics, batch = train_iter(state)

        if it % cfg.log_every == 0:
            if cfg.time_optimization_steps:
                jax.block_until_ready(metrics["loss"])
                now = time.perf_counter()
                if timing_last_wall is not None and timing_last_iter is not None:
                    elapsed = now - timing_last_wall
                    outer_steps = max(it - timing_last_iter, 1)
                    inner_steps = max(outer_steps * cfg.ppo_epochs, 1)
                    timing_total_elapsed += elapsed
                    timing_total_outer_steps += outer_steps
                    time_per_outer_iter_recent = elapsed / outer_steps
                    time_per_inner_ppo_step_recent = elapsed / inner_steps
                    time_per_outer_iter = timing_total_elapsed / max(
                        timing_total_outer_steps,
                        1,
                    )
                    time_per_inner_ppo_step = timing_total_elapsed / max(
                        timing_total_outer_steps * cfg.ppo_epochs,
                        1,
                    )
                timing_last_wall = now
                timing_last_iter = it

            current_lr = float(lr_schedule(state.step))
            msg = (
                f"it={it:04d}  loss={float(metrics['loss']): .6f}  "
                f"Lreal={float(metrics['loss_real']): .6f}  "
                f"Lphase={float(metrics['loss_phase']): .6f}  "
                f"E_real={float(metrics['E_mean_real']): .6f} ± {float(metrics['E_std_real']): .6f}  "
                f"E_imag={float(metrics['E_mean_imag']): .6f} ± {float(metrics['E_std_imag']): .6f}  "
                f"S={float(metrics['entropy']): .3f}  "
                f"S/Smax={float(metrics['entropy_frac_max']): .3f}  "
                f"ratio={float(metrics['ratio_mean']): .6f}  "
                f"clip={float(metrics['clip_frac']): .3f}  "
                f"kl={float(metrics['approx_kl']): .3e}  "
                f"dphi_rms={float(metrics['phase_delta_rms']): .3e}  "
                f"|M-target|={float(metrics['magnetization_deviation_abs_mean']): .3e}  "
                f"M_ok={float(metrics['magnetization_target_frac']): .3f}  "
                f"||g||={float(metrics['grad_norm']): .3e}  "
                f"||θ||={float(metrics['param_norm']): .3e}"
            )
            if cfg.use_phase_jacobian_baseline:
                msg += (
                    f"  ||g_real||={float(metrics['grad_norm_real']): .3e}"
                    f"  ||g_phase||={float(metrics['grad_norm_phase']): .3e}"
                )
            if cfg.reference_energy is not None:
                msg += f"  rel_ref={float(metrics['rel_error_reference']): .3e}"
            if cfg.time_optimization_steps:
                msg += (
                    f"  t/outer={time_per_outer_iter: .4f}s"
                    f"  t/ppo={time_per_inner_ppo_step: .4f}s"
                    f"  t/outer_recent={time_per_outer_iter_recent: .4f}s"
                    f"  t/ppo_recent={time_per_inner_ppo_step_recent: .4f}s"
                )
            print(msg)

            if cfg.wandb_mode != "disabled":
                log_dict = {
                    "iter": it,
                    "lr": current_lr,
                    **{k: float(v) for k, v in metrics.items()},
                }
                if cfg.time_optimization_steps:
                    log_dict.update({
                        "time_per_outer_iter": time_per_outer_iter,
                        "time_per_inner_ppo_step": time_per_inner_ppo_step,
                        "time_per_outer_iter_recent": time_per_outer_iter_recent,
                        "time_per_inner_ppo_step_recent": time_per_inner_ppo_step_recent,
                    })
                wandb.log(log_dict, step=it)

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_key = jax.random.fold_in(eval_base_key, it)
            eval_metrics = evaluate(state, eval_key)

            if cfg.log_gradient_info:
                grads_real = jax.grad(real_objective)(state.params, batch)
                grads_phase = jax.grad(phase_objective)(state.params, batch)
                if cfg.use_phase_jacobian_baseline:
                    grads_phase = apply_phase_jacobian_baseline(
                        state.params,
                        batch,
                        grads_phase,
                    )
                eval_metrics.update({
                    "eval_grad_norm_real": float(optax.global_norm(grads_real)),
                    "eval_grad_norm_phase": float(optax.global_norm(grads_phase)),
                    "eval_grad_norm_components": float(
                        jnp.sqrt(tree_l2_sq(grads_real) + tree_l2_sq(grads_phase))
                    ),
                })

            eval_msg = (
                f"[eval] it={it:04d}  "
                f"E_real={eval_metrics['eval_E_mean_real']:.8f}  "
                f"E_imag={eval_metrics['eval_E_mean_imag']:.8e}  "
                f"Var_real={eval_metrics['eval_E_var_real']:.8e}  "
                f"|E_imag|={eval_metrics['eval_abs_E_imag_mean']:.8e}  "
                f"S={eval_metrics['eval_entropy']:.6f}  "
                f"S/Smax={eval_metrics['eval_entropy_frac_max']:.3f}  "
                f"V_score={eval_metrics['eval_V_score']:.8e}  "
                f"|M-target|={eval_metrics['eval_magnetization_deviation_abs_mean']:.3e}  "
                f"M_ok={eval_metrics['eval_magnetization_target_frac']:.3f}"
            )
            if cfg.reference_energy is not None:
                eval_msg += f"  rel_ref={eval_metrics['eval_rel_error_reference']:.8e}"
            print(eval_msg)

            if cfg.wandb_mode != "disabled":
                wandb.log(eval_metrics, step=it)

        if cfg.save_checkpoint_every > 0 and it > 0 and it % cfg.save_checkpoint_every == 0:
            checkpointer.save(
                ckpt_dir / f"iter_{it:07d}",
                args=ocp.args.StandardSave(state.params),
            )

    final_ckpt_path = ckpt_dir / "final"
    checkpointer.save(final_ckpt_path, args=ocp.args.StandardSave(state.params))
    print(f"Saved final checkpoint to {final_ckpt_path}")

    final_metrics = evaluate(state, jax.random.PRNGKey(cfg.seed + 30_000))
    print(
        "Final sampled energy: "
        f"{final_metrics['eval_E_mean_real']:.10f} + "
        f"{final_metrics['eval_E_mean_imag']:.3e}j  "
        f"per site: {final_metrics['eval_E_mean_real'] / n_sites:.10f}  "
        f"entropy: {final_metrics['eval_entropy']:.6f} "
        f"({final_metrics['eval_entropy_frac_max']:.3f} of max)"
    )

    if cfg.wandb_mode != "disabled":
        wandb.log({
            "final_E_real": float(final_metrics["eval_E_mean_real"]),
            "final_E_imag": float(final_metrics["eval_E_mean_imag"]),
            "final_E_per_site": float(final_metrics["eval_E_mean_real"] / n_sites),
            "final_V_score": float(final_metrics["eval_V_score"]),
            "final_entropy": float(final_metrics["eval_entropy"]),
            "final_entropy_per_site": float(final_metrics["eval_entropy_per_site"]),
            "final_entropy_frac_max": float(final_metrics["eval_entropy_frac_max"]),
            "final_magnetization_deviation_abs_mean": float(
                final_metrics["eval_magnetization_deviation_abs_mean"]
            ),
            "final_magnetization_target_frac": float(
                final_metrics["eval_magnetization_target_frac"]
            ),
        })
        wandb.finish()


if __name__ == "__main__":
    main()
