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

from .ar_vit_rope_2d import ComplexRoPETransformer2DAR


@dataclass
class Args:
    seed: int = 42
    jax_platform_name: str = "gpu"

    L: int = 10
    pbc: bool = True
    J1: float = 1.0
    J2: float = 0.5
    total_sz: int = 0

    n_samples: int = 512
    machine_pow: int = 2

    width: int = 128
    depth: int = 4
    num_heads: int = 4
    embed_dim: int = 64
    mlp_dim: Optional[int] = None
    head_hidden: int = 128
    use_site_embedding: bool = True
    rope_base: float = 10_000.0
    dropout: float = 0.0
    residual_scale: float = 1.0
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    peak_lr: float = 1e-4
    pct_start: float = 0.15
    div_factor: float = 10.0
    final_div_factor: float = 200.0
    transition_steps: int = 100_000
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

    eval_every: int = 200
    eval_n_samples: int = 262_144
    eval_batch_size: int = 2048
    eval_n_discard_per_chain: int = 0
    reference_energy: Optional[float] = None

    local_energy_conn_chunk_size: int = 8

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
    wandb_tags: Optional[str] = "qps,j1j2_square,ar_vit_rope_2d"


ConfigStore.instance().store(name="config", node=Args)


class TrainState(train_state.TrainState):
    pass


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scale(tree, scalar):
    return jax.tree_util.tree_map(lambda x: scalar * x, tree)


def tree_l2_sq(tree):
    return sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree))


def make_tx(cfg: Args):
    lr_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=cfg.transition_steps,
        peak_value=cfg.peak_lr,
        pct_start=cfg.pct_start,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor,
    )

    opt = cfg.optimizer.lower()
    if opt == "adam":
        tx = optax.adam(lr_schedule)
    elif opt == "sgd":
        tx = optax.sgd(lr_schedule, momentum=cfg.sgd_momentum)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return tx, lr_schedule


def build_square_j1j2(cfg: Args):
    n_sites = cfg.L * cfg.L
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
    return ComplexRoPETransformer2DAR(
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
    )


def make_local_energy_fn(model, H, conn_chunk_size: int = 1):
    if conn_chunk_size < 1:
        raise ValueError("conn_chunk_size must be >= 1")

    def local_energy(params, configs):
        configs = configs.astype(jnp.int32)

        x_primes, mels = H.get_conn_padded(configs)  # (B, K, N), (B, K)
        logpsi_x = model.apply({"params": params}, configs)  # (B,)

        compute_dtype = logpsi_x.dtype
        mels = mels.astype(compute_dtype)

        B, K, N = x_primes.shape
        C = conn_chunk_size

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

        x_chunks = jnp.swapaxes(x_chunks, 0, 1)  # (n_chunks, B, C, N)
        m_chunks = jnp.swapaxes(m_chunks, 0, 1)  # (n_chunks, B, C)

        def scan_chunk(e_acc, xs):
            x_chunk, mel_chunk = xs  # (B, C, N), (B, C)

            x_flat = x_chunk.reshape(B * C, N)
            logpsi_xp = model.apply({"params": params}, x_flat)
            logpsi_xp = logpsi_xp.astype(compute_dtype).reshape(B, C)

            ratio = jnp.exp(logpsi_xp - logpsi_x[:, None])
            e_acc = e_acc + jnp.sum(mel_chunk * ratio, axis=1)
            return e_acc, None

        e0 = jnp.zeros(logpsi_x.shape, dtype=compute_dtype)
        e_loc, _ = jax.lax.scan(scan_chunk, e0, (x_chunks, m_chunks))

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

    dummy_configs = jnp.ones((2, n_sites), dtype=jnp.int32)
    dummy_xp, dummy_mels = h_jax.get_conn_padded(dummy_configs)
    jax.block_until_ready(dummy_xp)
    jax.block_until_ready(dummy_mels)
    print(f"Local-energy padded connectivity K={dummy_xp.shape[1]}")

    model = make_model(cfg, hi)

    sampler = nk.sampler.ARDirectSampler(hilbert=hi)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        n_discard_per_chain=0,
    )
    eval_vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=cfg.eval_batch_size,
        seed=cfg.seed + 1,
        n_discard_per_chain=cfg.eval_n_discard_per_chain,
    )

    params0 = vstate.variables["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params0))
    print(
        f"Model: ComplexRoPETransformer2DAR  L={cfg.L}  N={n_sites}  "
        f"J2/J1={cfg.J2 / cfg.J1:.3f}  Parameters: {n_params:,}"
    )

    tx, lr_schedule = make_tx(cfg)
    state = TrainState.create(apply_fn=model.apply, params=params0, tx=tx)
    sampler_state = vstate.sampler_state
    chain_length = vstate.chain_length
    local_energy = make_local_energy_fn(model, h_jax)

    def collect_batch(old_params, sampler_state):
        configs, sampler_state = sampler.sample(
            model,
            {"params": old_params},
            state=sampler_state,
            chain_length=chain_length,
            return_log_probabilities=False,
        )
        configs = jax.lax.stop_gradient(configs).reshape(-1, n_sites)

        old_logpsi = model.apply({"params": old_params}, configs)
        old_logp = jax.lax.stop_gradient(cfg.machine_pow * jnp.real(old_logpsi))
        old_phase = jax.lax.stop_gradient(jnp.imag(old_logpsi))
        e_loc = jax.lax.stop_gradient(local_energy(old_params, configs))
        return (configs, old_logp, old_phase, e_loc), sampler_state

    def loss_terms_on_batch(params, batch):
        configs, old_logp, old_phase, e_loc = batch

        new_logpsi = model.apply({"params": params}, configs)
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
            "ratio_mean": jnp.mean(ratio_real),
            "clip_frac": jnp.mean(jnp.abs(ratio_real - 1.0) > cfg.ppo_clip_eps),
            "approx_kl": jnp.mean(old_logp - new_logp),
            "phase_old_mean": jnp.mean(old_phase),
            "phase_new_mean": jnp.mean(new_phase),
            "phase_delta_mean": jnp.mean(phase_delta),
            "phase_delta_rms": jnp.sqrt(jnp.mean(phase_delta**2)),
            "phase_stat_mean": jnp.mean(phase_stat),
            "phase_stat_abs_mean": jnp.mean(jnp.abs(phase_stat)),
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
        new_logpsi = model.apply({"params": params}, configs)
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
        new_logpsi = model.apply({"params": params}, configs)
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
    def eval_chunk_sums(params, configs):
        e_loc = local_energy(params, configs)
        e_real = jnp.real(e_loc)
        e_imag = jnp.imag(e_loc)
        return (
            jnp.sum(e_real),
            jnp.sum(e_real**2),
            jnp.sum(e_imag),
            jnp.sum(jnp.abs(e_imag)),
            e_real.shape[0],
        )

    def evaluate(state):
        eval_vstate.parameters = state.params
        total_real = 0.0
        total_real_sq = 0.0
        total_imag = 0.0
        total_abs_imag = 0.0
        total_n = 0
        n_remaining = cfg.eval_n_samples

        while n_remaining > 0:
            eval_vstate.sample()
            configs = jnp.asarray(eval_vstate.samples).reshape(-1, n_sites)
            cur_n = min(configs.shape[0], n_remaining)
            configs = configs[:cur_n]

            real_sum, real_sq_sum, imag_sum, abs_imag_sum, chunk_n = eval_chunk_sums(
                state.params,
                configs,
            )
            total_real += float(real_sum)
            total_real_sq += float(real_sq_sum)
            total_imag += float(imag_sum)
            total_abs_imag += float(abs_imag_sum)
            total_n += int(chunk_n)
            n_remaining -= int(chunk_n)

        energy_real = total_real / total_n
        variance_real = max(total_real_sq / total_n - energy_real**2, 0.0)
        energy_imag = total_imag / total_n
        abs_imag_mean = total_abs_imag / total_n
        denom = max(energy_real * energy_real, 1e-12)

        out = {
            "eval_E_mean_real": energy_real,
            "eval_E_mean_imag": energy_imag,
            "eval_E_var_real": variance_real,
            "eval_abs_E_imag_mean": abs_imag_mean,
            "eval_V_score": float(n_sites * variance_real / denom),
            "eval_n_samples_used": total_n,
        }
        if cfg.reference_energy is not None:
            out["eval_reference_energy"] = float(cfg.reference_energy)
            out["eval_rel_error_reference"] = float(
                abs((energy_real - cfg.reference_energy) / cfg.reference_energy)
            )
        return out

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_iter(state, sampler_state):
        old_params = state.params
        batch, sampler_state = collect_batch(old_params, sampler_state)

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
        return state, sampler_state, metrics, batch

    timing_last_wall = None
    timing_last_iter = None
    timing_total_elapsed = 0.0
    timing_total_outer_steps = 0
    time_per_outer_iter = float("nan")
    time_per_inner_ppo_step = float("nan")
    time_per_outer_iter_recent = float("nan")
    time_per_inner_ppo_step_recent = float("nan")

    for it in range(cfg.n_iter):
        state, sampler_state, metrics, batch = train_iter(state, sampler_state)

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
                f"ratio={float(metrics['ratio_mean']): .6f}  "
                f"clip={float(metrics['clip_frac']): .3f}  "
                f"kl={float(metrics['approx_kl']): .3e}  "
                f"dphi_rms={float(metrics['phase_delta_rms']): .3e}  "
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
            eval_metrics = evaluate(state)

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
                f"V_score={eval_metrics['eval_V_score']:.8e}"
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

    vstate.parameters = state.params
    vstate.sample()
    final_stats = vstate.expect(h_nk)
    final_energy = final_stats.mean
    print(
        "Final energy (NetKet expect): "
        f"{final_energy.real:.10f} + {final_energy.imag:.3e}j  "
        f"per site: {final_energy.real / n_sites:.10f}"
    )

    if cfg.wandb_mode != "disabled":
        wandb.log({
            "final_E_real": float(final_energy.real),
            "final_E_imag": float(final_energy.imag),
            "final_E_per_site": float(final_energy.real / n_sites),
        })
        wandb.finish()


if __name__ == "__main__":
    main()
