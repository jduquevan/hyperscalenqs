from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import orbax.checkpoint as ocp

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

# Must be set before importing JAX; required by the ViT's float64 Dense layers.
import jax
jax.config.update("jax_enable_x64", True)

import optax
import jax.numpy as jnp
import netket as nk
from flax import linen as nn
from flax.training import train_state
from functools import partial
from transformers import FlaxAutoModel


@dataclass
class Args:
    seed: int = 42
    jax_platform_name: str = "gpu"

    # J1-J2 square lattice
    L: int = 10          # lattice side length; N = L*L
    pbc: bool = True
    J1: float = 1.0
    J2: float = 0.5
    total_sz: int = 0    # magnetization sector; 0 = physical ground state

    # ViT model
    vit_repo: str = "nqs-models/j1j2_square_10x10_05"
    # Model revision / branch to load from HuggingFace:
    #   "main"            – plain ViT with translation invariance among patches (~41 s/sweep)
    #   "symm_t"          – ViT with full translational symmetry (~166 s/sweep)
    #   "symm_trxy_ising" – ViT with translational, point-group and Sz-inversion symmetries (~3317 s/sweep)
    vit_revision: str = "main"

    # Sampling (MCMC)
    n_samples: int = 8192         # chain_length = n_samples // n_chains = 32 steps per chain
    machine_pow: int = 2
    n_chains: int = 256           # parallel MCMC chains
    mcmc_sweep_size: int = 0      # sweep size per chain step; 0 -> use N
    mcmc_d_max: int = 2           # MetropolisExchange max distance
    n_discard_per_chain: int = 100  # burn-in sweeps per chain per iter (~N=L*L)

    # Optimizer
    lr: float = 1e-5
    final_lr: float = 1e-7
    n_iter: int = 1_000_000
    optimizer: str = "adam"
    sgd_momentum: float = 0.0
    decay_rate: float = 0.5
    transition_steps: int = 100000
    use_phase_jacobian_baseline: bool = True
    phase_jacobian_baseline_eps: float = 1e-8

    # PPO / clipped objectives
    ppo_epochs: int = 4
    normalize_advantage: bool = True
    ppo_clip_eps: float = 1e-2   # larger default than AR; MCMC samples are correlated

    # SNIS (self-normalized importance sampling)
    use_snis: bool = True         # normalize IS weights so they sum to n_samples

    # Phase channel
    phase_loss_type: str = "delta_clip"   # {"delta_clip", "ratio"}
    phase_coef: float = 1.0
    phase_delta_clip: float = 0.3
    phase_ratio_tau: float = 1e-3
    phase_clip_eps: float = 1e-2
    center_imag_advantage: bool = True
    normalize_imag_advantage: bool = True
    phase_delta_l2_coef: float = 0.0

    # Logging / eval
    eval_every: int = 200
    eval_n_samples: int = 1048576
    eval_batch_size: int = 2048
    eval_n_discard_per_chain: int = 100  # match n_discard_per_chain to avoid stale configs

    wandb_project: str = "hyperscalenqs-j1j2-square"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    log_every: int = 10
    log_gradient_info: bool = False
    save_checkpoint_every: int = 0   # 0 = only save final; >0 = also save every N iters
    output_directory: Optional[str] = "."
    wandb_directory: Optional[str] = "."


ConfigStore.instance().store(name="config", node=Args)


class TrainState(train_state.TrainState):
    pass


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scale(tree, scalar):
    return jax.tree_util.tree_map(lambda x: scalar * x, tree)


def tree_l2_sq(tree):
    return sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree))


def tree_cast_like(tree, like_tree):
    return jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), tree, like_tree)


def make_tx(cfg: Args):
    lr_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=cfg.transition_steps,
        peak_value=1e-4,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=200.0,
    )

    opt = cfg.optimizer.lower()
    if opt == "sgd":
        tx = optax.sgd(learning_rate=lr_schedule, momentum=cfg.sgd_momentum)
    elif opt == "adam":
        tx = optax.adam(learning_rate=lr_schedule)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return tx, lr_schedule


# ---------------------------------------------------------------------------
# Local energy
# ---------------------------------------------------------------------------

def make_local_energy_fn(model: nn.Module, H):
    def local_energy(params, configs):
        configs = configs.astype(jnp.int32)
        x_primes, mels = H.get_conn_padded(configs)   # (B, K, N), (B, K)
        B, K, N = x_primes.shape
        logpsi_x  = model.apply({"params": params}, configs)
        logpsi_xp = model.apply(
            {"params": params},
            x_primes.reshape(B * K, N),
        ).reshape(B, K)
        ratios = jnp.exp(logpsi_xp - logpsi_x[:, None])
        e_loc  = jnp.sum(mels * ratios, axis=1)
        return e_loc
    return local_energy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_name="config")
def main(cfg: Args) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(jax.default_backend())

    N = cfg.L * cfg.L
    sweep_size = cfg.mcmc_sweep_size if cfg.mcmc_sweep_size > 0 else N

    ckpt_dir = Path(cfg.output_directory) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    if cfg.wandb_mode != "disabled":
        Path(cfg.wandb_directory).mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
            dir=cfg.wandb_directory,
            mode=cfg.wandb_mode,
        )

    # Hilbert space with magnetization constraint (total_sz=0 is the physical sector)
    hi = nk.hilbert.Spin(s=1 / 2, N=N, total_sz=cfg.total_sz)

    # 2-D square lattice with NN (color 0) and NNN (color 1) edges
    lattice = nk.graph.Hypercube(length=cfg.L, n_dim=2, pbc=cfg.pbc, max_neighbor_order=2)
    H_nk = nk.operator.Heisenberg(
        hilbert=hi,
        graph=lattice,
        J=[cfg.J1, cfg.J2],
        sign_rule=[False, False],
    )
    H = H_nk.to_jax_operator() if hasattr(H_nk, "to_jax_operator") else H_nk

    # ------------------------------------------------------------------
    # Load ViT architecture and randomly reinitialize weights
    # ------------------------------------------------------------------
    # FlaxAutoModel downloads architecture + pretrained weights from HuggingFace.
    # We discard those weights and reinit from scratch so we train with QPS.
    wf = FlaxAutoModel.from_pretrained(cfg.vit_repo, trust_remote_code=True, revision=cfg.vit_revision)
    # wf.module is the underlying Flax nn.Module (ViT); wf.params are the HF weights.
    model = wf.module

    dummy_input = jnp.zeros((1, N), dtype=jnp.float64)
    init_vars = model.init(jax.random.PRNGKey(cfg.seed), dummy_input)
    params0 = init_vars["params"]

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params0))
    print(f"Model: ViT ({cfg.vit_repo})  N={N}  Parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # MCMC sampler (MetropolisExchange; ViT is not autoregressive)
    # ------------------------------------------------------------------
    sampler = nk.sampler.MetropolisExchange(
        hilbert=hi,
        graph=lattice,
        d_max=cfg.mcmc_d_max,
        n_chains=cfg.n_chains,
        sweep_size=sweep_size,
    )

    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        n_discard_per_chain=cfg.n_discard_per_chain,
    )
    # Override randomly-initialized parameters with our own init
    vstate.parameters = params0

    eval_vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=cfg.eval_batch_size,
        seed=cfg.seed + 1,
        n_discard_per_chain=cfg.eval_n_discard_per_chain,
    )

    sampler_state0 = vstate.sampler_state
    chain_length   = vstate.chain_length   # = n_samples // n_chains

    tx, lr_schedule = make_tx(cfg)
    state = TrainState.create(apply_fn=model.apply, params=params0, tx=tx)

    local_energy = make_local_energy_fn(model, H)

    # ------------------------------------------------------------------
    # Batch collection
    # ------------------------------------------------------------------

    def collect_batch(old_params, sampler_state):
        configs, sampler_state = sampler.sample(
            model,
            {"params": old_params},
            state=sampler_state,
            chain_length=chain_length,
            return_log_probabilities=False,
        )
        configs = jax.lax.stop_gradient(configs)
        configs_flat = configs.reshape(-1, N)

        old_logpsi = model.apply({"params": old_params}, configs_flat)
        old_logp   = cfg.machine_pow * jnp.real(old_logpsi)
        old_phase  = jnp.imag(old_logpsi)
        old_logp   = jax.lax.stop_gradient(old_logp)
        old_phase  = jax.lax.stop_gradient(old_phase)

        e_loc = local_energy(old_params, configs_flat)
        e_loc = jax.lax.stop_gradient(e_loc)

        return (configs_flat, old_logp, old_phase, e_loc), sampler_state

    # ------------------------------------------------------------------
    # Loss / metrics
    # ------------------------------------------------------------------

    def loss_terms_on_batch(params, batch):
        configs_flat, old_logp, old_phase, e_loc = batch

        new_logpsi = model.apply({"params": params}, configs_flat)
        new_logp   = cfg.machine_pow * jnp.real(new_logpsi)
        new_phase  = jnp.imag(new_logpsi)

        # ===== Importance-sampling weights =====
        log_ratio        = jnp.clip(new_logp - old_logp, -20.0, 20.0)
        ratio_real       = jnp.exp(log_ratio)

        # Self-normalized IS: rescale weights so their mean == 1.
        # This corrects for the unknown normalisation constant Z_new/Z_old
        # when the wavefunction is not autoregressive (not normalised).
        if cfg.use_snis:
            snis_norm  = jax.lax.stop_gradient(jnp.mean(ratio_real) + 1e-8)
            eff_ratio  = ratio_real / snis_norm
        else:
            eff_ratio  = ratio_real

        clipped_ratio = jnp.clip(eff_ratio, 1.0 - cfg.ppo_clip_eps, 1.0 + cfg.ppo_clip_eps)

        # ===== Amplitude channel =====
        e_real   = jnp.real(e_loc)
        adv_real = e_real - jnp.mean(e_real)
        if cfg.normalize_advantage:
            adv_real = adv_real / (jnp.std(adv_real) + 1e-8)

        loss_real = jnp.mean(jnp.maximum(eff_ratio * adv_real, clipped_ratio * adv_real))

        # ===== Phase channel =====
        e_imag    = jnp.imag(e_loc)
        adv_phase = e_imag
        if cfg.center_imag_advantage:
            adv_phase = adv_phase - jnp.mean(adv_phase)
        if cfg.normalize_imag_advantage:
            adv_phase = adv_phase / (jnp.std(adv_phase) + 1e-8)

        phase_delta = jnp.atan2(
            jnp.sin(new_phase - old_phase),
            jnp.cos(new_phase - old_phase),
        )
        # Amplitude IS weight enters phase loss as a sample-reweighting factor.
        # Using stop_gradient so phase gradients don't flow through the IS weight.
        is_ratio = jax.lax.stop_gradient(eff_ratio)

        if cfg.phase_loss_type == "ratio":
            denom = jnp.where(
                jnp.abs(old_phase) >= cfg.phase_ratio_tau,
                old_phase,
                jnp.where(old_phase >= 0.0, cfg.phase_ratio_tau, -cfg.phase_ratio_tau),
            )
            ratio_phase         = new_phase / denom
            clipped_ratio_phase = jnp.clip(ratio_phase, 1.0 - cfg.phase_clip_eps, 1.0 + cfg.phase_clip_eps)
            loss_phase = jnp.mean(jnp.maximum(
                is_ratio * ratio_phase * adv_phase,
                is_ratio * clipped_ratio_phase * adv_phase,
            ))
            phase_stat = ratio_phase

        elif cfg.phase_loss_type == "delta_clip":
            clipped_phase_delta = jnp.clip(phase_delta, -cfg.phase_delta_clip, cfg.phase_delta_clip)
            loss_phase = jnp.mean(jnp.maximum(
                is_ratio * phase_delta * adv_phase,
                is_ratio * clipped_phase_delta * adv_phase,
            ))
            if cfg.phase_delta_l2_coef > 0.0:
                loss_phase = loss_phase + cfg.phase_delta_l2_coef * jnp.mean(phase_delta ** 2)
            phase_stat = phase_delta
        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")

        losses = {"real": loss_real, "phase": loss_phase}
        total_loss = loss_real + cfg.phase_coef * loss_phase

        metrics = {
            "loss":               total_loss,
            "loss_real":          loss_real,
            "loss_phase":         loss_phase,
            "E_mean_real":        jnp.mean(e_real),
            "E_std_real":         jnp.std(e_real),
            "E_mean_imag":        jnp.mean(e_imag),
            "E_std_imag":         jnp.std(e_imag),
            "ratio_mean":         jnp.mean(ratio_real),
            "snis_eff_ratio_mean": jnp.mean(eff_ratio),
            "clip_frac":          jnp.mean(jnp.abs(eff_ratio - 1.0) > cfg.ppo_clip_eps),
            "approx_kl":          jnp.mean(old_logp - new_logp),
            "phase_old_mean":     jnp.mean(old_phase),
            "phase_new_mean":     jnp.mean(new_phase),
            "phase_delta_mean":   jnp.mean(phase_delta),
            "phase_delta_rms":    jnp.sqrt(jnp.mean(phase_delta ** 2)),
            "phase_stat_mean":    jnp.mean(phase_stat),
            "phase_stat_abs_mean": jnp.mean(jnp.abs(phase_stat)),
        }

        return losses, metrics

    def phase_backward_weights(params, batch):
        configs_flat, _, old_phase, e_loc = batch
        new_logpsi = model.apply({"params": params}, configs_flat)
        new_phase  = jnp.imag(new_logpsi)
        new_logp   = cfg.machine_pow * jnp.real(new_logpsi)
        old_logp_  = batch[1]

        e_imag    = jnp.imag(e_loc)
        adv_phase = e_imag
        if cfg.center_imag_advantage:
            adv_phase = adv_phase - jnp.mean(adv_phase)
        if cfg.normalize_imag_advantage:
            adv_phase = adv_phase / (jnp.std(adv_phase) + 1e-8)

        B = adv_phase.shape[0]

        # Recompute eff_ratio the same way as in loss_terms_on_batch
        log_ratio  = jnp.clip(new_logp - old_logp_, -20.0, 20.0)
        ratio_real = jnp.exp(log_ratio)
        if cfg.use_snis:
            snis_norm  = jax.lax.stop_gradient(jnp.mean(ratio_real) + 1e-8)
            eff_ratio  = ratio_real / snis_norm
        else:
            eff_ratio  = ratio_real
        eff_ratio = jax.lax.stop_gradient(eff_ratio)

        if cfg.phase_loss_type == "delta_clip":
            phase_delta = jnp.atan2(
                jnp.sin(new_phase - old_phase),
                jnp.cos(new_phase - old_phase),
            )
            clipped = jnp.clip(phase_delta, -cfg.phase_delta_clip, cfg.phase_delta_clip)
            active  = (phase_delta * adv_phase >= clipped * adv_phase).astype(new_phase.dtype)
            w = eff_ratio * active * adv_phase / B
            if cfg.phase_delta_l2_coef > 0.0:
                w = w + (2.0 * cfg.phase_delta_l2_coef / B) * phase_delta

        elif cfg.phase_loss_type == "ratio":
            denom = jnp.where(
                jnp.abs(old_phase) >= cfg.phase_ratio_tau,
                old_phase,
                jnp.where(old_phase >= 0.0, cfg.phase_ratio_tau, -cfg.phase_ratio_tau),
            )
            ratio_phase         = new_phase / denom
            clipped_ratio_phase = jnp.clip(ratio_phase, 1.0 - cfg.phase_clip_eps, 1.0 + cfg.phase_clip_eps)
            active = (ratio_phase * adv_phase >= clipped_ratio_phase * adv_phase).astype(new_phase.dtype)
            w = eff_ratio * active * (adv_phase / denom) / B
        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")

        return jax.lax.stop_gradient(w.astype(new_phase.dtype))

    def phase_linearized_objective(params, batch, weights):
        configs_flat, _, _, _ = batch
        new_logpsi = model.apply({"params": params}, configs_flat)
        new_phase  = jnp.imag(new_logpsi)
        return jnp.sum(jax.lax.stop_gradient(weights) * new_phase)

    def apply_phase_jacobian_baseline(params, batch, grads_imag):
        """
        Applies the variance-reducing baseline:
            g <- g - (sum_i w_i) * c*
        where
            c* = (sum_i w_i^2 O_i) / (sum_i w_i^2)
        """
        w    = phase_backward_weights(params, batch)
        w_sq = jax.lax.stop_gradient(w ** 2)

        c_num = jax.grad(phase_linearized_objective)(params, batch, w_sq)
        c_den = jnp.sum(w_sq) + cfg.phase_jacobian_baseline_eps
        c_tree = jax.tree_util.tree_map(
            lambda x, g: jax.lax.stop_gradient((x / c_den).astype(g.dtype)),
            c_num,
            grads_imag,
        )

        sum_w = jax.lax.stop_gradient(jnp.sum(w))
        grads_imag = tree_add(grads_imag, tree_scale(c_tree, -sum_w))
        return grads_imag

    def phase_objective(params, batch):
        w = phase_backward_weights(params, batch)
        return phase_linearized_objective(params, batch, w)

    def real_objective(params, batch):
        losses, _ = loss_terms_on_batch(params, batch)
        return losses["real"]

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @jax.jit
    def eval_chunk_sums(params, configs):
        e_loc     = local_energy(params, configs)
        e_real    = jnp.real(e_loc)
        e_imag    = jnp.imag(e_loc)
        return (
            jnp.sum(e_real),
            jnp.sum(e_real ** 2),
            jnp.sum(e_imag),
            jnp.sum(jnp.abs(e_imag)),
            e_real.shape[0],
        )

    def evaluate(state):
        eval_vstate.parameters = state.params

        total_real = total_real_sq = total_imag = total_abs_imag = 0.0
        total_n    = 0
        n_remaining = cfg.eval_n_samples

        while n_remaining > 0:
            eval_vstate.sample()
            configs = jnp.asarray(eval_vstate.samples).reshape(-1, N)
            cur_n   = min(configs.shape[0], n_remaining)
            configs = configs[:cur_n]

            real_sum, real_sq_sum, imag_sum, abs_imag_sum, chunk_n = eval_chunk_sums(
                state.params, configs
            )
            total_real      += float(real_sum)
            total_real_sq   += float(real_sq_sum)
            total_imag      += float(imag_sum)
            total_abs_imag  += float(abs_imag_sum)
            total_n         += int(chunk_n)
            n_remaining     -= int(chunk_n)

        energy_real  = total_real / total_n
        variance_real = max(total_real_sq / total_n - energy_real ** 2, 0.0)
        energy_imag  = total_imag / total_n
        abs_imag_mean = total_abs_imag / total_n
        denom        = max(energy_real * energy_real, 1e-12)
        v_score      = float(N * variance_real / denom)

        return {
            "eval_E_mean_real":       energy_real,
            "eval_E_mean_imag":       energy_imag,
            "eval_E_var_real":        variance_real,
            "eval_abs_E_imag_mean":   abs_imag_mean,
            "eval_V_score":           v_score,
            "eval_n_samples_used":    total_n,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_iter(state, sampler_state):
        old_params = state.params
        batch, sampler_state = collect_batch(old_params, sampler_state)

        def total_objective(params, batch):
            losses, metrics = loss_terms_on_batch(params, batch)
            total_loss = losses["real"] + cfg.phase_coef * losses["phase"]
            return total_loss, (losses, metrics)

        def epoch_step(state, _):
            (loss, (losses, metrics)), grads = jax.value_and_grad(
                total_objective, argnums=0, has_aux=True
            )(state.params, batch)
            grad_norm = optax.global_norm(grads)
            state = state.apply_gradients(grads=grads)

            metrics = {
                **metrics,
                "grad_norm": grad_norm,
                "param_norm": optax.global_norm(state.params),
            }
            return state, metrics

        state, metrics_hist = jax.lax.scan(epoch_step, state, xs=None, length=cfg.ppo_epochs)
        metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics_hist)
        return state, sampler_state, metrics, batch

    sampler_state = sampler_state0

    for it in range(cfg.n_iter):
        state, sampler_state, metrics, batch = train_iter(state, sampler_state)

        if it % cfg.log_every == 0:
            current_lr = float(lr_schedule(state.step))
            msg = (
                f"it={it:04d}  loss={float(metrics['loss']): .6f}  "
                f"Lreal={float(metrics['loss_real']): .6f}  "
                f"Lphase={float(metrics['loss_phase']): .6f}  "
                f"E_real={float(metrics['E_mean_real']): .6f} ± {float(metrics['E_std_real']): .6f}  "
                f"E_imag={float(metrics['E_mean_imag']): .6f} ± {float(metrics['E_std_imag']): .6f}  "
                f"ratio={float(metrics['ratio_mean']): .6f}  "
                f"snis_ratio={float(metrics['snis_eff_ratio_mean']): .6f}  "
                f"clip={float(metrics['clip_frac']): .3f}  "
                f"kl={float(metrics['approx_kl']): .3e}  "
                f"dphi_rms={float(metrics['phase_delta_rms']): .3e}  "
                f"||g||={float(metrics['grad_norm']): .3e}  "
                f"||θ||={float(metrics['param_norm']): .3e}"
            )
            print(msg)

            if cfg.wandb_mode != "disabled":
                wandb.log({
                    "iter":                  it,
                    "lr":                    current_lr,
                    "loss":                  float(metrics["loss"]),
                    "loss_real":             float(metrics["loss_real"]),
                    "loss_phase":            float(metrics["loss_phase"]),
                    "E_mean_real":           float(metrics["E_mean_real"]),
                    "E_std_real":            float(metrics["E_std_real"]),
                    "E_mean_imag":           float(metrics["E_mean_imag"]),
                    "E_std_imag":            float(metrics["E_std_imag"]),
                    "grad_norm":             float(metrics["grad_norm"]),
                    "param_norm":            float(metrics["param_norm"]),
                    "ratio_mean":            float(metrics["ratio_mean"]),
                    "snis_eff_ratio_mean":   float(metrics["snis_eff_ratio_mean"]),
                    "clip_frac":             float(metrics["clip_frac"]),
                    "approx_kl":             float(metrics["approx_kl"]),
                    "phase_old_mean":        float(metrics["phase_old_mean"]),
                    "phase_new_mean":        float(metrics["phase_new_mean"]),
                    "phase_delta_mean":      float(metrics["phase_delta_mean"]),
                    "phase_delta_rms":       float(metrics["phase_delta_rms"]),
                    "phase_stat_mean":       float(metrics["phase_stat_mean"]),
                    "phase_stat_abs_mean":   float(metrics["phase_stat_abs_mean"]),
                }, step=it)

        if cfg.save_checkpoint_every > 0 and it > 0 and it % cfg.save_checkpoint_every == 0:
            checkpointer.save(
                ckpt_dir / f"iter_{it:07d}",
                args=ocp.args.StandardSave(state.params),
            )

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_metrics = evaluate(state)

            if cfg.log_gradient_info:
                grads_real  = jax.grad(real_objective)(state.params, batch)
                grads_phase = jax.grad(phase_objective)(state.params, batch)
                eval_metrics.update({
                    "eval_grad_norm_real":  float(optax.global_norm(grads_real)),
                    "eval_grad_norm_phase": float(optax.global_norm(grads_phase)),
                })

            print(
                f"[eval] it={it:04d}  "
                f"E_real={eval_metrics['eval_E_mean_real']:.8f}  "
                f"E_imag={eval_metrics['eval_E_mean_imag']:.8e}  "
                f"Var_real={eval_metrics['eval_E_var_real']:.8e}  "
                f"|E_imag|={eval_metrics['eval_abs_E_imag_mean']:.8e}  "
                f"V_score={eval_metrics['eval_V_score']:.8e}"
            )

            if cfg.wandb_mode != "disabled":
                wandb.log(eval_metrics, step=it)

    # Save final checkpoint
    final_ckpt_path = ckpt_dir / "final"
    checkpointer.save(
        final_ckpt_path,
        args=ocp.args.StandardSave(state.params),
    )
    print(f"Saved final checkpoint to {final_ckpt_path}")

    if cfg.wandb_mode != "disabled":
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"Final params: ViT ({cfg.vit_repo}), L={cfg.L}, J2/J1={cfg.J2/cfg.J1:.3f}",
        )
        artifact.add_dir(str(final_ckpt_path))
        wandb.log_artifact(artifact)

    vstate.parameters = state.params
    vstate.sample()
    final_stats = vstate.expect(H_nk)
    final_energy = final_stats.mean.real
    final_energy_err = final_stats.error_of_mean.real
    final_variance = final_stats.variance.real
    print(
        f"Final energy (NetKet expect): {final_energy:.10f} ± {final_energy_err:.2e}  "
        f"(per site: {final_energy / N:.10f})"
    )

    if cfg.wandb_mode != "disabled":
        wandb.log({
            "final_E":           final_energy,
            "final_E_per_site":  final_energy / N,
            "final_E_err":       final_energy_err,
            "final_E_variance":  final_variance,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
