from __future__ import annotations

import os
import math

import hydra
import wandb
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
from typing import Optional

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
import optax
import jax.numpy as jnp
import netket as nk
from flax import linen as nn
from flax.training import train_state
from functools import partial
from scipy.sparse.linalg import eigsh


@dataclass
class Args:
    seed: int = 42
    jax_platform_name: str = "gpu"
    compute_exact_diag: bool = True

    # Hamiltonian
    hamiltonian: str = "ising"   # {"ising", "heisenberg", "j1j2"}

    N: int = 12
    pbc: bool = True

    # Ising
    Gamma: float = -1.0
    V: float = -1.0

    # Heisenberg chain
    J: float = 0.25
    sign_rule: bool = False

    # J1-J2 chain
    J1: float = 1.0
    J2: float = 0.5
    j1j2_sign_rule: bool = False

    # Sampling / AR setup
    n_samples: int = 1024
    machine_pow: int = 2

    # Optimizer
    lr: float = 1e-5
    peak_lr: float = 1e-4
    pct_start: float = 0.15
    div_factor: float = 10.0
    final_div_factor: float = 200.0
    n_iter: int = 1_000_000
    optimizer: str = "adam"
    sgd_momentum: float = 0.0
    decay_rate: float = 0.5
    # transition_steps: int = 100_000
    transition_steps: int = 40000
    use_phase_jacobian_baseline: bool = True
    phase_jacobian_baseline_eps: float = 1e-8

    # PPO / clipped objectives
    ppo_epochs: int = 4
    normalize_advantage: bool = True
    ppo_clip_eps: float = 1e-3

    # Phase channel
    phase_loss_type: str = "delta_clip"   # {"delta_clip", "ratio"}
    phase_coef: float = 1.0
    phase_scale: float = math.pi
    phase_delta_clip: float = 0.3
    phase_ratio_tau: float = 1e-3
    phase_clip_eps: float = 1e-2
    center_imag_advantage: bool = True
    normalize_imag_advantage: bool = True
    phase_delta_l2_coef: float = 0.0

    # Architecture
    embed_dim: int = 32
    rnn_hidden: int = 256
    head_hidden: int = 256
    n_gru_layers: int = 3
    use_site_embedding: bool = True
    phase_init_std: float = 1e-3

    # Logging / eval
    eval_every: int = 200
    eval_n_samples: int = 1048576
    eval_batch_size: int = 2048
    eval_n_discard_per_chain: int = 0
    eval_mode: str = "exact"   # {"sample", "exact"}
    eval_exact_chunk_size: Optional[int] = None

    wandb_project: str = "hyperscalenqs"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    log_every: int = 10
    log_gradient_info: bool = False
    output_directory: Optional[str] = "."
    wandb_directory: Optional[str] = "."
    wandb_tags: Optional[str] = None


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


def build_hamiltonian(cfg: Args):
    hi = nk.hilbert.Spin(s=1 / 2, N=cfg.N)

    if cfg.hamiltonian == "ising":
        graph = nk.graph.Chain(length=cfg.N, pbc=cfg.pbc)
        H_nk = nk.operator.Ising(hi, graph, h=-cfg.Gamma, J=cfg.V)

    elif cfg.hamiltonian == "heisenberg":
        graph = nk.graph.Chain(length=cfg.N, pbc=cfg.pbc)
        H_nk = nk.operator.Heisenberg(
            hi,
            graph=graph,
            J=cfg.J,
            sign_rule=cfg.sign_rule,
        )

    elif cfg.hamiltonian == "j1j2":
        graph = nk.graph.Chain(length=cfg.N, pbc=cfg.pbc, max_neighbor_order=2)
        H_nk = nk.operator.Heisenberg(
            hilbert=hi,
            graph=graph,
            J=[cfg.J1, cfg.J2],
            sign_rule=[cfg.j1j2_sign_rule, cfg.j1j2_sign_rule],
        )

    else:
        raise ValueError(
            f"Unknown hamiltonian: {cfg.hamiltonian}. "
            "Choose from {'ising', 'heisenberg', 'j1j2'}."
        )

    return hi, H_nk


def make_tx(cfg: Args):

    lr_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=cfg.transition_steps,
        peak_value=cfg.peak_lr,
        pct_start=cfg.pct_start,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor,
    )

    opt = cfg.optimizer.lower()
    if opt == "sgd":
        tx = optax.sgd(learning_rate=lr_schedule, momentum=cfg.sgd_momentum)
    elif opt == "adam":
        tx = optax.adam(learning_rate=lr_schedule)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return tx, lr_schedule


class GRULayer(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, _ = x.shape
        carry0 = jnp.zeros((B, self.hidden_size), dtype=x.dtype)

        ScanGRU = nn.scan(
            nn.GRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        gru = ScanGRU(features=self.hidden_size)
        _, h = gru(carry0, x)
        return h

class ResidualMLPBlock(nn.Module):
    width: int
    residual_scale: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.width)(h)
        h = nn.gelu(h)
        h = nn.Dense(x.shape[-1])(h)
        return x + self.residual_scale * h


class ComplexRecurrentAR(nk.models.AbstractARNN):
    hilbert: object
    embed_dim: int = 32
    rnn_hidden: int = 128
    head_hidden: int = 128
    n_gru_layers: int = 2
    use_site_embedding: bool = True
    machine_pow: int = 2
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3


    @nn.compact
    def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        B, N = inputs.shape

        # NetKet spin configs are typically in {-1, +1}; map to {0, 1}
        spin_tokens = (inputs > 0).astype(jnp.int32)

        bos_token = jnp.full((B, 1), 2, dtype=jnp.int32)
        prev_tokens = jnp.concatenate([bos_token, spin_tokens[:, :-1]], axis=1)

        tok_emb = nn.Embed(num_embeddings=3, features=self.embed_dim, name="tok_emb")
        x = tok_emb(prev_tokens)

        if self.use_site_embedding:
            pos_ids = jnp.arange(N, dtype=jnp.int32)[None, :]
            pos_emb = nn.Embed(num_embeddings=N, features=self.embed_dim, name="pos_emb")
            x = x + pos_emb(pos_ids)

        x = nn.Dense(self.rnn_hidden, name="input_proj")(x)
        x = nn.tanh(x)
        x = nn.LayerNorm(name="input_ln")(x)

        for layer_idx in range(self.n_gru_layers):
            h = GRULayer(hidden_size=self.rnn_hidden, name=f"gru_{layer_idx}")(x)
            x = x + h

        # ----- amplitude tower -----
        x_amp = nn.Dense(self.head_hidden, name="amp_fc1")(x)
        x_amp = nn.gelu(x_amp)
        x_amp = nn.Dense(self.head_hidden, name="amp_fc2")(x_amp)
        x_amp = nn.gelu(x_amp)
        amp_logits = nn.Dense(self.hilbert.local_size, name="amp_head")(x_amp)

        amp_logp = jax.nn.log_softmax(amp_logits, axis=-1)
        amp_logpsi = amp_logp / float(self.machine_pow)

        # ----- phase tower -----
        x_phase = nn.Dense(self.head_hidden, name="phase_fc1")(x)
        x_phase = nn.gelu(x_phase)
        x_phase = nn.Dense(self.head_hidden, name="phase_fc2")(x_phase)
        x_phase = nn.gelu(x_phase)

        phase_raw = nn.Dense(
            self.hilbert.local_size,
            name="phase_head",
            kernel_init=nn.initializers.normal(stddev=self.phase_init_std),
            bias_init=nn.initializers.zeros_init(),
        )(x_phase)

        phase_raw = phase_raw - jnp.mean(phase_raw, axis=-1, keepdims=True)

        phase = self.phase_scale * jnp.tanh(phase_raw)

        return amp_logpsi + 1j * phase


def make_local_energy_fn(model: nn.Module, H):
    def local_energy(params, configs):
        configs = configs.astype(jnp.int32)

        x_primes, mels = H.get_conn_padded(configs)  # (B, K, N), (B, K)
        B, K, N = x_primes.shape

        logpsi_x = model.apply({"params": params}, configs)  # (B,)
        logpsi_xp = model.apply(
            {"params": params},
            x_primes.reshape(B * K, N),
        ).reshape(B, K)

        ratios = jnp.exp(logpsi_xp - logpsi_x[:, None])  # complex (B, K)
        e_loc = jnp.sum(mels * ratios, axis=1)          # complex (B,)
        return e_loc

    return local_energy


@hydra.main(version_base="1.3", config_name="config")
def main(cfg: Args) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(jax.default_backend())

    if cfg.wandb_mode != "disabled":
        Path(cfg.wandb_directory).mkdir(parents=True, exist_ok=True)
        tags = None if cfg.wandb_tags is None else [cfg.wandb_tags]

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            tags=tags,
            config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
            dir=cfg.wandb_directory,
            mode=cfg.wandb_mode,
        )

    hi, H_nk = build_hamiltonian(cfg)

    E_gs_exact = None
    if cfg.compute_exact_diag:
        sp_h = H_nk.to_sparse()
        eig_vals = eigsh(sp_h, k=1, which="SA", return_eigenvectors=False)
        E_gs_exact = float(jnp.real(eig_vals[0]))
        print(f"Exact ground-state energy (sparse eigsh): {E_gs_exact:.10f}")

    H = H_nk.to_jax_operator() if hasattr(H_nk, "to_jax_operator") else H_nk

    model = ComplexRecurrentAR(
        hilbert=hi,
        embed_dim=cfg.embed_dim,
        rnn_hidden=cfg.rnn_hidden,
        head_hidden=cfg.head_hidden,
        n_gru_layers=cfg.n_gru_layers,
        use_site_embedding=cfg.use_site_embedding,
        machine_pow=cfg.machine_pow,
        phase_scale=cfg.phase_scale,
        phase_init_std=cfg.phase_init_std,
    )

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
    print(f"Number of parameters: {n_params:,}")
    
    sampler_state0 = vstate.sampler_state
    chain_length = vstate.chain_length
    N = cfg.N

    tx, lr_schedule = make_tx(cfg)
    state = TrainState.create(apply_fn=model.apply, params=params0, tx=tx)

    local_energy = make_local_energy_fn(model, H)

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

        old_logpsi = model.apply({"params": old_params}, configs_flat)  # complex (B,)
        old_logp = cfg.machine_pow * jnp.real(old_logpsi)               # log P
        old_phase = jnp.imag(old_logpsi)                                # total phase
        old_logp = jax.lax.stop_gradient(old_logp)
        old_phase = jax.lax.stop_gradient(old_phase)

        e_loc = local_energy(old_params, configs_flat)                  # complex (B,)
        e_loc = jax.lax.stop_gradient(e_loc)

        batch = (configs_flat, old_logp, old_phase, e_loc)
        return batch, sampler_state


    def loss_terms_on_batch(params, batch):
        configs_flat, old_logp, old_phase, e_loc = batch

        new_logpsi = model.apply({"params": params}, configs_flat)  # complex (B,)
        new_logp = cfg.machine_pow * jnp.real(new_logpsi)
        new_phase = jnp.imag(new_logpsi)

        # ===== Real / amplitude channel =====
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

        surrogate_real_1 = ratio_real * adv_real
        surrogate_real_2 = clipped_ratio_real * adv_real
        loss_real = jnp.mean(jnp.maximum(surrogate_real_1, surrogate_real_2))

        # ===== Imag / phase channel =====
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

        if cfg.phase_loss_type == "ratio":
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
            surrogate_phase_1 = is_ratio * ratio_phase * adv_phase
            surrogate_phase_2 = is_ratio * clipped_ratio_phase * adv_phase
            loss_phase = jnp.mean(jnp.maximum(surrogate_phase_1, surrogate_phase_2))
            phase_stat = ratio_phase

        elif cfg.phase_loss_type == "delta_clip":
            clipped_phase_delta = jnp.clip(
                phase_delta,
                -cfg.phase_delta_clip,
                cfg.phase_delta_clip,
            )
            surrogate_phase_1 = is_ratio * phase_delta * adv_phase
            surrogate_phase_2 = is_ratio * clipped_phase_delta * adv_phase
            loss_phase = jnp.mean(jnp.maximum(surrogate_phase_1, surrogate_phase_2))
            if cfg.phase_delta_l2_coef > 0.0:
                loss_phase = loss_phase + cfg.phase_delta_l2_coef * jnp.mean(phase_delta**2)
            phase_stat = phase_delta
        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")

        losses = {
            "real": loss_real,
            "phase": loss_phase,
        }

        total_loss = losses["real"] + cfg.phase_coef * losses["phase"]

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

        if E_gs_exact is not None:
            rel_error_exact = jnp.abs((metrics["E_mean_real"] - E_gs_exact) / E_gs_exact)
            metrics["E_exact"] = jnp.array(E_gs_exact)
            metrics["rel_error_exact"] = rel_error_exact

        return losses, metrics

    def phase_backward_weights(params, batch):
        """
        Returns detached per-sample weights w_i such that

            grad_phase = sum_i w_i * d new_phase_i / d params

        matches the first-order gradient of your current phase surrogate.
        """
        configs_flat, _, old_phase, e_loc = batch

        new_logpsi = model.apply({"params": params}, configs_flat)
        new_phase = jnp.imag(new_logpsi)

        e_imag = jnp.imag(e_loc)
        adv_phase = e_imag
        if cfg.center_imag_advantage:
            adv_phase = adv_phase - jnp.mean(adv_phase)
        if cfg.normalize_imag_advantage:
            adv_phase = adv_phase / (jnp.std(adv_phase) + 1e-8)

        B = adv_phase.shape[0]

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

            surrogate_1 = phase_delta * adv_phase
            surrogate_2 = clipped_phase_delta * adv_phase

            active = (surrogate_1 >= surrogate_2).astype(new_phase.dtype)
            w = active * adv_phase / B

            if cfg.phase_delta_l2_coef > 0.0:
                w = w + (2.0 * cfg.phase_delta_l2_coef / B) * phase_delta

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

            surrogate_1 = ratio_phase * adv_phase
            surrogate_2 = clipped_ratio_phase * adv_phase

            active = (surrogate_1 >= surrogate_2).astype(new_phase.dtype)
            w = active * (adv_phase / denom) / B

        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")
        
        w = w.astype(new_phase.dtype)
        return jax.lax.stop_gradient(w)


    def phase_linearized_objective(params, batch, weights):
        """
        Linearized objective whose gradient is exactly:
            sum_i weights_i * d new_phase_i / d params
        """
        configs_flat, _, _, _ = batch
        new_logpsi = model.apply({"params": params}, configs_flat)
        new_phase = jnp.imag(new_logpsi)
        weights = jax.lax.stop_gradient(weights)
        return jnp.sum(weights * new_phase)


    def apply_phase_jacobian_baseline(params, batch, grads_imag):
        """
        Applies the variance-reducing baseline:
            g <- g - (sum_i w_i) * c*
        where
            c* = (sum_i w_i^2 O_i) / (sum_i w_i^2)
        """
        w = phase_backward_weights(params, batch)
        w_sq = jax.lax.stop_gradient(w ** 2)

        c_num = jax.grad(phase_linearized_objective)(params, batch, w_sq)
        c_den = jnp.sum(w_sq) + cfg.phase_jacobian_baseline_eps
        c_tree = jax.tree_util.tree_map(
            lambda x, g: jax.lax.stop_gradient((x / c_den).astype(g.dtype)),
            c_num,
            grads_imag,
        )

        sum_w = jax.lax.stop_gradient(jnp.sum(w))

        grads_imag = tree_add(
            grads_imag,
            tree_scale(c_tree, -sum_w),
        )
        return grads_imag

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
        extra_metrics = {
            "eval_E_var_real": float("nan"),
            "eval_abs_E_imag_mean": float("nan"),
            "eval_V_score": float("nan"),
            "eval_n_samples_used": float("nan"),
        }

        if cfg.eval_mode == "exact":
            full_vstate = nk.vqs.FullSumState(
                hi,
                model,
                variables={"params": state.params},
                chunk_size=cfg.eval_exact_chunk_size,
                seed=cfg.seed + 2,
            )

            exact_stats = full_vstate.expect(H_nk)
            energy = exact_stats.mean

            energy_real = float(jnp.real(energy))
            energy_imag = float(jnp.imag(energy))

            variance_real = float(jnp.real(exact_stats.variance))
            denom = max(energy_real * energy_real, 1e-12)
            v_score = float(cfg.N * variance_real / denom)

            extra_metrics.update({
                "eval_E_var_real": variance_real,
                "eval_V_score": v_score,
                "eval_n_samples_used": float(hi.n_states),
            })

        elif cfg.eval_mode == "sample":
            eval_vstate.parameters = state.params

            total_real = 0.0
            total_real_sq = 0.0
            total_imag = 0.0
            total_abs_imag = 0.0
            total_n = 0

            n_remaining = cfg.eval_n_samples

            while n_remaining > 0:
                eval_vstate.sample()
                configs = jnp.asarray(eval_vstate.samples).reshape(-1, N)

                cur_n = min(configs.shape[0], n_remaining)
                configs = configs[:cur_n]

                real_sum, real_sq_sum, imag_sum, abs_imag_sum, chunk_n = eval_chunk_sums(
                    state.params, configs
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
            v_score = float(cfg.N * variance_real / denom)

            extra_metrics.update({
                "eval_E_var_real": variance_real,
                "eval_abs_E_imag_mean": abs_imag_mean,
                "eval_V_score": v_score,
                "eval_n_samples_used": total_n,
            })

        else:
            raise ValueError(f"Unknown eval_mode: {cfg.eval_mode}")

        out = {
            "eval_E_mean_real": energy_real,
            "eval_E_mean_imag": energy_imag,
            **extra_metrics,
        }

        if E_gs_exact is not None:
            out["eval_E_exact"] = float(E_gs_exact)
            out["eval_rel_error_exact"] = float(abs((energy_real - E_gs_exact) / E_gs_exact))

        return out

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
            # losses, metrics = loss_terms_on_batch(state.params, batch)

            # grads_real = jax.grad(real_objective)(state.params, batch)
            # grads_imag = jax.grad(phase_objective)(state.params, batch)

            # if cfg.use_phase_jacobian_baseline:
            #     grads_imag = apply_phase_jacobian_baseline(state.params, batch, grads_imag)

            # grads = tree_add(
            #     grads_real,
            #     tree_scale(grads_imag, cfg.phase_coef),
            # )

            # grad_norm_real = optax.global_norm(grads_real)
            # grad_norm_imag = optax.global_norm(grads_imag)
            grad_norm = optax.global_norm(grads)
            state = state.apply_gradients(grads=grads)

            metrics = {
                **metrics,
                # "grad_norm_real": grad_norm_real,
                # "grad_norm_imag": grad_norm_imag,
                "grad_norm": grad_norm,
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

    
    def real_objective(params, batch):
        losses, _ = loss_terms_on_batch(params, batch)
        return losses["real"]

    # def phase_objective(params, batch):
    #     losses, _ = loss_terms_on_batch(params, batch)
    #     return cfg.phase_coef * losses["phase"]
    def phase_objective(params, batch):
        w = phase_backward_weights(params, batch)
        return phase_linearized_objective(params, batch, w)


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
                f"clip={float(metrics['clip_frac']): .3f}  "
                f"kl={float(metrics['approx_kl']): .3e}  "
                f"dphi_rms={float(metrics['phase_delta_rms']): .3e}  "
                f"||g||={float(metrics['grad_norm']): .3e}  "
                f"||θ||={float(metrics['param_norm']): .3e}"
            )
            if E_gs_exact is not None:
                msg += (
                    f"  E_exact={float(metrics['E_exact']): .6f}"
                    f"  rel_err_batch={float(metrics['rel_error_exact']): .3e}"
                )
            print(msg)

            if cfg.wandb_mode != "disabled":
                log_dict = {
                    "iter": it,
                    "lr": current_lr,
                    "loss": float(metrics["loss"]),
                    "loss_real": float(metrics["loss_real"]),
                    "loss_phase": float(metrics["loss_phase"]),
                    "E_mean_real": float(metrics["E_mean_real"]),
                    "E_std_real": float(metrics["E_std_real"]),
                    "E_mean_imag": float(metrics["E_mean_imag"]),
                    "E_std_imag": float(metrics["E_std_imag"]),
                    "grad_norm": float(metrics["grad_norm"]),
                    "param_norm": float(metrics["param_norm"]),
                    "ratio_mean": float(metrics["ratio_mean"]),
                    "clip_frac": float(metrics["clip_frac"]),
                    "approx_kl": float(metrics["approx_kl"]),
                    "phase_old_mean": float(metrics["phase_old_mean"]),
                    "phase_new_mean": float(metrics["phase_new_mean"]),
                    "phase_delta_mean": float(metrics["phase_delta_mean"]),
                    "phase_delta_rms": float(metrics["phase_delta_rms"]),
                    "phase_stat_mean": float(metrics["phase_stat_mean"]),
                    "phase_stat_abs_mean": float(metrics["phase_stat_abs_mean"]),
                }
                if E_gs_exact is not None:
                    log_dict["E_exact"] = float(metrics["E_exact"])
                    log_dict["rel_error_exact"] = float(metrics["rel_error_exact"])
                wandb.log(log_dict, step=it)

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_metrics = evaluate(state)

            if cfg.log_gradient_info:
                grads_real = jax.grad(real_objective)(state.params, batch)
                grads_phase = jax.grad(phase_objective)(state.params, batch)

                grad_norm_real = optax.global_norm(grads_real)
                grad_norm_phase = optax.global_norm(grads_phase)

                if "amp_head" in grads_real:
                    grad_norm_real_amp_head = optax.global_norm({"amp_head": grads_real["amp_head"]})
                else:
                    grad_norm_real_amp_head = jnp.array(float("nan"))

                if "phase_head" in grads_phase:
                    grad_norm_phase_head = optax.global_norm({"phase_head": grads_phase["phase_head"]})
                else:
                    grad_norm_phase_head = jnp.array(float("nan"))

                grad_norm_components = jnp.sqrt(
                    tree_l2_sq(grads_real) + tree_l2_sq(grads_phase)
                )

                grad_dict = {
                    "eval_grad_norm_real": float(grad_norm_real),
                    "eval_grad_norm_phase": float(grad_norm_phase),
                    "eval_grad_norm_components": float(grad_norm_components),
                    "eval_grad_norm_real_amp_head": float(grad_norm_real_amp_head),
                    "eval_grad_norm_phase_head": float(grad_norm_phase_head),
                }
                eval_metrics.update(grad_dict)

            eval_msg = (
                f"[eval] it={it:04d}  "
                f"E_real={eval_metrics['eval_E_mean_real']:.8f}  "
                f"E_imag={eval_metrics['eval_E_mean_imag']:.8e}  "
                f"Var_real={eval_metrics['eval_E_var_real']:.8e}  "
                f"|E_imag|={eval_metrics['eval_abs_E_imag_mean']:.8e}  "
                f"V_score={eval_metrics['eval_V_score']:.8e}  "
            )

            if "eval_rel_error_exact" in eval_metrics:
                eval_msg += (
                    f"E_exact={eval_metrics['eval_E_exact']:.8f}"
                    f"  rel_err={eval_metrics['eval_rel_error_exact']:.8e}"
                )

            print(eval_msg)

            if cfg.wandb_mode != "disabled":
                wandb.log(eval_metrics, step=it,)

    vstate.parameters = state.params
    vstate.sample()
    print("Final energy (NetKet expect):", vstate.expect(H))

    if cfg.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()