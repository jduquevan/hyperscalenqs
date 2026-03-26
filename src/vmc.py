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

    # Heisenberg chain
    N: int = 12
    J: float = 0.25
    pbc: bool = True
    sign_rule: bool = True

    # Sampling / AR setup
    n_samples: int = 512
    machine_pow: int = 2
    enforce_total_sz_zero: bool = True  # only valid for even N

    # Optimizer
    lr: float = 3e-5
    n_iter: int = 1_000_000
    optimizer: str = "adam"  # {"adam", "sgd"}
    sgd_momentum: float = 0.0
    decay_rate: float = 0.5
    transition_steps: int = 10_000

    # Architecture
    embed_dim: int = 32
    rnn_hidden: int = 128
    head_hidden: int = 128
    n_gru_layers: int = 2
    use_site_embedding: bool = True
    phase_scale: float = 0.0
    phase_init_std: float = 1e-3

    # Logging / eval
    eval_every: int = 100
    eval_n_samples: int = 262144
    eval_batch_size: int = 2048
    eval_n_discard_per_chain: int = 0
    log_every: int = 10

    wandb_project: str = "hyperscalenqs"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    output_directory: Optional[str] = "."
    wandb_directory: Optional[str] = "."


ConfigStore.instance().store(name="config", node=Args)


class TrainState(train_state.TrainState):
    pass


def make_tx(cfg: Args):
    lr_schedule = optax.exponential_decay(
        init_value=cfg.lr,
        transition_steps=cfg.transition_steps,
        decay_rate=cfg.decay_rate,
        staircase=True,
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


class ComplexRecurrentAR(nk.models.AbstractARNN):
    hilbert: object
    embed_dim: int = 32
    rnn_hidden: int = 128
    head_hidden: int = 128
    n_gru_layers: int = 2
    use_site_embedding: bool = True
    machine_pow: int = 2
    phase_scale: float = math.pi
    enforce_total_sz_zero: bool = False
    phase_init_std: float = 1e-3

    def _zero_mag_mask(self, spin_tokens: jax.Array) -> jax.Array:
        """
        spin_tokens: (B, N) in {0,1}, where 0 -> down, 1 -> up
        returns mask: (B, N, 2) for local states [down, up]
        """
        B, N = spin_tokens.shape
        if N % 2 != 0:
            raise ValueError("enforce_total_sz_zero=True requires even N.")

        target_up = N // 2
        target_down = N // 2

        up_used = jnp.concatenate(
            [
                jnp.zeros((B, 1), dtype=jnp.int32),
                jnp.cumsum(spin_tokens[:, :-1], axis=1),
            ],
            axis=1,
        )
        down_used = jnp.arange(N, dtype=jnp.int32)[None, :] - up_used

        valid_down = down_used < target_down
        valid_up = up_used < target_up

        return jnp.stack([valid_down, valid_up], axis=-1)

    @nn.compact
    def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        B, N = inputs.shape

        # NetKet spin configs are typically in {-1, +1}; map to {0,1}
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

        for layer_idx in range(self.n_gru_layers):
            h = GRULayer(hidden_size=self.rnn_hidden, name=f"gru_{layer_idx}")(x)
            x = x + h

        x = nn.Dense(self.head_hidden, name="head_fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.head_hidden, name="head_fc2")(x)
        x = nn.gelu(x)

        amp_logits = nn.Dense(self.hilbert.local_size, name="amp_head")(x)

        if self.enforce_total_sz_zero:
            valid_mask = self._zero_mag_mask(spin_tokens)
            neg_inf = jnp.full_like(amp_logits, -1.0e30)
            amp_logits = jnp.where(valid_mask, amp_logits, neg_inf)

        amp_logp = jax.nn.log_softmax(amp_logits, axis=-1)
        amp_logpsi = amp_logp / float(self.machine_pow)

        phase_raw = nn.Dense(
            self.hilbert.local_size,
            name="phase_head",
            kernel_init=nn.initializers.normal(stddev=self.phase_init_std),
            bias_init=nn.initializers.zeros_init(),
        )(x)
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
        e_loc = jnp.sum(mels * ratios, axis=1)  # complex (B,)
        return e_loc

    return local_energy


@hydra.main(version_base="1.3", config_name="config")
def main(cfg: Args) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(jax.default_backend())

    if cfg.enforce_total_sz_zero and cfg.N % 2 != 0:
        raise ValueError("Zero-magnetization masking requires even N.")

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

    hi = nk.hilbert.Spin(s=1 / 2, N=cfg.N)
    graph = nk.graph.Chain(length=cfg.N, pbc=cfg.pbc)

    H_nk = nk.operator.Heisenberg(
        hi,
        graph=graph,
        J=cfg.J,
        sign_rule=cfg.sign_rule,
    )

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
        enforce_total_sz_zero=cfg.enforce_total_sz_zero,
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
    sampler_state0 = vstate.sampler_state
    chain_length = vstate.chain_length
    N = cfg.N

    tx, lr_schedule = make_tx(cfg)
    state = TrainState.create(apply_fn=model.apply, params=params0, tx=tx)

    local_energy = make_local_energy_fn(model, H)

    def collect_batch(params, sampler_state):
        configs, sampler_state = sampler.sample(
            model,
            {"params": params},
            state=sampler_state,
            chain_length=chain_length,
            return_log_probabilities=False,
        )
        configs = jax.lax.stop_gradient(configs)
        configs_flat = configs.reshape(-1, N)

        e_loc = local_energy(params, configs_flat)
        e_loc = jax.lax.stop_gradient(e_loc)

        return (configs_flat, e_loc), sampler_state

    def vmc_loss_on_batch(params, batch):
        configs_flat, e_loc = batch

        logpsi = model.apply({"params": params}, configs_flat)  # complex (B,)

        # Standard centered VMC covariance estimator.
        # For samples drawn from P(x) ∝ |psi(x)|^machine_pow.
        e_mean = jnp.mean(e_loc)
        e_centered = jax.lax.stop_gradient(e_loc - e_mean)

        loss = cfg.machine_pow * jnp.real(
            jnp.mean(jnp.conj(e_centered) * logpsi)
        )

        e_real = jnp.real(e_loc)
        e_imag = jnp.imag(e_loc)

        metrics = {
            "loss": loss,
            "E_mean_real": jnp.mean(e_real),
            "E_std_real": jnp.std(e_real),
            "E_mean_imag": jnp.mean(e_imag),
            "E_std_imag": jnp.std(e_imag),
        }

        if E_gs_exact is not None:
            rel_error_exact = jnp.abs((metrics["E_mean_real"] - E_gs_exact) / E_gs_exact)
            metrics["E_exact"] = jnp.array(E_gs_exact)
            metrics["rel_error_exact"] = rel_error_exact

        return loss, metrics

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

        out = {
            "eval_E_mean_real": energy_real,
            "eval_E_var_real": variance_real,
            "eval_E_mean_imag": energy_imag,
            "eval_abs_E_imag_mean": abs_imag_mean,
            "eval_V_score": v_score,
            "eval_n_samples_used": total_n,
        }

        if E_gs_exact is not None:
            out["eval_E_exact"] = float(E_gs_exact)
            out["eval_rel_error_exact"] = float(abs((energy_real - E_gs_exact) / E_gs_exact))

        return out

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_iter(state, sampler_state):
        batch, sampler_state = collect_batch(state.params, sampler_state)

        (loss, metrics), grads = jax.value_and_grad(
            vmc_loss_on_batch, argnums=0, has_aux=True
        )(state.params, batch)

        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)

        metrics = {
            **metrics,
            "grad_norm": grad_norm,
            "param_norm": optax.global_norm(state.params),
        }
        return state, sampler_state, metrics

    sampler_state = sampler_state0

    for it in range(cfg.n_iter):
        state, sampler_state, metrics = train_iter(state, sampler_state)

        if it % cfg.log_every == 0:
            current_lr = float(lr_schedule(state.step))
            msg = (
                f"it={it:04d}  "
                f"loss={float(metrics['loss']): .6f}  "
                f"E_real={float(metrics['E_mean_real']): .6f} ± {float(metrics['E_std_real']): .6f}  "
                f"E_imag={float(metrics['E_mean_imag']): .6f} ± {float(metrics['E_std_imag']): .6f}  "
                f"lr={current_lr: .3e}  "
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
                    "E_mean_real": float(metrics["E_mean_real"]),
                    "E_std_real": float(metrics["E_std_real"]),
                    "E_mean_imag": float(metrics["E_mean_imag"]),
                    "E_std_imag": float(metrics["E_std_imag"]),
                    "grad_norm": float(metrics["grad_norm"]),
                    "param_norm": float(metrics["param_norm"]),
                }
                if E_gs_exact is not None:
                    log_dict["E_exact"] = float(metrics["E_exact"])
                    log_dict["rel_error_exact"] = float(metrics["rel_error_exact"])
                wandb.log(log_dict, step=it)

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_metrics = evaluate(state)

            eval_msg = (
                f"[eval] it={it:04d}  "
                f"E_real={eval_metrics['eval_E_mean_real']:.8f}  "
                f"Var_real={eval_metrics['eval_E_var_real']:.8e}  "
                f"E_imag={eval_metrics['eval_E_mean_imag']:.8e}  "
                f"|E_imag|={eval_metrics['eval_abs_E_imag_mean']:.8e}  "
                f"V_score={eval_metrics['eval_V_score']:.8e}"
            )
            if "eval_rel_error_exact" in eval_metrics:
                eval_msg += (
                    f"  E_exact={eval_metrics['eval_E_exact']:.8f}"
                    f"  rel_err={eval_metrics['eval_rel_error_exact']:.8e}"
                )
            print(eval_msg)

            if cfg.wandb_mode != "disabled":
                wandb.log(eval_metrics, step=it)

    vstate.parameters = state.params
    vstate.sample()
    print("Final energy (NetKet expect):", vstate.expect(H))

    if cfg.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()