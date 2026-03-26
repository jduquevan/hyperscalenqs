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
    sign_rule: bool = False

    # Sampling / AR setup
    n_samples: int = 1024
    machine_pow: int = 2

    # Optimizer
    lr: float = 3e-5
    final_lr: float = 1e-7
    n_iter: int = 1_000_000
    optimizer: str = "adam"
    sgd_momentum: float = 0.0
    decay_rate: float = 0.5
    transition_steps: int = 10_000
    use_phase_jacobian_baseline: bool = True
    phase_jacobian_baseline_eps: float = 1e-8

    # SR-specific
    phase_scale: float = math.pi
    sr_diag_shift: float = 1e-3
    sr_diag_scale: Optional[float] = None
    sr_cg_maxiter: int = 50
    sr_cg_tol: float = 1e-6
    sr_use_qgt_onthefly: bool = True

    # Architecture
    embed_dim: int = 32
    rnn_hidden: int = 256
    head_hidden: int = 256
    n_gru_layers: int = 3
    use_site_embedding: bool = True
    phase_init_std: float = 1e-3

    # Logging / eval
    eval_every: int = 100
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
    # lr_schedule = optax.schedules.cosine_onecycle_schedule(
    #     transition_steps=cfg.transition_steps,
    #     peak_value=1e-4,
    #     pct_start=0.15,
    #     div_factor=10.0,
    #     final_div_factor=100.0,
    # )
    lr_schedule = optax.constant_schedule(1e-3)

    # For SR baseline: use SGD
    tx = optax.sgd(learning_rate=lr_schedule, momentum=0.0)
    return tx, lr_schedule


def make_sr(cfg: Args):
    solver = partial(
        jax.scipy.sparse.linalg.cg,
        maxiter=cfg.sr_cg_maxiter,
        tol=cfg.sr_cg_tol,
    )

    qgt = nk.optimizer.qgt.QGTOnTheFly if cfg.sr_use_qgt_onthefly else None

    sr = nk.optimizer.SR(
        qgt=qgt,
        solver=solver,
        diag_shift=cfg.sr_diag_shift,
        diag_scale=cfg.sr_diag_scale,
    )
    return sr


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
    sr = make_sr(cfg)
    opt_state = tx.init(vstate.parameters)

    def sr_train_step(opt_state, step):
        energy_stats, grad = vstate.expect_and_grad(H_nk)

        dp = sr(vstate, grad, step=step)

        updates, opt_state = tx.update(dp, opt_state, vstate.parameters)
        vstate.parameters = optax.apply_updates(vstate.parameters, updates)

        metrics = {
            "loss": float(jnp.real(energy_stats.mean)),
            "E_mean_real": float(jnp.real(energy_stats.mean)),
            "E_mean_imag": float(jnp.imag(energy_stats.mean)),
            "grad_norm": float(optax.global_norm(grad)),
            "sr_update_norm": float(optax.global_norm(dp)),
            "param_norm": float(optax.global_norm(vstate.parameters)),
        }
        return opt_state, metrics

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

    
    def evaluate(params):
        if cfg.eval_mode == "exact":
            full_vstate = nk.vqs.FullSumState(
                hi,
                model,
                variables={"params": params},
                chunk_size=cfg.eval_exact_chunk_size,
                seed=cfg.seed + 2,
            )
            exact_stats = full_vstate.expect(H_nk)
            energy = exact_stats.mean
            energy_real = float(jnp.real(energy))
            energy_imag = float(jnp.imag(energy))
            extra_metrics = {}

        elif cfg.eval_mode == "sample":
            eval_vstate.parameters = params

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
                    params, configs
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

            extra_metrics = {
                "eval_E_var_real": variance_real,
                "eval_abs_E_imag_mean": abs_imag_mean,
                "eval_V_score": v_score,
                "eval_n_samples_used": total_n,
            }
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
        

    for it in range(cfg.n_iter):
        opt_state, metrics = sr_train_step(opt_state, it)

        if it % cfg.log_every == 0:
            current_lr = float(lr_schedule(it))
            print(
                f"it={it:04d}  "
                f"E_real={metrics['E_mean_real']:.8f}  "
                f"E_imag={metrics['E_mean_imag']:.3e}  "
                f"||g||={metrics['grad_norm']:.3e}  "
                f"||sr||={metrics['sr_update_norm']:.3e}  "
                f"||θ||={metrics['param_norm']:.3e}"
            )

            if cfg.wandb_mode != "disabled":
                wandb.log({"iter": it, "lr": current_lr, **metrics}, step=it)

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_metrics = evaluate(vstate.parameters)
            print(
                f"[eval] it={it:04d}  "
                f"E_real={eval_metrics['eval_E_mean_real']:.8f}  "
                f"E_imag={eval_metrics['eval_E_mean_imag']:.8e}  "
                + (
                    f"E_exact={eval_metrics['eval_E_exact']:.8f}  "
                    f"rel_err={eval_metrics['eval_rel_error_exact']:.8e}"
                    if "eval_rel_error_exact" in eval_metrics else ""
                )
            )

            if cfg.wandb_mode != "disabled":
                wandb.log(eval_metrics, step=it)

    print("Final energy (NetKet expect):", vstate.expect(H_nk))

    if cfg.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()