from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
import jax.numpy as jnp
import netket as nk
import optax
from flax import linen as nn
from scipy.sparse.linalg import eigsh


@dataclass
class Args:
    seed: int = 42
    jax_platform_name: str = "gpu"
    compute_exact_diag: bool = True

    # Ising
    # N: int = 12
    # Gamma: float = -1.0
    # V: float = -1.0

    # Heisenberg chain
    # N: int = 12
    # J: float = 0.25
    # pbc: bool = True
    # sign_rule: bool = False

    # J2/J1
    N: int = 12
    J1: float = 1.0
    J2: float = 0.5
    pbc: bool = True
    Gamma: float = -1.0
    V: float = -1.0

    # Sampling / AR
    n_samples: int = 1024
    machine_pow: int = 2

    # minSR / SR driver
    n_iter: int = 1000000
    lr: float = 0.001
    sr_diag_shift: float = 1e-2
    use_ntk: bool = True          # True => minSR, False => standard SR
    on_the_fly: bool = True       
    chunk_size_bwd: Optional[int] = None
    mode: str = "complex"         # complex wavefunction

    # Architecture: match qps.py
    embed_dim: int = 32
    rnn_hidden: int = 256
    head_hidden: int = 256
    n_gru_layers: int = 3
    use_site_embedding: bool = True
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    # Logging / eval
    log_every: int = 10
    eval_every: int = 200
    eval_mode: str = "exact"      # {"exact", "sample"}
    eval_n_samples: int = 1048576
    eval_batch_size: int = 2048
    eval_n_discard_per_chain: int = 0
    eval_exact_chunk_size: Optional[int] = None

    # LR Scheduler
    lr_peak: float = 3e-4          # peak LR
    lr_div_factor: float = 10.0    # init = 2e-4
    lr_final_div_factor: float = 100.0  # final = 2e-6
    lr_pct_start: float = 0.15
    transition_steps: int = 4000

    wandb_project: str = "hyperscalenqs"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_directory: Optional[str] = "."


ConfigStore.instance().store(name="config", node=Args)


class GRULayer(nn.Module):
    hidden_size: int
    param_dtype: any = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, _ = x.shape

        cell = nn.GRUCell(
            features=self.hidden_size,
            param_dtype=self.param_dtype,
            name="cell",
        )

        carry = jnp.zeros((B, self.hidden_size), dtype=x.dtype)

        ys = []
        for t in range(T):
            carry, y = cell(carry, x[:, t, :])
            ys.append(y)

        return jnp.stack(ys, axis=1)


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
    param_dtype: any = jnp.float32

    @nn.compact
    def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        B, N = inputs.shape
        spin_tokens = (inputs > 0).astype(jnp.int32)

        bos_token = jnp.full((B, 1), 2, dtype=jnp.int32)
        prev_tokens = jnp.concatenate([bos_token, spin_tokens[:, :-1]], axis=1)

        tok_emb = nn.Embed(
            num_embeddings=3,
            features=self.embed_dim,
            name="tok_emb",
            param_dtype=self.param_dtype,
        )
        x = tok_emb(prev_tokens)

        if self.use_site_embedding:
            pos_ids = jnp.arange(N, dtype=jnp.int32)[None, :]
            pos_emb = nn.Embed(
                num_embeddings=N,
                features=self.embed_dim,
                name="pos_emb",
                param_dtype=self.param_dtype,
            )
            x = x + pos_emb(pos_ids)

        x = nn.Dense(self.rnn_hidden, name="input_proj", param_dtype=self.param_dtype)(x)
        x = nn.tanh(x)
        x = nn.LayerNorm(name="input_ln", param_dtype=self.param_dtype)(x)

        for layer_idx in range(self.n_gru_layers):
            h = GRULayer(
                hidden_size=self.rnn_hidden,
                name=f"gru_{layer_idx}",
                param_dtype=self.param_dtype,
            )(x)
            x = x + h

        x_amp = nn.Dense(self.head_hidden, name="amp_fc1", param_dtype=self.param_dtype)(x)
        x_amp = nn.gelu(x_amp)
        x_amp = nn.Dense(self.head_hidden, name="amp_fc2", param_dtype=self.param_dtype)(x_amp)
        x_amp = nn.gelu(x_amp)
        amp_logits = nn.Dense(
            self.hilbert.local_size,
            name="amp_head",
            param_dtype=self.param_dtype,
        )(x_amp)

        amp_logp = jax.nn.log_softmax(amp_logits, axis=-1)
        amp_logpsi = amp_logp / float(self.machine_pow)

        x_phase = nn.Dense(self.head_hidden, name="phase_fc1", param_dtype=self.param_dtype)(x)
        x_phase = nn.gelu(x_phase)
        x_phase = nn.Dense(self.head_hidden, name="phase_fc2", param_dtype=self.param_dtype)(x_phase)
        x_phase = nn.gelu(x_phase)

        phase_raw = nn.Dense(
            self.hilbert.local_size,
            name="phase_head",
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.normal(stddev=self.phase_init_std),
            bias_init=nn.initializers.zeros_init(),
        )(x_phase)

        phase_raw = phase_raw - jnp.mean(phase_raw, axis=-1, keepdims=True)
        phase = self.phase_scale * jnp.tanh(phase_raw)

        return amp_logpsi + 1j * phase


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
        
    # hi = nk.hilbert.Spin(s=1 / 2, N=cfg.N)
    # graph = nk.graph.Chain(length=cfg.N, pbc=cfg.pbc)

    # H_nk = nk.operator.Heisenberg(
    #     hi,
    #     graph=graph,
    #     J=cfg.J,
    #     sign_rule=cfg.sign_rule,
    # )   

    # Match qps.py Hamiltonian
    hi = nk.hilbert.Spin(s=0.5, N=cfg.N)
    g = nk.graph.Chain(length=cfg.N, pbc=cfg.pbc, max_neighbor_order=2)

    H_nk = nk.operator.Heisenberg(
        hilbert=hi,
        graph=g,
        J=[cfg.J1, cfg.J2],
        sign_rule=[False, False],
    )

    # hi = nk.hilbert.Spin(s=1 / 2, N=cfg.N)
    # graph = nk.graph.Chain(length=cfg.N, pbc=True)

    # H_nk = nk.operator.Ising(hi, graph, h=-cfg.Gamma, J=cfg.V)

    E_gs_exact = None
    if cfg.compute_exact_diag:
        sp_h = H_nk.to_sparse()
        eig_vals = eigsh(sp_h, k=1, which="SA", return_eigenvectors=False)
        E_gs_exact = float(jnp.real(eig_vals[0]))
        print(f"Exact ground-state energy: {E_gs_exact:.10f}")

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
        sampler=sampler,
        model=model,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        n_discard_per_chain=0,
    )

    lr_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=cfg.transition_steps,
        peak_value=cfg.lr_peak,
        pct_start=cfg.lr_pct_start,
        div_factor=cfg.lr_div_factor,
        final_div_factor=cfg.lr_final_div_factor,
    )

    lr_schedule = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=1500,
        decay_rate=0.5,
        staircase=True,
    )

    # This is the key line: use_ntk=True => minSR
    driver = nk.driver.VMC_SR(
        hamiltonian=H_nk,
        optimizer=optax.sgd(cfg.lr),
        variational_state=vstate,
        diag_shift=cfg.sr_diag_shift,
        use_ntk=cfg.use_ntk,
        on_the_fly=cfg.on_the_fly,
        chunk_size_bwd=cfg.chunk_size_bwd,
        mode=cfg.mode,
        momentum=0.8,
    )

    print("Number of parameters:", sum(x.size for x in jax.tree_util.tree_leaves(vstate.parameters)))

    eval_vstate = nk.vqs.MCState(
        sampler=sampler,
        model=model,
        n_samples=cfg.eval_batch_size,
        seed=cfg.seed + 1,
        n_discard_per_chain=cfg.eval_n_discard_per_chain,
    )

    def evaluate():
        if cfg.eval_mode == "exact":
            full_vstate = nk.vqs.FullSumState(
                hi,
                model,
                variables={"params": vstate.parameters},
                chunk_size=cfg.eval_exact_chunk_size,
                seed=cfg.seed + 2,
            )
            exact_stats = full_vstate.expect(H_nk)
            energy = exact_stats.mean
            e_real = float(jnp.real(energy))
            e_imag = float(jnp.imag(energy))
            out = {
                "eval_E_mean_real": e_real,
                "eval_E_mean_imag": e_imag,
            }
        else:
            eval_vstate.parameters = vstate.parameters

            total_real = 0.0
            total_real_sq = 0.0
            total_imag = 0.0
            total_abs_imag = 0.0
            total_n = 0
            n_remaining = cfg.eval_n_samples

            while n_remaining > 0:
                eval_vstate.sample()
                stats = eval_vstate.expect(H_nk)
                e = stats.mean

                # crude chunked accumulation
                cur_n = min(cfg.eval_batch_size, n_remaining)
                er = float(jnp.real(e))
                ei = float(jnp.imag(e))

                total_real += cur_n * er
                total_real_sq += cur_n * (er ** 2)
                total_imag += cur_n * ei
                total_abs_imag += cur_n * abs(ei)
                total_n += cur_n
                n_remaining -= cur_n

            e_real = total_real / total_n
            var_real = max(total_real_sq / total_n - e_real ** 2, 0.0)
            e_imag = total_imag / total_n
            abs_imag_mean = total_abs_imag / total_n
            denom = max(e_real * e_real, 1e-12)

            out = {
                "eval_E_mean_real": e_real,
                "eval_E_mean_imag": e_imag,
                "eval_E_var_real": var_real,
                "eval_abs_E_imag_mean": abs_imag_mean,
                "eval_V_score": float(cfg.N * var_real / denom),
                "eval_n_samples_used": total_n,
            }

        if E_gs_exact is not None:
            out["eval_E_exact"] = float(E_gs_exact)
            out["eval_rel_error_exact"] = float(abs((out["eval_E_mean_real"] - E_gs_exact) / E_gs_exact))

        return out

    for it in range(cfg.n_iter):
        driver.advance(1)

        if it % cfg.log_every == 0:
            stats = driver.energy if getattr(driver, "energy", None) is not None else driver.estimate(H_nk)
            e = stats.mean
            e_real = float(jnp.real(e))
            e_imag = float(jnp.imag(e))

            msg = f"it={it:04d}  E_real={e_real:.8f}  E_imag={e_imag:.3e}"
            if E_gs_exact is not None:
                rel_err = abs((e_real - E_gs_exact) / E_gs_exact)
                msg += f"  E_exact={E_gs_exact:.8f}  rel_err={rel_err:.3e}"
            print(msg)

            if cfg.wandb_mode != "disabled":
                log_dict = {
                    "iter": it,
                    "E_mean_real": e_real,
                    "E_mean_imag": e_imag,
                    "step_count": int(driver.step_count),
                }
                if E_gs_exact is not None:
                    log_dict["E_exact"] = E_gs_exact
                    log_dict["rel_error_exact"] = rel_err
                wandb.log(log_dict, step=it)

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_metrics = evaluate()
            print(
                f"[eval] it={it:04d}  "
                f"E_real={eval_metrics['eval_E_mean_real']:.8f}  "
                f"E_imag={eval_metrics['eval_E_mean_imag']:.3e}  "
                f"rel_err={eval_metrics.get('eval_rel_error_exact', float('nan')):.3e}"
            )
            if cfg.wandb_mode != "disabled":
                wandb.log(eval_metrics, step=it)

    if cfg.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()