from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from huggingface_hub.constants import HF_HOME

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training import train_state
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from scipy.sparse.linalg import eigsh
import hydra

from .base_model import CommonInit, CommonParams
from .common import simple_es_tree_key
from .auto import get_model, load

jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


# -----------------------------------------------------------------------------
# Global RWKV runtime state used by the NetKet wrapper.
# -----------------------------------------------------------------------------
_RWKV_MODEL = None
_RWKV_CONFIG = None
_RWKV_BASE_EVO_KEYS = None


@dataclass
class Args:
    seed: int = 42
    jax_platform_name: str = "gpu"
    compute_exact_diag: bool = True

    # Hamiltonian 
    hamiltonian: str = "ising"  # one of: ising, heisenberg
    N: int = 12
    Gamma: float = -1.0
    V: float = -1.0
    J: float = 0.25
    pbc: bool = True
    sign_rule: bool = False

    # PPO / optimization
    n_iter: int = 1000000
    batch_size: int = 64
    eval_n_samples: int = 4096
    eval_batch_size: int = 128
    eval_n_discard_per_chain: int = 0
    ppo_epochs: int = 4
    sampler_uses_variables_dict: bool = True
    normalize_advantage: bool = True
    ppo_clip_eps: float = 1e-3
    kl_coef: float = 0.0
    lr: float = 1e-6
    optimizer: str = "adam"
    sgd_momentum: float = 0.0
    decay_rate: float = 0.5
    transition_steps: int = 8000
    machine_pow: int = 2

    # RWKV setup
    model_choice: str = "7g1.5B"
    rwkv_type: str = "AssociativeScanRWKV"
    dtype: Optional[str] = None
    load_model: bool = False
    load_path: Optional[str] = None

    # Optional explicit token ids for the two local states and BOS.
    bos_token_id: Optional[int] = None
    spin_down_token_id: Optional[int] = None
    spin_up_token_id: Optional[int] = None

    # Logging / eval
    log_every: int = 10
    eval_every: int = 200
    wandb_project: str = "hyperscalenqs"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_directory: Optional[str] = "."
    output_directory: Optional[str] = "."


ConfigStore.instance().store(name="config", node=Args)


class TrainState(train_state.TrainState):
    pass


class IdentityNoiser:
    @staticmethod
    def get_noisy_standard(frozen_noiser_params, noiser_params, params, es_tree_key, iterinfo):
        return params

    @staticmethod
    def do_mm(frozen_noiser_params, noiser_params, params, es_tree_key, iterinfo, x):
        return x @ params.T

    @staticmethod
    def do_Tmm(frozen_noiser_params, noiser_params, params, es_tree_key, iterinfo, x):
        return x @ params

    @staticmethod
    def do_emb(frozen_noiser_params, noiser_params, params, es_tree_key, iterinfo, x):
        return params[x]


_IDENTITY_NOISER = IdentityNoiser()

def _resolve_ckpt_path(load_path: str | Path) -> Path:
    p = Path(load_path)
    if p.is_dir():
        cand = p / "latest.model"
        if not cand.exists():
            files = sorted(list(p.glob("epoch_*.model")))
            if files:
                cand = files[-1]
            else:
                raise FileNotFoundError(f"No .model files in directory: {p}")
        return cand
    if p.suffix != ".model":
        raise ValueError(f"Expected a .model file or directory, got: {p}")
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    return p


def _to_jnp_tree(x: Any, dtype_str: Optional[str]):
    if dtype_str is None:
        return jax.tree_util.tree_map(
            lambda y: jnp.asarray(y) if isinstance(y, np.ndarray) else y,
            x,
        )
    target_dtype = jnp.bfloat16 if dtype_str == "bfloat16" else jnp.float32
    return jax.tree_util.tree_map(
        lambda y: jnp.asarray(y, dtype=target_dtype) if isinstance(y, np.ndarray) else y,
        x,
    )


def _make_common_params(params_rwkv):
    return CommonParams(
        noiser=_IDENTITY_NOISER,
        frozen_noiser_params=None,
        noiser_params=None,
        frozen_params=_RWKV_CONFIG,
        params=params_rwkv,
        es_tree_key=_RWKV_BASE_EVO_KEYS,
        iterinfo=None,
    )


def _count_params(params) -> int:
    return int(
        sum(int(np.prod(x.shape)) for x in jax.tree_util.tree_leaves(params) if hasattr(x, "shape"))
    )


def _encode_single_token(tokenizer, text: str, fallback: int) -> int:
    if tokenizer is None:
        return fallback

    encode_fn = getattr(tokenizer, "encode", None)
    if encode_fn is None:
        return fallback

    try:
        toks = encode_fn(text)
        if isinstance(toks, np.ndarray):
            toks = toks.tolist()
        if isinstance(toks, (list, tuple)) and len(toks) == 1:
            return int(toks[0])
    except Exception:
        pass
    return fallback


def _rwkv_forward_logits(params, tokens: jax.Array, state):
    """Run the raw RWKV stack without the GRPO noiser path."""
    tokens = jnp.asarray(tokens, dtype=jnp.int32).reshape(-1)
    common_params = CommonParams(params=params, es_tree_key=_RWKV_BASE_EVO_KEYS)
    x = _RWKV_MODEL.embed(common_params, tokens)
    new_starts = jnp.zeros((tokens.shape[0],), dtype=jnp.bool_)
    x, state = _RWKV_MODEL.forward_seq(
        common_params,
        x,
        state,
        length=tokens.shape[0],
        new_starts=new_starts,
    )
    logits = _RWKV_MODEL.outhead(common_params, x)
    return logits, state

def _rwkv_forward_features(params_rwkv, tokens, state):
    tokens = jnp.asarray(tokens, dtype=jnp.int32).reshape(-1)
    common_params = _make_common_params(params_rwkv)
    x = _RWKV_MODEL.embed(common_params, tokens)
    new_starts = jnp.zeros((tokens.shape[0],), dtype=jnp.bool_)
    x, state = _RWKV_MODEL.forward_seq(
        common_params, x, state, length=tokens.shape[0], new_starts=new_starts
    )
    h = _RWKV_MODEL.features(common_params, x)
    return h, state


class RWKVAutoregressiveNQS(nk.models.AbstractARNN):
    """
    NetKet autoregressive wrapper around a pretrained / finetunable RWKV-7 model.
    """

    hilbert: Any
    bos_token_id: int
    spin_down_token_id: int
    spin_up_token_id: int
    machine_pow: int = 2

    def _single_conditionals(self, params, sigma: jax.Array) -> jax.Array:
        # NetKet spin configs are typically in {-1, +1}.
        spin_tokens = jnp.where(
            sigma > 0,
            jnp.asarray(self.spin_up_token_id, dtype=jnp.int32),
            jnp.asarray(self.spin_down_token_id, dtype=jnp.int32),
        )
        prev_tokens = jnp.concatenate(
            [jnp.asarray([self.bos_token_id], dtype=jnp.int32), spin_tokens[:-1]],
            axis=0,
        )

        state0 = _RWKV_MODEL.default_state(params["rwkv"], _RWKV_CONFIG)
        h, _ = _rwkv_forward_features(params["rwkv"], prev_tokens, state0)   # (N, d_model)

        W = params["nqs_head"]["W"]
        b = params["nqs_head"]["b"]

        two_logits = h @ W + b                      # (N, 2)
        log_cond = jax.nn.log_softmax(two_logits.astype(jnp.float32), axis=-1)
        return log_cond / float(self.machine_pow)

    def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
        x = jnp.asarray(inputs)

        if x.ndim == 1:
            x = x[None, :]   # always batched

        B, N = x.shape

        if "params" not in self.variables:
            return jnp.zeros((B, N, self.hilbert.local_size), dtype=jnp.float32)

        params = self.variables["params"]
        return jax.vmap(self._single_conditionals, in_axes=(None, 0))(params, x)


def make_tx(cfg: Args):
    # lr_schedule = optax.exponential_decay(
    #     init_value=cfg.lr,
    #     transition_steps=cfg.transition_steps,
    #     decay_rate=cfg.decay_rate,
    #     staircase=True,
    # )

    lr_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=cfg.transition_steps,
        peak_value=1e-5,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    opt = cfg.optimizer.lower()
    if opt == "sgd":
        tx = optax.sgd(learning_rate=lr_schedule, momentum=cfg.sgd_momentum)
    elif opt == "adam":
        tx = optax.adam(learning_rate=lr_schedule)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return tx, lr_schedule


def build_hamiltonian(cfg: Args):
    hi = nk.hilbert.Spin(s=1 / 2, N=cfg.N)
    graph = nk.graph.Chain(length=cfg.N, pbc=cfg.pbc)

    name = cfg.hamiltonian.lower()
    if name == "ising":
        H_nk = nk.operator.Ising(hi, graph, h=-cfg.Gamma, J=cfg.V)
    elif name == "heisenberg":
        H_nk = nk.operator.Heisenberg(
            hilbert=hi,
            graph=graph,
            J=cfg.J,
            sign_rule=cfg.sign_rule,
        )
    else:
        raise ValueError(f"Unknown hamiltonian: {cfg.hamiltonian}")

    return hi, H_nk


def make_local_energy_fn(model: nn.Module, H):
    def local_energy(params, configs):
        configs = configs.astype(jnp.int32)
        x_primes, mels = H.get_conn_padded(configs)
        B, K, N = x_primes.shape

        logpsi_x = model.apply({"params": params}, configs)
        logpsi_xp = model.apply({"params": params}, x_primes.reshape(B * K, N)).reshape(B, K)

        ratios = jnp.exp(logpsi_xp - logpsi_x[:, None])
        e_loc = jnp.sum(mels * ratios, axis=1)
        return jnp.real(e_loc)

    return local_energy


def init_rwkv_runtime(cfg: Args):
    global _RWKV_MODEL
    global _RWKV_CONFIG
    global _RWKV_BASE_EVO_KEYS

    master_key = jax.random.key(cfg.seed)
    base_model_key = jax.random.fold_in(master_key, 0)

    rwkv, full_params, tokenizer = get_model(
        cfg.model_choice,
        rwkv_type=cfg.rwkv_type,
        verbose=True,
        dtype=cfg.dtype,
    )
    config, params, scan_map, _ = full_params

    if cfg.load_model and cfg.load_path:
        ckpt_path = _resolve_ckpt_path(cfg.load_path)
        print(f"Loading weights from: {ckpt_path}")
        ckpt: CommonInit = load(ckpt_path)
        config = _to_jnp_tree(ckpt.frozen_params, cfg.dtype)
        params = _to_jnp_tree(ckpt.params, cfg.dtype)
        scan_map = ckpt.scan_map

    _RWKV_MODEL = rwkv
    _RWKV_CONFIG = config
    _RWKV_BASE_EVO_KEYS = simple_es_tree_key(params, base_model_key, scan_map)

    return params, tokenizer


@hydra.main(version_base="1.3", config_name="config")
def main(cfg: Args) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("JAX backend:", jax.default_backend())

    rwkv_params, tokenizer = init_rwkv_runtime(cfg)
    print(f"RWKV parameter count: {_count_params(rwkv_params):,}")

    head_dtype = rwkv_params["emb"]["weight"].dtype
    d_model = rwkv_params["emb"]["weight"].shape[1]

    head_key = jax.random.fold_in(jax.random.key(cfg.seed), 123)

    params0 = {
        "rwkv": rwkv_params,
        "nqs_head": {
            "W": 0.02 * jax.random.normal(head_key, (d_model, 2), dtype=head_dtype),
            "b": jnp.zeros((2,), dtype=head_dtype),
        },
    }

    bos_token_id = cfg.bos_token_id
    if bos_token_id is None:
        bos_token_id = _encode_single_token(tokenizer, "\n", 0)

    spin_down_token_id = cfg.spin_down_token_id
    if spin_down_token_id is None:
        spin_down_token_id = _encode_single_token(tokenizer, "0", 1)

    spin_up_token_id = cfg.spin_up_token_id
    if spin_up_token_id is None:
        spin_up_token_id = _encode_single_token(tokenizer, "1", 2)

    if len({bos_token_id, spin_down_token_id, spin_up_token_id}) != 3:
        raise ValueError(
            "bos_token_id, spin_down_token_id, and spin_up_token_id must be distinct. "
            "Please pass them explicitly if tokenizer inference collides."
        )

    print(
        "Using token ids:",
        {
            "bos_token_id": bos_token_id,
            "spin_down_token_id": spin_down_token_id,
            "spin_up_token_id": spin_up_token_id,
        },
    )

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

    hi, H_nk = build_hamiltonian(cfg)

    E_gs_exact = None
    if cfg.compute_exact_diag:
        sp_h = H_nk.to_sparse()
        eig_vals = eigsh(sp_h, k=1, which="SA", return_eigenvectors=False)
        E_gs_exact = float(eig_vals[0])
        print(f"Exact ground-state energy (sparse eigsh): {E_gs_exact:.10f}")

    H = H_nk.to_jax_operator() if hasattr(H_nk, "to_jax_operator") else H_nk

    model = RWKVAutoregressiveNQS(
        hilbert=hi,
        bos_token_id=bos_token_id,
        spin_down_token_id=spin_down_token_id,
        spin_up_token_id=spin_up_token_id,
        machine_pow=cfg.machine_pow,
    )

    sampler = nk.sampler.ARDirectSampler(hilbert=hi)

    def sampler_parameters(params):
        return {"params": params} if cfg.sampler_uses_variables_dict else params

    sampler_state0 = sampler.init_state(model, sampler_parameters(params0), seed=cfg.seed)

    tx, lr_schedule = make_tx(cfg)
    state = TrainState.create(apply_fn=model.apply, params=params0, tx=tx)
    local_energy = make_local_energy_fn(model, H)

    @jax.jit
    def collect_batch(old_params, sampler_state):
        configs, sampler_state = sampler.sample(
            model,
            sampler_parameters(old_params),
            state=sampler_state,
            chain_length=cfg.batch_size,
            return_log_probabilities=False,
        )

        configs_flat = configs.reshape(-1, cfg.N)
        old_logpsi = model.apply({"params": old_params}, configs_flat)
        old_logp = cfg.machine_pow * jnp.real(old_logpsi)
        old_logp = jax.lax.stop_gradient(old_logp)

        e_loc = local_energy(old_params, configs_flat)
        e_loc = jax.lax.stop_gradient(e_loc)

        adv = -(e_loc - jnp.mean(e_loc))
        if cfg.normalize_advantage:
            adv = adv / (jnp.std(adv) + 1e-8)
        adv = jax.lax.stop_gradient(adv)

        batch = (configs_flat, old_logp, adv, e_loc)
        return batch, sampler_state

    def loss_on_batch(params, batch):
        configs_flat, old_logp, adv, e_loc = batch

        new_logpsi = model.apply({"params": params}, configs_flat)
        new_logp = cfg.machine_pow * jnp.real(new_logpsi)

        log_ratio = jnp.clip(new_logp - old_logp, -20.0, 20.0)
        ratio = jnp.exp(log_ratio)
        clipped_ratio = jnp.clip(ratio, 1.0 - cfg.ppo_clip_eps, 1.0 + cfg.ppo_clip_eps)

        surrogate_1 = ratio * adv
        surrogate_2 = clipped_ratio * adv
        policy_loss = -jnp.mean(jnp.minimum(surrogate_1, surrogate_2))

        approx_kl = jnp.mean(old_logp - new_logp)
        loss = policy_loss + cfg.kl_coef * jnp.square(approx_kl)

        metrics = {
            "loss": loss,
            "policy_loss": policy_loss,
            "E_mean": jnp.mean(e_loc),
            "E_std": jnp.std(e_loc),
            "ratio_mean": jnp.mean(ratio),
            "clip_frac": jnp.mean(jnp.abs(ratio - 1.0) > cfg.ppo_clip_eps),
            "approx_kl": approx_kl,
        }

        if E_gs_exact is not None:
            rel_error_exact = jnp.abs((metrics["E_mean"] - E_gs_exact) / E_gs_exact)
            metrics["E_exact"] = jnp.asarray(E_gs_exact)
            metrics["rel_error_exact"] = rel_error_exact

        return loss, metrics

    @jax.jit
    def eval_chunk_sums(params, configs):
        e_loc = local_energy(params, configs)
        e_loc = jnp.real(e_loc)
        return jnp.sum(e_loc), jnp.sum(e_loc**2), e_loc.shape[0]

    def evaluate(state, sampler_state):
        total_sum = 0.0
        total_sum_sq = 0.0
        total_n = 0
        n_remaining = cfg.eval_n_samples
        eval_state = sampler_state

        while n_remaining > 0:
            configs, eval_state = sampler.sample(
                model,
                sampler_parameters(state.params),
                state=eval_state,
                chain_length=cfg.eval_batch_size,
                return_log_probabilities=False,
            )
            configs = jnp.asarray(configs).reshape(-1, cfg.N)

            cur_n = min(int(configs.shape[0]), n_remaining)
            configs = configs[:cur_n]

            chunk_sum, chunk_sum_sq, chunk_n = eval_chunk_sums(state.params, configs)
            total_sum += float(chunk_sum)
            total_sum_sq += float(chunk_sum_sq)
            total_n += int(chunk_n)
            n_remaining -= int(chunk_n)

        energy = total_sum / total_n
        variance = max(total_sum_sq / total_n - energy**2, 0.0)
        denom = max(energy * energy, 1e-12)
        v_score = float(cfg.N * variance / denom)

        out = {
            "eval_E_mean": energy,
            "eval_E_var": variance,
            "eval_V_score": v_score,
            "eval_n_samples_used": total_n,
        }
        if E_gs_exact is not None:
            out["eval_E_exact"] = float(E_gs_exact)
            out["eval_rel_error_exact"] = float(abs((energy - E_gs_exact) / E_gs_exact))
        return out, eval_state

    @jax.jit
    def train_iter(state, sampler_state):
        old_params = state.params
        batch, sampler_state = collect_batch(old_params, sampler_state)

        def epoch_step(state, _):
            (loss, metrics), grads = jax.value_and_grad(
                loss_on_batch,
                argnums=0,
                has_aux=True,
            )(state.params, batch)
            grad_norm = optax.global_norm(grads)
            state = state.apply_gradients(grads=grads)
            metrics = {
                **metrics,
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
        return state, sampler_state, metrics

    sampler_state = sampler_state0

    for it in range(cfg.n_iter):
        state, sampler_state, metrics = train_iter(state, sampler_state)

        if it % cfg.log_every == 0:
            current_lr = float(lr_schedule(state.step))
            msg = (
                f"it={it:06d}  loss={float(metrics['loss']): .6f}  "
                f"policy={float(metrics['policy_loss']): .6f}  "
                f"E_batch={float(metrics['E_mean']): .6f} ± {float(metrics['E_std']): .6f}  "
                f"ratio={float(metrics['ratio_mean']): .6f}  "
                f"clip={float(metrics['clip_frac']): .3f}  "
                f"kl={float(metrics['approx_kl']): .3e}  "
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
                    "policy_loss": float(metrics["policy_loss"]),
                    "E_mean": float(metrics["E_mean"]),
                    "E_std": float(metrics["E_std"]),
                    "grad_norm": float(metrics["grad_norm"]),
                    "param_norm": float(metrics["param_norm"]),
                    "ratio_mean": float(metrics["ratio_mean"]),
                    "clip_frac": float(metrics["clip_frac"]),
                    "approx_kl": float(metrics["approx_kl"]),
                }
                if E_gs_exact is not None:
                    log_dict["E_exact"] = float(metrics["E_exact"])
                    log_dict["rel_error_exact"] = float(metrics["rel_error_exact"])
                wandb.log(log_dict, step=it)

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_metrics, _ = evaluate(state, sampler_state)
            eval_msg = (
                f"[eval] it={it:06d}  "
                f"E={eval_metrics['eval_E_mean']:.8f}  "
                f"Var(E)={eval_metrics['eval_E_var']:.8e}  "
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

    final_eval, _ = evaluate(state, sampler_state)
    print("Final eval:", final_eval)

    if cfg.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
