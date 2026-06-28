from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import hydra
import jax

jax.config.update("jax_enable_x64", True)

import flax

flax.config.update("flax_use_orbax_checkpointing", False)

import jax.numpy as jnp
import netket as nk
import wandb
from flax.training import checkpoints
from huggingface_hub import hf_hub_download, snapshot_download
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

try:
    from transformers import FlaxAutoModel
except ImportError as exc:
    raise ImportError(
        "This Hugging Face checkpoint is Flax-only and requires Transformers "
        "with Flax support. With uv, install a compatible version with:\n\n"
        "  uv pip install 'transformers==4.48.0' "
        "'huggingface_hub>=0.24' 'safetensors>=0.4.3'\n\n"
        "Then verify with:\n\n"
        "  uv run python -c \"from transformers import FlaxAutoModel, "
        "FlaxPreTrainedModel; import transformers; print(transformers.__version__)\""
    ) from exc


@dataclass
class Args:
    seed: int = 0
    jax_platform_name: str = "gpu"

    # HF model.
    # Useful revisions:
    #   main
    #   symm_t
    #   symm_trxy_ising
    repo_id: str = "nqs-models/j1j2_square_10x10_05"
    revision: str = "main"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    force_download: bool = False

    download_snapshot: bool = True

    L: int = 10
    pbc: bool = True
    J1: float = 1.0
    J2: float = 0.5
    total_sz: Optional[int] = 0

    n_samples: int = 2048
    n_chains: int = 2048
    d_max: int = 2
    sweep_size: int = 0  # 0 means L * L
    n_discard_per_chain: int = 0
    thermalization_sweeps: int = 0
    n_eval: int = 10
    chunk_size: int = 2048

    use_hf_spins: bool = True

    compute_local_extras: bool = True

    # Reference energies from the HF model card, in the convention E / (4N).
    reference_energy_per_site_conventional: Optional[float] = None

    clear_transformers_dynamic_cache_on_retry: bool = True

    wandb_project: str = "hyperscalenqs-j1j2-square"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    wandb_directory: Optional[str] = "."
    wandb_tags: Optional[str] = "sota,hf,j1j2_square,vit_nqs"

    # Avoid accidental resume from inherited Slurm/shell W&B env vars.
    wandb_id: Optional[str] = None
    wandb_resume: str = "never"
    ignore_wandb_resume_env: bool = True

    output_directory: Optional[str] = "."
    output_jsonl: str = "hf_sota_eval_metrics.jsonl"


ConfigStore.instance().store(name="config", node=Args)


REFERENCE_PER_SITE_CONVENTIONAL = {
    "main": -0.497505103,
    "symm_t": -0.49760546,
    "symm_trxy_ising": -0.497676335,
}


def build_square_j1j2(cfg: Args):
    n_sites = cfg.L * cfg.L

    if cfg.total_sz is None:
        hilbert = nk.hilbert.Spin(s=1 / 2, N=n_sites)
    else:
        hilbert = nk.hilbert.Spin(s=1 / 2, N=n_sites, total_sz=cfg.total_sz)

    graph = nk.graph.Hypercube(
        length=cfg.L,
        n_dim=2,
        pbc=cfg.pbc,
        max_neighbor_order=2,
    )

    hamiltonian = nk.operator.Heisenberg(
        hilbert=hilbert,
        graph=graph,
        J=[cfg.J1, cfg.J2],
        sign_rule=[False, False],
    )

    return hilbert, graph, hamiltonian


def maybe_download_snapshot(cfg: Args) -> None:
    """Optionally prefetch the HF repo into the local cache.

    Important: this function intentionally does not return the local snapshot
    path. Passing a local snapshot path to FlaxAutoModel can trigger a
    Transformers dynamic-module cache bug for this repo, where relative imports
    such as attentions.py are not copied correctly.

    We prefetch if requested, but load the model by repo_id.
    """
    if not cfg.download_snapshot:
        return

    allow_patterns = [
        "*.json",
        "*.py",
        "*.safetensors",
        "spins",
        "README.md",
    ]

    snapshot_download(
        repo_id=cfg.repo_id,
        revision=cfg.revision,
        cache_dir=cfg.cache_dir,
        local_files_only=cfg.local_files_only,
        force_download=cfg.force_download,
        allow_patterns=allow_patterns,
    )


def transformers_dynamic_module_cache_root() -> Path:
    """Return the Transformers dynamic-module cache directory."""
    try:
        from transformers.utils import HF_MODULES_CACHE

        modules_root = Path(HF_MODULES_CACHE)
    except Exception:
        hf_home = Path(
            os.environ.get(
                "HF_HOME",
                str(Path.home() / ".cache" / "huggingface"),
            )
        )
        modules_root = Path(os.environ.get("HF_MODULES_CACHE", str(hf_home / "modules")))

    return modules_root / "transformers_modules"


def clear_transformers_dynamic_module_cache() -> None:
    """Clear only the dynamic Python module cache used by trust_remote_code.

    This does not delete model weights from the HF hub cache.
    """
    root = transformers_dynamic_module_cache_root()
    if root.exists():
        print(f"[warn] Clearing Transformers dynamic-module cache: {root}")
        shutil.rmtree(root, ignore_errors=True)


def _from_pretrained(cfg: Args):
    return FlaxAutoModel.from_pretrained(
        cfg.repo_id,
        revision=cfg.revision,
        cache_dir=cfg.cache_dir,
        local_files_only=cfg.local_files_only,
        force_download=cfg.force_download,
        trust_remote_code=True,
    )


def load_hf_model(cfg: Args):
    """Load the HF model by repo id, with one cache-clearing retry if needed."""
    try:
        return _from_pretrained(cfg)
    except FileNotFoundError as exc:
        msg = str(exc)
        is_dynamic_module_error = (
            "transformers_modules" in msg
            and (
                "attentions.py" in msg
                or "transformer.py" in msg
                or "vitnqs_model.py" in msg
            )
        )

        if not (cfg.clear_transformers_dynamic_cache_on_retry and is_dynamic_module_error):
            raise

        print("[warn] HF dynamic-module cache appears stale or incomplete.")
        print(f"[warn] Original error: {exc}")
        clear_transformers_dynamic_module_cache()
        print("[warn] Retrying HF model load by repo id.")
        return _from_pretrained(cfg)


def restore_flax_checkpoint_file(path: str, prefix: str = "spins"):
    """Restore the HF 'spins' file robustly across Flax versions.

    The model card uses:
        checkpoints.restore_checkpoint(ckpt_dir=path, prefix='spins', target=None)

    Some Flax versions expect ckpt_dir to be the containing directory instead,
    so we try both forms.
    """
    attempts = [
        (path, prefix),
        (str(Path(path).parent), Path(path).name),
    ]

    last_error: Optional[BaseException] = None
    for ckpt_dir, ckpt_prefix in attempts:
        try:
            return checkpoints.restore_checkpoint(
                ckpt_dir=ckpt_dir,
                prefix=ckpt_prefix,
                target=None,
            )
        except BaseException as exc:
            last_error = exc

    raise RuntimeError(f"Could not restore spins checkpoint at {path}") from last_error


def maybe_load_hf_spins(cfg: Args):
    if not cfg.use_hf_spins:
        return None

    path = hf_hub_download(
        repo_id=cfg.repo_id,
        filename="spins",
        revision=cfg.revision,
        cache_dir=cfg.cache_dir,
        local_files_only=cfg.local_files_only,
        force_download=cfg.force_download,
    )

    spins = restore_flax_checkpoint_file(path, prefix="spins")
    return jnp.asarray(spins, dtype=jnp.int8)


def install_sampler_sigma(vstate: nk.vqs.MCState, spins, n_sites: int) -> bool:
    """Overwrite the MCMC sampler state with HF-provided thermalized samples."""
    if spins is None:
        return False

    sigma0 = getattr(vstate.sampler_state, "σ", None)
    if sigma0 is None:
        print("[warn] sampler_state has no σ field; skipping HF spins restore.")
        return False

    target_shape = tuple(sigma0.shape)
    spins = jnp.asarray(spins, dtype=sigma0.dtype).reshape((-1, n_sites))

    if len(target_shape) == 2 and target_shape[1] == n_sites:
        target_n = target_shape[0]
        if spins.shape[0] < target_n:
            reps = math.ceil(target_n / spins.shape[0])
            spins = jnp.tile(spins, (reps, 1))
        spins = spins[:target_n]
    else:
        target_size = math.prod(target_shape)
        flat = spins.reshape(-1)
        if flat.size < target_size:
            reps = math.ceil(target_size / flat.size)
            flat = jnp.tile(flat, (reps,))
        spins = flat[:target_size].reshape(target_shape)

    vstate.sampler_state = vstate.sampler_state.replace(**{"σ": spins})
    return True


def finite_float(x) -> float:
    return float(jax.device_get(jnp.asarray(x)))


def safe_stats_attr(stats, name: str, default=float("nan")) -> float:
    value = getattr(stats, name, default)
    try:
        return finite_float(jnp.real(value))
    except Exception:
        return float(default)


def tree_num_params(tree) -> int:
    try:
        return int(nk.jax.tree_size(tree))
    except Exception:
        return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))


def local_estimator_metrics(vstate: nk.vqs.MCState, hamiltonian):
    """Metrics matching the explicit local-energy aggregation in your script."""
    loc = vstate.local_estimators(hamiltonian)
    loc = jnp.asarray(loc).reshape(-1)

    e_real = jnp.real(loc)
    e_imag = jnp.imag(loc)

    energy_real = jnp.mean(e_real)
    variance_real = jnp.maximum(jnp.mean(e_real**2) - energy_real**2, 0.0)

    return {
        "eval_E_mean_real": finite_float(energy_real),
        "eval_E_mean_imag": finite_float(jnp.mean(e_imag)),
        "eval_E_var_real": finite_float(variance_real),
        "eval_E_std_real": finite_float(jnp.sqrt(variance_real)),
        "eval_abs_E_imag_mean": finite_float(jnp.mean(jnp.abs(e_imag))),
        "eval_n_samples_used": int(loc.shape[0]),
    }


def evaluate_once(cfg: Args, vstate: nk.vqs.MCState, hamiltonian, n_sites: int, step: int):
    t0 = time.perf_counter()
    stats = vstate.expect(hamiltonian)
    jax.block_until_ready(stats.mean)
    elapsed = time.perf_counter() - t0

    mean = stats.mean

    metrics = {
        "iter": step,
        "sota_revision": cfg.revision,
        "eval_wall_time_s": elapsed,
        "eval_E_mean_real": finite_float(jnp.real(mean)),
        "eval_E_mean_imag": finite_float(jnp.imag(mean)),
        "eval_E_var_real": safe_stats_attr(stats, "variance"),
        "eval_E_error_of_mean": safe_stats_attr(stats, "error_of_mean"),
        "eval_R_hat": safe_stats_attr(stats, "R_hat"),
        "eval_tau_corr": safe_stats_attr(stats, "tau_corr"),
        "eval_n_samples_used": int(vstate.n_samples),
    }

    if cfg.compute_local_extras:
        try:
            # This overwrites mean/variance with the same explicit local-energy
            # aggregation style as your current script.
            metrics.update(local_estimator_metrics(vstate, hamiltonian))
        except Exception as exc:
            print(f"[warn] Could not compute local-estimator extras: {exc}")
            metrics.setdefault("eval_abs_E_imag_mean", float("nan"))
            metrics.setdefault(
                "eval_E_std_real",
                math.sqrt(max(metrics["eval_E_var_real"], 0.0)),
            )
    else:
        metrics["eval_E_std_real"] = math.sqrt(max(metrics["eval_E_var_real"], 0.0))
        metrics["eval_abs_E_imag_mean"] = float("nan")

    e_total = metrics["eval_E_mean_real"]
    var_real = max(metrics["eval_E_var_real"], 0.0)
    denom = max(e_total * e_total, 1e-12)

    # The HF model card reports E / (4N). Log both.
    metrics.update(
        {
            "eval_E_total_raw": e_total,
            "eval_E_per_site_raw": e_total / n_sites,
            "eval_E_per_site_conventional": e_total / (4.0 * n_sites),
            "eval_V_score": n_sites * var_real / denom,
        }
    )

    ref_conv = cfg.reference_energy_per_site_conventional
    if ref_conv is None:
        ref_conv = REFERENCE_PER_SITE_CONVENTIONAL.get(cfg.revision)

    if ref_conv is not None:
        ref_total = float(ref_conv) * 4.0 * n_sites
        metrics.update(
            {
                "eval_reference_E_per_site_conventional": float(ref_conv),
                "eval_reference_E_total_raw": ref_total,
                "eval_rel_error_reference": abs((e_total - ref_total) / ref_total),
            }
        )

    return metrics


def append_jsonl(path: Path, metrics: dict):
    serializable = {}
    for k, v in metrics.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            serializable[k] = v
        else:
            try:
                serializable[k] = float(v)
            except Exception:
                serializable[k] = str(v)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(serializable, sort_keys=True) + "\n")


def setup_wandb(cfg: Args):
    if cfg.wandb_mode == "disabled":
        return

    if cfg.ignore_wandb_resume_env and cfg.wandb_id is None:
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)

    Path(cfg.wandb_directory or ".").mkdir(parents=True, exist_ok=True)

    tags = None
    if cfg.wandb_tags is not None:
        tags = [tag.strip() for tag in cfg.wandb_tags.split(",") if tag.strip()]

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_run_name or f"hf_sota_{cfg.revision}",
        id=cfg.wandb_id,
        resume=cfg.wandb_resume,
        tags=tags,
        config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
        dir=cfg.wandb_directory,
        mode=cfg.wandb_mode,
    )


@hydra.main(version_base="1.3", config_name="config")
def main(cfg: Args) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("JAX backend:", jax.default_backend())

    n_sites = cfg.L * cfg.L

    output_dir = Path(cfg.output_directory or ".")
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / cfg.output_jsonl
    if jsonl_path.exists():
        jsonl_path.unlink()

    setup_wandb(cfg)

    maybe_download_snapshot(cfg)

    print(f"Loading HF model from repo id: {cfg.repo_id}@{cfg.revision}")
    wf = load_hf_model(cfg)

    n_params = tree_num_params(wf.params)
    print(f"HF model {cfg.repo_id}@{cfg.revision}: parameters={n_params:,}")

    hilbert, graph, hamiltonian = build_square_j1j2(cfg)
    hamiltonian = (
        hamiltonian.to_jax_operator()
        if hasattr(hamiltonian, "to_jax_operator")
        else hamiltonian
    )

    sampler = nk.sampler.MetropolisExchange(
        hilbert=hilbert,
        graph=graph,
        d_max=cfg.d_max,
        n_chains=cfg.n_chains,
        sweep_size=cfg.sweep_size if cfg.sweep_size > 0 else n_sites,
    )

    key = jax.random.PRNGKey(cfg.seed)
    key, sampler_key = jax.random.split(key)

    vstate = nk.vqs.MCState(
        sampler=sampler,
        apply_fun=wf.__call__,
        sampler_seed=sampler_key,
        n_samples=cfg.n_samples,
        n_discard_per_chain=cfg.n_discard_per_chain,
        variables=wf.params,
        chunk_size=cfg.chunk_size,
    )

    restored = False
    if cfg.use_hf_spins:
        try:
            spins = maybe_load_hf_spins(cfg)
            restored = install_sampler_sigma(vstate, spins, n_sites)
        except Exception as exc:
            print(f"[warn] Could not load/install HF thermalized spins: {exc}")
            print("[warn] Continuing from NetKet sampler initialization.")

    print(f"HF thermalized spins restored: {restored}")

    for _ in range(cfg.thermalization_sweeps):
        vstate.sample()

    final_metrics = None

    for it in range(cfg.n_eval):
        metrics = evaluate_once(cfg, vstate, hamiltonian, n_sites, step=it)
        final_metrics = metrics

        msg = (
            f"[hf-sota] it={it:04d} "
            f"E_total={metrics['eval_E_total_raw']:.10f} "
            f"E/N={metrics['eval_E_per_site_raw']:.10f} "
            f"E/(4N)={metrics['eval_E_per_site_conventional']:.10f} "
            f"E_imag={metrics['eval_E_mean_imag']:.3e} "
            f"Var_real={metrics['eval_E_var_real']:.3e} "
            f"V_score={metrics['eval_V_score']:.3e} "
            f"time={metrics['eval_wall_time_s']:.2f}s"
        )

        if "eval_rel_error_reference" in metrics:
            msg += f" rel_ref={metrics['eval_rel_error_reference']:.3e}"

        print(msg)

        append_jsonl(jsonl_path, metrics)

        if cfg.wandb_mode != "disabled":
            wandb.log(metrics, step=it)

        if it != cfg.n_eval - 1:
            vstate.sample()

    if final_metrics is not None:
        print(
            "Final HF SOTA sampled energy: "
            f"{final_metrics['eval_E_total_raw']:.10f} + "
            f"{final_metrics['eval_E_mean_imag']:.3e}j  "
            f"per site raw E/N: {final_metrics['eval_E_per_site_raw']:.10f}  "
            f"per site conventional E/(4N): "
            f"{final_metrics['eval_E_per_site_conventional']:.10f}"
        )

        if cfg.wandb_mode != "disabled":
            wandb.log(
                {
                    "final_E_real": final_metrics["eval_E_total_raw"],
                    "final_E_imag": final_metrics["eval_E_mean_imag"],
                    "final_E_per_site": final_metrics["eval_E_per_site_raw"],
                    "final_E_per_site_conventional": final_metrics[
                        "eval_E_per_site_conventional"
                    ],
                    "final_V_score": final_metrics["eval_V_score"],
                }
            )
            wandb.finish()

    print(f"Wrote metrics to {jsonl_path}")


if __name__ == "__main__":
    main()