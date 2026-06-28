from __future__ import annotations

import math
import random
import shlex
import subprocess

import fire


SLURM = "sweep/run_qps_2d.slurm"
MODULE = "src.2d.qps_j1j2_square_kv_cache_with_prefix_magnetization_features"

SEED = 42
CONN_CHUNK_SIZE = 8
N_SAMPLES = 2048
WANDB_MODE = "online"

FIXED_OVERRIDES = {
    "eval_n_samples": 2028,
    "eval_batch_size": 2048,
    "eval_every": 200,
    "transition_steps": 5000,
    "debug_local_energy_progress": False,
    "validate_cached_sampler": False,
}


def log_uniform(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def run(n: int = 10, sweep_seed: int = 0, dry: bool = True, submit: bool = True):
    rng = random.Random(sweep_seed)

    for i in range(n):
        ppo_epochs = rng.choices(
            [2, 3, 4],
            weights=[5, 3, 1],
        )[0]

        cfg = dict(FIXED_OVERRIDES)
        cfg.update({
            "peak_lr": log_uniform(rng, 3e-5, 3e-4),
            "ppo_clip_eps": log_uniform(rng, 4e-4, 3e-3),

            "phase_coef": rng.choices(
                [1.0],
                weights=[1],
            )[0],
            "phase_delta_clip": rng.choices(
                [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
                weights=[1, 1, 1, 1, 1, 1],
            )[0],
            "normalize_imag_advantage": rng.choices(
                [True],
                weights=[1],
            )[0],
            "phase_delta_l2_coef": rng.choices(
                [0.0],
                weights=[1],
            )[0],

            "wandb_run_name": f"pwo_sweep_{i:02d}",
        })

        extra_overrides = " ".join(f"{k}={fmt(v)}" for k, v in cfg.items())

        if submit:
            # Your Slurm script expects:
            # $1 seed, $2 conn_chunk, $3 n_samples, $4 ppo_epochs, $5 wandb_mode.
            #
            # We pass "online extra=overrides..." as the 5th argument.
            # The Slurm script expands it into:
            #   wandb_mode=online extra=overrides...
            wandb_arg = f"{WANDB_MODE} {extra_overrides}"

            cmd = [
                "sbatch",
                "--job-name",
                f"pwo_sweep_{i:02d}",
                SLURM,
                str(SEED),
                str(CONN_CHUNK_SIZE),
                str(N_SAMPLES),
                str(ppo_epochs),
                wandb_arg,
            ]
        else:
            cmd = [
                "python",
                "-m",
                MODULE,
                f"seed={SEED}",
                f"local_energy_conn_chunk_size={CONN_CHUNK_SIZE}",
                f"n_samples={N_SAMPLES}",
                f"ppo_epochs={ppo_epochs}",
                f"wandb_mode={WANDB_MODE}",
                *[f"{k}={fmt(v)}" for k, v in cfg.items()],
            ]

        print(f"\n# run {i}")
        print(" ".join(shlex.quote(x) for x in cmd))

        if not dry:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    fire.Fire(run)