#!/usr/bin/env python3
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import fire


def _parse_list(values, cast_fn):
    if isinstance(values, (list, tuple)):
        return [cast_fn(v) for v in values]
    if isinstance(values, str):
        values = values.strip()
        if not values:
            return []
        return [cast_fn(v.strip()) for v in values.split(",")]
    return [cast_fn(values)]


def _fmt(x: float) -> str:
    return f"{x:.8g}"


def submit(
    n_runs: int = 20,
    slurm_script: str = "sweep/run_qps.slurm",
    hamiltonian: str = "j1j2",
    wandb_tag: str = "random_sweep",
    run_prefix: str = "qps",
    start_seed: int = 42,
    sweep_seed: int = 0,
    lrs: str = "3e-5,1e-4,3e-4",
    ppo_epochs: str = "2,4,8",
    ppo_clip_eps: str = "1e-4,3e-4,1e-3,3e-3,1e-2",
    phase_delta_clip: str = "0.05,0.1,0.2,0.3,0.5",
    extra_args: str = "",
    dry_run: bool = False,
):
    script_path = Path(slurm_script)
    if not script_path.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")

    lr_values = _parse_list(lrs, float)
    ppo_epoch_values = _parse_list(ppo_epochs, int)
    clip_values = _parse_list(ppo_clip_eps, float)
    phase_delta_values = _parse_list(phase_delta_clip, float)

    if not lr_values:
        raise ValueError("Empty lrs list.")
    if not ppo_epoch_values:
        raise ValueError("Empty ppo_epochs list.")
    if not clip_values:
        raise ValueError("Empty ppo_clip_eps list.")
    if not phase_delta_values:
        raise ValueError("Empty phase_delta_clip list.")

    shared_extra_args = extra_args.split() if extra_args.strip() else []
    rng = random.Random(sweep_seed)

    for i in range(n_runs):
        seed = start_seed + i

        lr = rng.choice(lr_values)
        epochs = rng.choice(ppo_epoch_values)
        clip = rng.choice(clip_values)
        dphi = rng.choice(phase_delta_values)
        transition_steps = 10000 * epochs

        run_name = (
            f"{run_prefix}_{hamiltonian}_s{seed}"
            f"_lr{_fmt(lr)}"
            f"_ep{epochs}"
            f"_clip{_fmt(clip)}"
            f"_dphi{_fmt(dphi)}"
        )

        overrides = [
            f"lr={_fmt(lr)}",
            f"peak_lr={_fmt(lr)}",
            f"ppo_epochs={epochs}",
            f"ppo_clip_eps={_fmt(clip)}",
            f"phase_delta_clip={_fmt(dphi)}",
            f"transition_steps={transition_steps}",
            *shared_extra_args,
        ]

        cmd = [
            "sbatch",
            str(script_path),
            str(seed),
            run_name,
            hamiltonian,
            wandb_tag,
            *overrides,
        ]

        print(f"[{i+1}/{n_runs}] {' '.join(cmd)}")

        if not dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    fire.Fire({"submit": submit})