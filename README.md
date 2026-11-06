# HyperscaleNQS

This repository contains the code used for the experiments in:

> Juan Agustín Duque, Sergio García Heredia, Vinicius Hernandes, Eliška Greplová, Thomas Spriggs, Aaron Courville, Anna Dawid. **"One More Time: Revisiting Neural Quantum States from a Reinforcement Learning Perspective."** [arXiv:2607.02292](https://arxiv.org/abs/2607.02292)

which introduces **Proximal Wavefunction Optimization (PWO)**, a clipped, PPO-style dual-channel objective that optimizes the amplitude (real) and phase (imaginary) channels of `log ψ` separately.

The default wavefunction ansatz is `ComplexRecurrentAR`, an autoregressive network with a shared GRU trunk feeding separate amplitude and phase towers, sampled exactly via NetKet's `ARDirectSampler`.

## Repository layout

```
src/
├── adam.py, minsr.py, pwo.py   # 1D chain algorithms (Ising / Heisenberg / J1-J2)
├── 2d/                         # 2D square-lattice J1-J2 (PWO + SOTA eval, ViT-RoPE ansatz)
├── scaling/                    # Variants of adam/minsr/pwo for scaling experiments
└── llm/                        # RWKV-7 backbone integration as an NQS ansatz for PWO

sweep/
├── run_adam.slurm, run_minsr.slurm, run_pwo.slurm, run_pwo_llm.slurm, run_pwo_2d_*.slurm
├── chains/run_chains.sh        # Launches all algorithms across {ising, heisenberg, j1j2}
├── scaling/                    # Scaling sweeps (model size, system size, sample count)
├── submit_minsr_grid.sh, submit_pwo_grid.sh, sweep_pwo.py
└── ...

analysis/
└── count_parameters.py         # Model parameter counting utility

submit_pwo_sweep.py              # Randomized hyperparameter sweep submitter (SLURM)
tutorial.ipynb                   # End-to-end walkthrough of PWO on J1-J2 (N=12)
```

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

Key dependencies: `jax[cuda12]`, `netket[cuda]` (installed from git), `flax`, `optax`, `hydra-core`, `wandb`, `torch`/`transformers` (for the RWKV-7 LLM backbone).

## Usage

All entry points are Hydra scripts configured via `@dataclass` args, so any field can be overridden on the command line.

```bash
# VMC with Adam
python src/chains/adam.py hamiltonian=j1j2 seed=42

# minSR / SPRING
python src/chains/minsr.py hamiltonian=j1j2 seed=42 n_samples=1024

# PWO
python src/chains/pwo.py hamiltonian=j1j2 seed=42 n_samples=1024
```

Common Hydra options: `hamiltonian` (`ising`/`heisenberg`/`j1j2`), `N`, `J1`, `J2`, `n_samples`, learning-rate schedule params, and `wandb_project`/`wandb_mode` for logging. The chain entrypoints use periodic boundaries for Ising; `pbc` applies to the Heisenberg chains.

### SLURM sweeps

```bash
# Run scaling sweeps for Adam, minSR, and PWO on the J₁-J₂ chain
bash sweep/scaling/run_scaling_model.sh

# Randomized hyperparameter sweep for PWO
python submit_pwo_sweep.py submit --hamiltonian=j1j2 --n_runs=20
```

### Tutorial

[tutorial.ipynb](tutorial.ipynb) walks through PWO on the J₁-J₂ chain at the Majumdar–Ghosh point (`J2/J1=0.5`), comparing against exact-diagonalization ground truth.

## Outputs

- `outputs/`, `checkpoints/` — Hydra run outputs and model checkpoints.
- `wandb/` — Weights & Biases run logs.
- `logs/` — SLURM stdout/stderr per job/array-task.
