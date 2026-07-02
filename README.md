# HyperscaleNQS

The repo compares three variational optimization methods on frustrated spin chains and lattices:

- **VMC (Adam)** — plain stochastic-gradient VMC with a cosine one-cycle learning-rate schedule.
- **minSR / SPRING** — stochastic reconfiguration via the neural-tangent-kernel (NTK) trick, avoiding materialization of the full Jacobian for large networks.
- **PWO (Proximal Wavefunction Optimization)** — a clipped, PPO-style dual-channel objective that optimizes the amplitude (real) and phase (imaginary) channels of `log ψ` separately.

The default wavefunction ansatz is `ComplexRecurrentAR`, an autoregressive network with a shared GRU trunk feeding separate amplitude and phase towers, sampled exactly via NetKet's `ARDirectSampler`.

## Repository layout

```
src/
├── adam.py, minsr.py, qps.py   # 1D chain algorithms (Ising / Heisenberg / J1-J2)
├── 2d/                         # 2D square-lattice J1-J2 (PWO + SOTA eval, ViT-RoPE ansatz)
├── scaling/                    # Variants of adam/minsr/qps for scaling experiments
└── llm/                        # RWKV-7 backbone integration as an NQS ansatz for PWO

sweep/
├── run_adam.slurm, run_minsr.slurm, run_qps.slurm, run_qps_llm.slurm, run_qps_2d_*.slurm
├── chains/run_chains.sh        # Launches all algorithms across {ising, heisenberg, j1j2}
├── scaling/                    # Scaling sweeps (model size, system size, sample count)
├── submit_minsr_grid.sh, submit_qps_grid.sh, sweep_pwo.py
└── ...

analysis/
└── count_parameters.py         # Model parameter counting utility

submit_qps_sweep.py              # Randomized hyperparameter sweep submitter (SLURM)
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
python src/adam.py hamiltonian=j1j2 seed=42

# minSR / SPRING
python src/minsr.py hamiltonian=j1j2 seed=42

# PWO
python src/qps.py hamiltonian=j1j2 seed=42
```

Common Hydra options: `hamiltonian` (`ising`/`heisenberg`/`j1j2`), `N`, `J1`, `J2`, `pbc`, `n_samples`, learning-rate schedule params, and `wandb_project`/`wandb_mode` for logging.

### SLURM sweeps

```bash
# Run scaling sweeps for Adam, minSR, and PWO on the J₁-J₂ chain
bash sweep/scaling/run_scaling_model.sh

# Randomized hyperparameter sweep for PWO
python submit_qps_sweep.py --slurm_script=sweep/run_qps.slurm --hamiltonian=j1j2 --n_runs=20
```

### Tutorial

[tutorial.ipynb](tutorial.ipynb) walks through PWO on the J₁-J₂ chain at the Majumdar–Ghosh point (`J2/J1=0.5`), comparing against exact-diagonalization ground truth.

## Outputs

- `outputs/`, `checkpoints/` — Hydra run outputs and model checkpoints.
- `wandb/` — Weights & Biases run logs.
- `logs/` — SLURM stdout/stderr per job/array-task.
