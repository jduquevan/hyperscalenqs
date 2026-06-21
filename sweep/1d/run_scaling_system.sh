#!/bin/bash
# Scaling experiment: system size
# Trains the Medium model on the frustrated Heisenberg J1-J2 chain
# (J1=1, J2=0.5, PBC) with Adam, minSR, and PWO,
# sweeping the chain length N over {8, 16, 32} with n_samples=2048.
# Each combination runs as a 10-seed SLURM array (--array=0-9).
#
# Usage: bash sweep/run_scaling_system.sh

for N in 8 16 24; do
    export CHAIN_N=$N
    export N_SAMPLES=2048
    export N_GRU_LAYERS=3
    export RNN_HIDDEN=256
    export HEAD_HIDDEN=256
    export TRANSITION_STEPS=40000
    export WANDB_PROJECT=scaling-system
    export RUN_TAG=N${N}

    sbatch sweep/1d/run_adam.slurm j1j2
    sbatch sweep/1d/run_minsr.slurm j1j2 minsr
    sbatch sweep/1d/run_qps.slurm j1j2
done
