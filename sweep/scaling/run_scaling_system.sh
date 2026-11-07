#!/bin/bash
# Scaling experiment: system size
# Trains the Medium model on the frustrated Heisenberg J1-J2 chain
# (J1=1, J2=0.5, PBC) with Adam, minSR, and PWO,
# sweeping the chain length N over {8, 12, 16} with n_samples=1024.
# Each combination runs as a 10-seed SLURM array (--array=0-9).
#
# Usage: bash sweep/run_scaling_system.sh

for N in 8 12 16; do
    export CHAIN_N=$N
    export N_SAMPLES=1024
    export N_GRU_LAYERS=3
    export RNN_HIDDEN=256
    export HEAD_HIDDEN=256
    export WANDB_PROJECT=scaling-system-corrected
    export RUN_TAG=N${N}

    TRANSITION_STEPS=10000 sbatch sweep/run_adam.slurm j1j2
    sbatch sweep/run_minsr.slurm j1j2 minsr
    PPO_EPOCHS=4 TRANSITION_STEPS=40000 sbatch sweep/run_pwo.slurm j1j2
done
