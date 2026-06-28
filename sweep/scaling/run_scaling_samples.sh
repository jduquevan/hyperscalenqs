#!/bin/bash
# Scaling experiment: number of training samples
# Trains the Medium model on the frustrated Heisenberg J1-J2 chain
# (N=12, J1=1, J2=0.5, PBC) with Adam, minSR, and PWO,
# sweeping n_samples over {512, 1024, 2048}.
# Each combination runs as a 10-seed SLURM array (--array=0-9).
#
# Usage: bash sweep/run_scaling_samples.sh

for S in 512 1024 2048; do
    export N_SAMPLES=$S
    export CHAIN_N=12
    export N_GRU_LAYERS=3
    export RNN_HIDDEN=256
    export HEAD_HIDDEN=256
    export TRANSITION_STEPS=40000
    export WANDB_PROJECT=scaling-samples
    export RUN_TAG=n${S}

    sbatch sweep/scaling/run_adam.slurm j1j2
    sbatch sweep/scaling/run_minsr.slurm j1j2 minsr
    sbatch sweep/scaling/run_qps.slurm j1j2
done
