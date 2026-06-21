#!/bin/bash
# Scaling experiment: model size
# Trains Tiny / Small / Medium on the frustrated Heisenberg J1-J2 chain
# (N=12, J1=1, J2=0.5, PBC) with Adam, minSR, and PWO.
# Each combination runs as a 10-seed SLURM array (--array=0-9).
#
# Model sizes:
#   Tiny   | n_gru_layers=1 | rnn_hidden=64  | head_hidden=64
#   Small  | n_gru_layers=2 | rnn_hidden=128 | head_hidden=128
#   Medium | n_gru_layers=3 | rnn_hidden=256 | head_hidden=256
#
# Usage: bash sweep/run_scaling_model.sh

for SIZE_SPEC in "tiny:1:64:64" "small:2:128:128" "medium:3:256:256"; do
    SIZE=$(echo "$SIZE_SPEC"   | cut -d: -f1)
    LAYERS=$(echo "$SIZE_SPEC" | cut -d: -f2)
    HIDDEN=$(echo "$SIZE_SPEC" | cut -d: -f3)
    HEAD=$(echo "$SIZE_SPEC"   | cut -d: -f4)

    export N_GRU_LAYERS=$LAYERS
    export RNN_HIDDEN=$HIDDEN
    export HEAD_HIDDEN=$HEAD
    export N_SAMPLES=2048
    export CHAIN_N=12
    export TRANSITION_STEPS=40000
    export WANDB_PROJECT=hyperscalenqs-scaling-model
    export RUN_TAG=$SIZE

    sbatch sweep/run_adam_a100.slurm j1j2
    sbatch sweep/run_minsr_a100.slurm j1j2 minsr
    sbatch sweep/run_qps_a100.slurm j1j2
done
