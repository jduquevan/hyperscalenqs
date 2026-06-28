#!/bin/bash
set -euo pipefail

LR_ISING=0.01
LR_HEISENBERG=0.001
LR_J1J2=0.001

for SEED in $(seq 42 51); do
    sbatch sweep/run_minsr.slurm \
        "${SEED}" \
        "minsr_ising_seed_${SEED}" \
        "ising" \
        "${LR_ISING}" \
        "minsr_ising" \
        "Gamma=-1.0" \
        "V=-1.0" \
        "pbc=True"
done

for SEED in $(seq 42 51); do
    sbatch sweep/run_minsr.slurm \
        "${SEED}" \
        "minsr_heisenberg_chain_seed_${SEED}" \
        "heisenberg" \
        "${LR_HEISENBERG}" \
        "minsr_heisenberg_chain" \
        "J=0.25" \
        "pbc=True" \
        "sign_rule=False"
done

for SEED in $(seq 42 51); do
    sbatch sweep/run_minsr.slurm \
        "${SEED}" \
        "minsr_j1j2_seed_${SEED}" \
        "j1j2" \
        "${LR_J1J2}" \
        "minsr_j1j2" \
        "J1=1.0" \
        "J2=0.5" \
        "pbc=True" \
        "j1j2_sign_rule=False"
done