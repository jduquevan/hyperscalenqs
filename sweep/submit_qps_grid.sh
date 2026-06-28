#!/bin/bash
set -euo pipefail

for SEED in $(seq 42 51); do
    sbatch sweep/run_qps.slurm \
        "${SEED}" \
        "adam_ising_seed_${SEED}" \
        "ising" \
        "adam_j1j2" \
        "Gamma=-1.0" \
        "V=-1.0" \
        "pbc=True"
done

# for SEED in $(seq 42 51); do
#     sbatch sweep/run_qps.slurm \
#         "${SEED}" \
#         "adam_heisenberg_chain_seed_${SEED}" \
#         "heisenberg" \
#         "adam_heisenberg_chain" \
#         "J=0.25" \
#         "pbc=True" \
#         "sign_rule=False"
# done

# for SEED in $(seq 42 51); do
#     sbatch sweep/run_qps.slurm \
#         "${SEED}" \
#         "adam_j1j2_seed_${SEED}" \
#         "j1j2" \
#         "adam_j1j2" \
#         "J1=1.0" \
#         "J2=0.5" \
#         "pbc=True" \
#         "j1j2_sign_rule=False"
# done