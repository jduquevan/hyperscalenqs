#!/bin/bash

#SBATCH --job-name=minsr
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --time=0-00:39:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=8

set -euo pipefail

module purge
eval "$(conda shell.bash hook)"
conda deactivate
export HYDRA_FULL_ERROR=1
source .venv/bin/activate

SEED="${1}"
RUN_NAME="${2}"
HAMILTONIAN="${3}"
LR="${4}"
WANDB_TAG="${5}"
shift 5
EXTRA_ARGS=("$@")

CMD=(
    python src/minsr.py
    "seed=${SEED}"
    "wandb_run_name=${RUN_NAME}"
    "hamiltonian=${HAMILTONIAN}"
    "lr=${LR}"
    "wandb_tags=${WANDB_TAG}"
    "${EXTRA_ARGS[@]}"
)

printf 'Executing command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"