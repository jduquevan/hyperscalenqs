#!/bin/bash

#SBATCH --job-name=pwo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --time=0-00:39:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=6

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

if [[ -n "${MODULES_TO_LOAD:-}" ]]; then
    module purge
    module load ${MODULES_TO_LOAD}
fi

export HYDRA_FULL_ERROR=1
source "${VENV_PATH:-.venv/bin/activate}"

WANDB_MODE="${WANDB_MODE:-disabled}"
WANDB_PROJECT="${WANDB_PROJECT:-anonymous-nqs}"

SEED="${1}"
RUN_NAME="${2}"
HAMILTONIAN="${3}"
WANDB_TAG="${4}"
shift 4
EXTRA_ARGS=("$@")

CMD=(
    python src/pwo.py
    "seed=${SEED}"
    "wandb_run_name=${RUN_NAME}"
    "hamiltonian=${HAMILTONIAN}"
    "wandb_tags=${WANDB_TAG}"
    "wandb_mode=${WANDB_MODE}"
    "wandb_project=${WANDB_PROJECT}"
    "${EXTRA_ARGS[@]}"
)

printf 'Executing command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"