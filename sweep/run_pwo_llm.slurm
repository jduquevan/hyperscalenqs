#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=pwo

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Define how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-06:59:00
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
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
source "${VENV_PATH:-.venv/bin/activate}"

# Define all arguments
SEED=${1:-42}

# Construct the full command
CMD=(python -m src.llm.rwkv7_pwo "seed=${SEED}")

# Print the full command for debugging
printf 'Executing command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

# Execute the command
"${CMD[@]}"