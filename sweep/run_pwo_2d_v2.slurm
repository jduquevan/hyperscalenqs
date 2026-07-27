#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=pwo_j1j2_square_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=40G
#SBATCH --time=0-00:59:00
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
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export TF_GPU_ALLOCATOR="cuda_malloc_async"

source "${VENV_PATH:-.venv/bin/activate}"

SEED=${1:-42}
CONN_CHUNK_SIZE=${2:-8}
N_SAMPLES=${3:-2048}
PPO_EPOCHS=${4:-4}
WANDB_MODE=${5:-disabled}

shift 5 || true
EXTRA_OVERRIDES="$*"

CMD="python -m src.2d.pwo_j1j2_square \
  seed=${SEED} \
  local_energy_conn_chunk_size=${CONN_CHUNK_SIZE} \
  n_samples=${N_SAMPLES} \
  ppo_epochs=${PPO_EPOCHS} \
  wandb_mode=${WANDB_MODE} \
  ${EXTRA_OVERRIDES}"

stdbuf -o0 echo "Executing command: $CMD"

eval $CMD
