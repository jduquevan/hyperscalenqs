#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=pwo_j1j2_square

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=40G
#SBATCH --time=0-04:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-short
#SBATCH --constraint=A100.80G
#SBATCH --cpus-per-task=8
# Uncomment and set if your cluster requires an account:
##SBATCH --account=<YOUR_ACCOUNT>

# Environment setup
module purge
module load ALICE/default
module load CUDA/12.4.0
export HYDRA_FULL_ERROR=1
source .venv/bin/activate
# Load repo-local secrets (not tracked by git)
set -a; [ -f "$SLURM_SUBMIT_DIR/.env" ] && source "$SLURM_SUBMIT_DIR/.env"; set +a

mkdir -p "$HOME/slurm_logs" logs/

# Arguments
SEED=${SLURM_ARRAY_TASK_ID}
L=${1:-10}
J2=${2:-0.5}
CONN_CHUNK_SIZE=${3:-8}
RUN_NAME=${4:-pwo_j1j2_L${L}_J2${J2}_seed${SEED}}
WANDB_MODE=${5:-online}

shift 5 || true
EXTRA_OVERRIDES="$*"

CMD="python -m src.2d.pwo_j1j2_square \
    seed=${SEED} \
    local_energy_conn_chunk_size=${CONN_CHUNK_SIZE} \
    L=${L} \
    J2=${J2} \
    n_samples=4096 \
    ppo_epochs=4 \
    wandb_mode=${WANDB_MODE} \
    wandb_project=hyperscalenqs-j1j2-square \
    ${EXTRA_OVERRIDES}"

echo "Executing: $CMD"
eval $CMD
