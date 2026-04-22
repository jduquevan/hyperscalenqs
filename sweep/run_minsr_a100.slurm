#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=minsr_tfim

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-9

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
#SBATCH --time=0-02:10:00
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
HAMILTONIAN=${1:-ising}
RUN_NAME=${2:-minsr_${HAMILTONIAN}_seed_${SEED}_tiny}

CMD="python src/minsr.py \
    seed=${SEED} \
    hamiltonian=${HAMILTONIAN} \
    N=12 \
    n_gru_layers=1 \
    rnn_hidden=64 \
    head_hidden=64 \
    wandb_run_name=${RUN_NAME} \
    wandb_mode=online"

echo "Executing: $CMD"
eval $CMD
