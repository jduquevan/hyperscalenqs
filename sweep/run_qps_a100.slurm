#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=qps

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-9

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
HAMILTONIAN=${1:-ising}
RUN_NAME=${2:-qps_${HAMILTONIAN}_seed_${SEED}}

CMD="python src/qps.py \
    seed=${SEED} \
    hamiltonian=${HAMILTONIAN} \
    n_samples=1024 \
    transition_steps=$([ "$HAMILTONIAN" = "ising" ] && echo 20000 || echo 40000) \
    lr_peak=1e-4 \
    N=12 \
    n_gru_layers=3 \
    rnn_hidden=256 \
    head_hidden=256 \
    wandb_run_name=${RUN_NAME} \
    wandb_mode=online \
    wandb_project=hyperscalenqs"

echo "Executing: $CMD"
eval $CMD
