#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=qps_vit_j1j2

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-4

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=60G
#SBATCH --time=0-00:10:00
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
RUN_NAME=${1:-qps_square_j1j2_vit_seed_${SEED}}

CMD="python src/qps_vit_j1j2.py \
    seed=${SEED} \
    L=10 \
    vit_revision=main \
    n_samples=8192 \
    n_chains=256 \
    n_discard_per_chain=100 \
    eval_n_discard_per_chain=100 \
    mcmc_d_max=2 \
    use_snis=true \
    ppo_clip_eps=1e-2 \
    ppo_epochs=4 \
    transition_steps=1000000 \
    wandb_run_name=${RUN_NAME} \
    wandb_mode=online"

echo "Executing: $CMD"
eval $CMD
