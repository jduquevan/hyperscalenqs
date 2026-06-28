#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=minsr

#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-9

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
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
# Usage: sbatch run_minsr_a100.slurm <hamiltonian> <method>
#   hamiltonian: ising | heisenberg | j1j2  (default: ising)
#   method:      minsr | spring              (default: minsr)
SEED=${SLURM_ARRAY_TASK_ID}
HAMILTONIAN=${1:-ising}
METHOD=${2:-minsr}

# SPRING uses momentum=0.8; minSR uses momentum=0.0
MOMENTUM=$([ "$METHOD" = "spring" ] && echo 0.8 || echo 0.0)

# Constant LR: 1e-2 for Ising, 1e-3 for Heisenberg and J1-J2
LR=$([ "$HAMILTONIAN" = "ising" ] && echo 1e-2 || echo 1e-3)

# Scaling overrides — env vars with defaults that preserve existing behaviour
N_SAMPLES=${N_SAMPLES:-1024}
CHAIN_N=${CHAIN_N:-12}
N_GRU_LAYERS=${N_GRU_LAYERS:-3}
RNN_HIDDEN=${RNN_HIDDEN:-256}
HEAD_HIDDEN=${HEAD_HIDDEN:-256}
WANDB_PROJECT=${WANDB_PROJECT:-hyperscalenqs}
RUN_NAME=${METHOD}_${HAMILTONIAN}${RUN_TAG:+_${RUN_TAG}}_seed_${SEED}

CMD="python src/scaling/minsr.py \
    seed=${SEED} \
    hamiltonian=${HAMILTONIAN} \
    n_samples=${N_SAMPLES} \
    lr=${LR} \
    sr_momentum=${MOMENTUM} \
    N=${CHAIN_N} \
    n_gru_layers=${N_GRU_LAYERS} \
    rnn_hidden=${RNN_HIDDEN} \
    head_hidden=${HEAD_HIDDEN} \
    wandb_run_name=${RUN_NAME} \
    wandb_mode=online \
    wandb_project=${WANDB_PROJECT}"

echo "Executing: $CMD"
eval $CMD
