#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=adam

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
SEED=${SLURM_ARRAY_TASK_ID}
HAMILTONIAN=${1:-ising}

# Scaling overrides — env vars with defaults that preserve existing behaviour
N_SAMPLES=${N_SAMPLES:-1024}
CHAIN_N=${CHAIN_N:-12}
N_GRU_LAYERS=${N_GRU_LAYERS:-3}
RNN_HIDDEN=${RNN_HIDDEN:-256}
HEAD_HIDDEN=${HEAD_HIDDEN:-256}
_DEFAULT_STEPS=$([ "$HAMILTONIAN" = "ising" ] && echo 5000 || echo 10000)
TRANSITION_STEPS=${TRANSITION_STEPS:-$_DEFAULT_STEPS}
WANDB_PROJECT=${WANDB_PROJECT:-hyperscalenqs}
RUN_NAME=${2:-adam_${HAMILTONIAN}${RUN_TAG:+_${RUN_TAG}}_seed_${SEED}}

CMD="python src/1d/adam.py \
    seed=${SEED} \
    hamiltonian=${HAMILTONIAN} \
    n_samples=${N_SAMPLES} \
    transition_steps=${TRANSITION_STEPS} \
    lr_peak=$([ "$HAMILTONIAN" = "ising" ] && echo 1e-4 || echo 3e-4) \
    lr_div_factor=$([ "$HAMILTONIAN" = "ising" ] && echo 10.0 || echo 30.0) \
    N=${CHAIN_N} \
    n_gru_layers=${N_GRU_LAYERS} \
    rnn_hidden=${RNN_HIDDEN} \
    head_hidden=${HEAD_HIDDEN} \
    wandb_run_name=${RUN_NAME} \
    wandb_mode=online \
    wandb_project=${WANDB_PROJECT}"

echo "Executing: $CMD"
eval $CMD
