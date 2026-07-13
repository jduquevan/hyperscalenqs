#!/bin/bash

##############################
#       Job blueprint        #
##############################

#SBATCH --job-name=pwo_j1j2_square_v2
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=40G
#SBATCH --time=0-00:59:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=6

module purge
eval "$(conda shell.bash hook)"
conda deactivate

export HYDRA_FULL_ERROR=1
export HF_HOME="/home/mila/j/juan.duque/scratch/huggingface"
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export TF_GPU_ALLOCATOR="cuda_malloc_async"

source .venv/bin/activate

SEED=${1:-42}
CONN_CHUNK_SIZE=${2:-8}
N_SAMPLES=${3:-2048}
PPO_EPOCHS=${4:-4}
WANDB_MODE=${5:-online}

shift 5 || true
EXTRA_OVERRIDES="$*"

CMD="python -m src.2d.pwo_j1j2_square_v2 \
  seed=${SEED} \
  local_energy_conn_chunk_size=${CONN_CHUNK_SIZE} \
  n_samples=${N_SAMPLES} \
  ppo_epochs=${PPO_EPOCHS} \
  wandb_mode=${WANDB_MODE} \
  ${EXTRA_OVERRIDES}"

stdbuf -o0 echo "Executing command: $CMD"

eval $CMD
