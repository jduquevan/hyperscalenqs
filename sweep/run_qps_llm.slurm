#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=qps

# Remove one # to uncomment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-03:59:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=8

# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda deactivate
export HYDRA_FULL_ERROR=1
export HF_HOME="/home/mila/j/juan.duque/scratch/huggingface"
source .venv/bin/activate

# Define all arguments
SEED=${1}
ENV=${2}

# Construct the full command
CMD="python -m src.llm.rwkv7_qps"

# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD