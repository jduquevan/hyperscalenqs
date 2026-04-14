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
#SBATCH --mem=16G
#SBATCH --time=0-01:09:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=6

# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda deactivate
export HYDRA_FULL_ERROR=1
source .venv/bin/activate

# Define all arguments
SEED=${1}
RUN_NAME=${2:-minsr_seed_${SEED}}

# Construct the full command
CMD="python src/minsr.py seed=${SEED} wandb_run_name=${RUN_NAME}"

# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD