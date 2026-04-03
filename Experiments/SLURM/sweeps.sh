#!/bin/bash
#SBATCH --job-name=rl-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-3

# activate env
source ~/.bashrc
conda activate ENV_NAME

# run wandb agent
wandb agent ivanzelich24-columbia-university/connect4-rl/SWEEP_ID