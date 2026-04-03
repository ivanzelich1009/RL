#!/bin/bash

set -e 

#Configs
echo "Let's set up the sweeping agent"
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter the name of the repository you want to clone (e.g., RL): " REPO_NAME
REPO_SSH="git@github.com:${GITHUB_USERNAME}/${REPO_NAME}.git"
REPO_DIR="${REPO_NAME}"
ENV_NAME=".venv"
read -p "Enter the wandb API key: " WANDB_API
read -p "Enter your wandb entity (usually your username or team name): " entity

echo "Thanks. We will now initiate a sweep agent using your choice of environment and algorithm."
read -p "Which environment shall we use? Options are: pong, seaquest, connect4. " env
read -p "Which algorithm shall we run? Options are: ppo, alphazero, ppodistill. " algo
project="$env-$algo-rl"

#Check if repo exists, if not clone it
if [ ! -d "$REPO_DIR" ]; then
    git clone $REPO_SSH
fi


#activate env
if [ ! -d "$ENV_NAME" ]; then
    python3 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

#Set the python path to the repo
export PYTHONPATH=$REPO_DIR
echo "Set PYTHONPATH to $REPO_DIR"

#Install requirements
echo "Installing requirements..."
pip install -r RL/Experiments/requirements.txt
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Number of GPUs available: $NUM_GPUS"

#Login to wandb
echo "Logging into Wandb..."
wandb login --relogin $WANDB_API

#Initiate sweep
sweep_id=$(python3 -m Experiments.SLURM.sweeps --algo $algo --env $env | tail -n 1)
#Run jobs in parallel
for i in $(seq 0 $((NUM_GPUS-1))); do
    echo "Starting agent on GPU $i"
    CUDA_VISIBLE_DEVICES=$i wandb agent $entity/$project/$sweep_id &
    sleep 5 # Stagger the start times slightly
done

wait

echo "All agents finished."



