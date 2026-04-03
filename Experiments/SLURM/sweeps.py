'''
This file initializes the sweep config with 
algo and env_name and calls the wandb sweeping agent
'''

import wandb
import argparse
from Experiments.configs.sweeps import SWEEP_CONFIG

def make_sweep(env, algo):

    if algo not in SWEEP_CONFIG:
        raise ValueError(f"Unknown algorithm {algo}, valid algorithms are {list(SWEEP_CONFIG.keys())}")

    project = f"{env}-{algo}-rl"

    config = SWEEP_CONFIG[algo].copy()
    config["parameters"]["env_name"] = {"values": [env]}

    sweep_id=wandb.sweep(config, project=project)

    return sweep_id

def parseargs():
    parser=argparse.ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--algo', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args=parseargs()

    sweep_id=make_sweep(args.env, args.algo)
    print(sweep_id)