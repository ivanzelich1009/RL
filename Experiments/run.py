import uuid
import wandb
from Experiments.configs.build import build_config
from Experiments.training.train import train
import torch

def main():
    wandb.init()
    config = build_config(wandb.config)

    run_name = f'{config["env_name"]}_{config["algo"]}_{uuid.uuid4()}'
    config['run_name']=run_name
    wandb.init(
        name=run_name,
        config=config,
        save_code=True
        )

    train(config)
    
    wandb.finish()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()