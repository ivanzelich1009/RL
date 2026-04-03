'''
Helper function used to merge base configs, algorithmic specific configs and sweeps together.
'''

from Experiments.configs.base import BASE_CONFIG
from Experiments.configs.algo import ALGO_DEFAULTS

def build_config(wandb_config):
    algo = wandb_config['algo']

    config = BASE_CONFIG.copy()
    config.update(ALGO_DEFAULTS.get(algo, {}))
    config.update(wandb_config)

    # derived values
    config['batch_size'] = config['num_envs'] * config['num_steps']
    config['minibatch_size'] = config['batch_size'] // config['num_minibatches']
    config['algo'] = algo

    return config
