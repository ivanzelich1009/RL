'''
This file creates a dictionary (keys=algo) with a dictionary which will be used
as the config for a wandb sweep.
'''

SWEEP_CONFIG = {
    'ppo':
        {'method': 'grid',
        'parameters': {
            'algo': {'values': ['ppo']},
            'learning_rate': {'values':[0.1, 0.00025]},
            'clip_coef': {'values': [0.05, 0.15, 0.25]},
            'gaelambda': {'values': [0.95,0.97]},
            'num_envs': {'values': [16]},
            'num_steps': {'values': [128]},
            'gamma': {'values': [0.98, 0.99]},
            'clip_ploss': {'values': [True]},
            'total_timesteps': {'values': [2000000]},
        },
        'command': ['python3', '-m', 'Experiments.run'],
    },
    'alphazero': 
    {'method': 'grid',
    'parameters': {
        'algo': {'values': ['alphazero']},
        'learning_rate': {'values':[0.1, 0.001]},
        'c_puct': {'values': [1.5,2.5]},
        'mcts_batch_size': {'values': [80, 200]},
        'total_timesteps': {'values': [2000000]},
        'num_envs': {'values': [1]},
    },
    'command': ['python3', '-m', 'Experiments.run'],
    },
    'ppodistill':
        {'method': 'grid',
        'parameters': {
            'algo': {'values': ['ppodistill']},
            'learning_rate': {'values':[0.1, 0.00025]},
            'clip_coef': {'values': [0.15, 0.25]},
            'gaelambda': {'values': [0.95,0.97]},
            'num_envs': {'values': [8]},
            'num_steps': {'values': [500]},
            'gamma': {'values': [0.98, 0.99]},
            'distill_coef': {'values': [0.5,1]},
            'clip_ploss': {'values': [True]},
            'total_timesteps': {'values': [2000000]},
            'rnd_lr': {'values': [0.01]},
            'anneal_lr': {'values': [True, False]},
        },
        'command': ['python3', '-m', 'Experiments.run'],
    },
}