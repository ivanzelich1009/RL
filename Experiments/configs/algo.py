ALGO_DEFAULTS = {
    'ppo': {
        'clip_coef': 0.2,
        'update_epochs': 4,
        'clip_vloss': True,
    },
    'a2c': {
        'update_epochs': 1,
    },
    'trpo': {
        'target_kl': 0.01,
    },
    'alphazero': {
        'mcts_batches': 80,
        'evaluation_rounds': 20,
        'play_episodes': 1,
        'num_envs': {'values': [1]},
    },
    'ppodistill': {
        'clip_coef': 0.2,
        'update_epochs': 4,
        'clip_vloss': True,
        'distill_coef': 0.5,
        'rnd_lr': 0.01
    }
}