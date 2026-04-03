BASE_CONFIG = {
    # experiment
    'exp_name': 'Pong',

    # lr scheduling
    'lr_decay': 'linear',          # 'linear' or 'cosine'
    'min_lr_frac': 0.0,
    'learning_rate': 2.5e-4,
    'warmup_period': 0.0,
    'anneal_lr': False,
    # rollout
    'horizon_length': 42,
    'num_steps': 2000,
    'num_envs': 4,
    'total_timesteps': 200000,
    'seed':1,

    # logging
    'wandb_log': False,

    # RL core
    'gamma': 0.99,
    'gaelambda': 0.95,
    'entropy_coef': 0.01,
    'value_coef': 0.5,
    'norm_adv' : True,


    # batching
    'num_minibatches': 4,

    # updates
    'update_epochs': 1,

    # clipping / PPO-ish
    'clip_vloss': False,
    'clip_ploss': False,
    'clip_coef': 0.2,
    'target_kl': 0.01,
    'is_loss_clip': False,
    'beta': 0.9,

    # AlphaZero / MCTS
    'mcts_batch_size': 80,
    'mcts_minibatches': 8,
    'min_replay_to_train': 2000,
    'best_win_ratio': 0.6,
    'evaluation_rounds': 20,
    'replay_buffer': 5000,
    'steps_before_tau_0': 10,

    # network
    'num_filters': 64,
}