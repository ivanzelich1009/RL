import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
EnvID="Humanoid-v5"

def make_ens(num_envs):
    return SyncVectorEnv([lambda: gym.make(EnvID) for _ in range(num_envs)])