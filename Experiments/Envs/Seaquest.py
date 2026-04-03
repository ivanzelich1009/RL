import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
EnvID="Seaquest-v4"
difficulty=1

def Seaquest():
    env=gym.make(EnvID, difficulty=difficulty, frameskip=1)
    env = AtariPreprocessing(
    env,
    frame_skip=4,
    grayscale_obs=True,
    scale_obs=True,
    )
    env = FrameStackObservation(env, stack_size=4)
    return env
