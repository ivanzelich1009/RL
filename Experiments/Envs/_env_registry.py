from Experiments.Envs.Connect4 import Connect4Env
from Experiments.Envs.Pong import Pong
from Experiments.Envs.Seaquest import Seaquest
env_registry={'connect4': Connect4Env, 'Connect4': Connect4Env, 
              'pong': Pong, 'Pong': Pong, 
              'seaquest': Seaquest, 'Seaquest': Seaquest}