from Experiments.Agents.A2C import A2CAgent
from Experiments.Agents.PPO import PPOAgent
from Experiments.Agents.TRPO import TRPOAgent
from Experiments.Agents.AlphaZero import AlphaZero
from Experiments.Agents.PPOdistill import PPOdistillAgent
from Experiments.training.ActorCriticTrain import training_loop as actor_critic_training_loop
from Experiments.training.AlphaZeroTrain import training_loop as alpha_zero_training_loop


AGENTS={
    'ppo': (PPOAgent, actor_critic_training_loop),
    'a2c': (A2CAgent, actor_critic_training_loop),
    'trpo': (TRPOAgent, actor_critic_training_loop),
    'alphazero': (AlphaZero, alpha_zero_training_loop),
    'ppodistill': (PPOdistillAgent, actor_critic_training_loop),
}