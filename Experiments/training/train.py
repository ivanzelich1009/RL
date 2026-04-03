from Experiments.training._agent_registry import AGENTS
from Experiments.Envs._env_registry import env_registry
from torch.optim import Adam as AdamSandler
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


def make_envs(config):
    return SyncVectorEnv([lambda: env_registry.get(config['env_name'], None)() for _ in range(config['num_envs'])])

def train(config):
    name = config['algo']
    agentfn, trainfn = AGENTS.get(name, None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    envs = make_envs(config)
    try:
        if name == 'ppodistill':
            agent = agentfn(envs, num_filters=config['num_filters'], rnd_lr=config.get('rnd_lr', 0.01)).to(device)
            optimizer = AdamSandler(list(agent.critic.parameters()) + list(agent.actor.parameters()), 
                                    lr = config.get('lr', 2.5e-4), 
                                    eps = config.get('epsilon', 1e-5))
        else:
            agent = agentfn(envs, config['num_filters']).to(device)
            optimizer = AdamSandler(agent.parameters(), 
                                    lr = config.get('lr', 2.5e-4), 
                                    eps = config.get('epsilon', 1e-5))

        trainfn(
            envs,
            config,
            device,
            optimizer,
            agent
        )
    finally:
        envs.close()