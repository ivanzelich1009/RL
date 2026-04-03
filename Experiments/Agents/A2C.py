from torch import nn
from Experiments.Agents.nn_utils import build_network
from torch.distributions import Categorical


class A2CAgent(nn.Module):
    def __init__(self, env, num_filters = 64):
        super().__init__()
        self.name='A2C'
        #Warning: For board games the head should be flattened before passing to fully connected linear layers.
        self.critic_head = [('conv2d', 1, {'std': 1}, {'kernel':1 }), 
                    ('flatten',), 
                    ('linear',20, {'std':1}), 
                    ('leaky_relu',), 
                    ('linear',1, {'std':1}), 
                    ('tanh',)]
        self.actor_head = [('conv2d', 1, {'std': 0.01}, {'kernel':1 }), 
                    ('flatten',), 
                    ('linear', env.single_action_space.n)]
        self.CriticSpec = [('conv2d', num_filters, {}, {'kernel': 8, 'stride':4}), 
                           ('conv2d', num_filters, {}, {'kernel': 4, 'stride': 2, 'padding': 1}),
                           ('residual', num_filters),
                           ('residual', num_filters),
                           ('residual', num_filters),
                           *self.critic_head]
        self.ActorSpec = [('conv2d', num_filters, {}, {'kernel': 8, 'stride':4}), 
                           ('conv2d', num_filters, {}, {'kernel': 4, 'stride': 2, 'padding': 1}),
                           ('residual', num_filters),
                           ('residual', num_filters),
                           *self.actor_head]
        self.critic = build_network(env.single_observation_space.shape, self.CriticSpec)
        self.actor = build_network(env.single_observation_space.shape, self.ActorSpec)
    
    def get_value(self, obs):
        return self.critic(obs)
    
    def get_valid_logits(self, obs):
        logits = self.actor(obs)
        return logits
    
    def get_action_value(self, obs, action=None):
        logits = self.get_valid_logits(obs)
        dist = Categorical(logits=logits)
        value = self.critic(obs)
        if action == None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value