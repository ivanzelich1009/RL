from torch import nn
import torch
from Experiments.Agents.nn_utils import build_network
from torch.distributions import Categorical
from Experiments.Envs import utils as moves
import Experiments.Agents.MCTS as mcts
import numpy as np

class AlphaZero(nn.Module):
    def __init__(self, env, num_filters = 64):
        super().__init__()
        self.name='AlphaZero'
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
        
        self.CriticSpec = [('conv2d', num_filters)] + [('residual', num_filters)] * 5 + self.critic_head
        self.ActorSpec =  [('conv2d', num_filters)] + [('residual', num_filters)] * 5 + self.actor_head
        self.critic = build_network(env.single_observation_space.shape, self.CriticSpec)
        self.actor = build_network(env.single_observation_space.shape, self.ActorSpec)
    
    def get_value(self, obs):
        return self.critic(obs)
    
    def get_valid_logits(self, obs):
        logits = self.actor(obs)
        logits = torch.clamp(logits, -20, 20)
        mask = moves.get_mask(obs).to(dtype=torch.bool)
        invalid_rows = mask.sum(dim=-1) == 0
        masked_logits = logits.masked_fill(~mask, -1e9)
        if invalid_rows.any():
            masked_logits[invalid_rows] = torch.zeros_like(masked_logits[invalid_rows])
        return masked_logits
    
    def get_action_value(self, obs, action=None):
        value = self.critic(obs)
        logits = self.get_valid_logits(obs)
        dist = Categorical(logits=logits)
        if action == None:
            action = dist.sample()
        return action, dist(action), value

    def get_logprobs_value(self, obs):
        logits = self.get_valid_logits(obs)
        dist = Categorical(logits=logits)
        value = self.critic(obs)
        return dist.log_prob(obs), value

    def play_game(self,
                  envs, 
                  mcts_stores,
                  replay_buffer, 
                  net1, net2,
                  steps_before_tau_0, 
                  mcts_searches, 
                  mcts_batch_size,
                  net1_plays_first = None,
                  device='cpu'):
        """
        Play one single game, memorizing transitions into the replay buffer
        :param mcts_stores: could be None or single MCTS or two MCTSes for individual net
        :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
        :param net1: player1
        :param net2: player2
        :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
        """
        if mcts_stores is None:
            mcts_stores = [mcts.MCTS(), mcts.MCTS()]
        elif isinstance(mcts_stores, mcts.MCTS):
            mcts_stores = [mcts_stores, mcts_stores]

        _, _ = envs.reset()
        state = envs.get_attr('state')[0]
        nets = [net1, net2]

        if net1_plays_first is None:
            cur_player = np.random.choice(2)
        else:
            cur_player = 0 if net1_plays_first else 1

        step = 0
        tau = 1 if steps_before_tau_0 > 0 else 0
        game_history = []

        result = None
        net1_result = None

        while result is None:
            mcts_stores[cur_player].search_batch(
                mcts_searches, mcts_batch_size, state,
                cur_player, nets[cur_player], device=device)
            probs, _ = mcts_stores[cur_player].get_policy_value(state, tau=tau)
            game_history.append((envs.get_attr('_get_obs')[0], cur_player, probs))
            action = np.random.choice(moves.GAME_COLS, p=probs)
            _, reward_v, done_v, _, _ = envs.step(np.array([action]))
            state = envs.get_attr('state')[0]
            reward = reward_v[0]
            done = done_v[0]

            if done:
                if reward==0:
                    result = 0
                    net1_result = 0
                    break
                else:
                    result = 1
                    net1_result = 1 if cur_player == 0 else -1
                    break
            step += 1
            if step >= steps_before_tau_0:
                tau = 0

        if replay_buffer is not None:
            for state, _, probs in reversed(game_history):
                replay_buffer.append((state, probs, result))
                result = -result

        return net1_result, step