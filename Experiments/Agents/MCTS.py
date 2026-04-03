import math as m
import numpy as np
from Experiments.Envs import utils as moves
import torch.nn.functional as F
import torch


class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """
    def __init__(self, c_puct: float = 1.0):
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's act, state_int -> [W(s, a)]
        self.value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs = {}

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def find_leaf(self, state_int: int, player: int):
        """
        Traverse the tree until the end of game or leaf node
        :param state_int: root node state
        :param player: player to move
        :return: tuple of (value, leaf_state, player, states, actions)
        1. value: None if leaf node, otherwise equals to the game outcome for the player at leaf
        2. leaf_state: state_int of the last state
        3. player: player at the leaf node
        4. states: list of states traversed
        5. list of actions taken
        """
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = m.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            # choose action to take, in the root node add the Dirichlet noise to the probs
            valid_moves = moves.possible_moves(cur_state)
            if cur_state == state_int:
                noises = np.random.dirichlet([0.03] * len(valid_moves))
                for index,a in enumerate(valid_moves):
                    probs[a] = 0.75*probs[a] + 0.25*noises[index]

            score = [-np.inf] * moves.GAME_COLS
            for a in valid_moves:
                score[a] = (values_avg[a]+ self.c_puct * probs[a] * total_sqrt / (1 + counts[a]))
            action = int(np.argmax(score))
            actions.append(action)

            cur_state, won = moves.move(cur_state, action, cur_player)
            if won:
                # if somebody won the game, the value of the final state is -1 (as it is on opponent's turn)
                value = -1.0
            cur_player = 1-cur_player
            # check for the draw
            moves_count = len(moves.possible_moves(cur_state))
            if value is None and moves_count == 0:
                value = 0.0

        return value, cur_state, cur_player, states, actions

    def is_leaf(self, state_int):
        return state_int not in self.probs

    def search_batch(self, count, batch_size, state_int, player, net, device="cpu"):
        minibatch_size=batch_size//count
        for _ in range(count):
            self.search_minibatch(minibatch_size, state_int, player, net, device)

    def search_minibatch(self, count, state_int, player, net, device="cpu"):
        """
        Perform several MCTS searches.
        """
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned = set()

        for _ in range(count):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(state_int, player)
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    expand_states.append(leaf_state)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states, actions))

        # do expansion of nodes
        if expand_queue:
            batch = [torch.tensor(moves.bin_to_batch(expand_state), dtype=torch.float32, device=device) for expand_state in expand_states]
            batch_v = torch.stack(batch, dim=0)
            with torch.no_grad():
                values_v = net.critic(batch_v)
                logits_v = net.actor(batch_v)
                probs_v = F.softmax(logits_v, dim=-1)
                values = values_v.cpu().numpy()[:, 0]
                probs = probs_v.cpu().numpy()

            # create the nodes
            for (leaf_state, states, actions), value, prob in \
                    zip(expand_queue, values, probs):
                self.visit_count[leaf_state] = [0]*moves.GAME_COLS
                self.value[leaf_state] = [0.0]*moves.GAME_COLS
                self.value_avg[leaf_state] = [0.0]*moves.GAME_COLS
                self.probs[leaf_state] = prob
                backup_queue.append((value, states, actions))

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state_int, action in zip(states[::-1], actions[::-1]):
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += cur_value
                self.value_avg[state_int][action] = self.value[state_int][action] / self.visit_count[state_int][action]
                cur_value = -cur_value

    def get_policy_value(self, state_int, tau=1):
        """
        Extract policy and action-values by the state
        :param state_int: state of the board
        :return: (probs, values)
        """
        counts = self.visit_count[state_int]
        if tau == 0:
            probs = [0.0] * moves.GAME_COLS
            probs[np.argmax(counts)] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            total = sum(counts)
            probs = [count / total for count in counts]
        values = self.value_avg[state_int]
        return probs, values