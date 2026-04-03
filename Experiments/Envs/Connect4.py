import gymnasium
from gymnasium import spaces
import numpy as np
from . import utils as moves

#Constants
GAME_ROWS = 6
GAME_COLS = 7
BITS_IN_LEN = 3
PLAYER_BLACK = 1
PLAYER_WHITE = 0
COUNT_TO_WIN = 4

INITIAL_STATE = moves.encode_lists([[]] * GAME_COLS)

class Connect4Env(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.action_space=spaces.Discrete(GAME_COLS)
        self.observation_space=spaces.Box(
            low = 0,
            high = 1,
            shape = (2, GAME_ROWS, GAME_COLS),
            dtype = np.int8
        )
        self.state = INITIAL_STATE
        self.current_player=None
        self.done = False

    def _get_obs(self):
        return moves.bin_to_batch(self.state) #shape=(2,GAME_ROWS, GAME_COLS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = INITIAL_STATE
        self.current_player = PLAYER_WHITE
        self.done = False

        return self._get_obs(), {}
    
    def _get_info_state(self):
        return {'action_mask': moves.get_mask(self._get_obs())}
    
    def step(self, action):
        if isinstance(action, (np.ndarray, list, tuple)):
            action = int(np.asarray(action).item())
        else:
            action = int(action)

        if self.done:
            # Return terminal state if already done
            return self._get_obs(), 0.0, True, False, {}

        terminated = False
        truncated = False

        new_state, won = moves.move(self.state, action, self.current_player)
        self.state = new_state
        self.current_player = 1 - self.current_player

        reward = 0.0
        if won:
            terminated = True
            reward += 1
        elif len(moves.possible_moves(new_state)) == 0:
            terminated = True
        self.done = terminated or truncated

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            print("\n".join(moves.render(self.state)))