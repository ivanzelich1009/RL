'''Helper functions used for the Connect4 Environment'''
import numpy as np

#Constants
GAME_ROWS = 6
GAME_COLS = 7
BITS_IN_LEN = 3
PLAYER_BLACK = 1
PLAYER_WHITE = 0
COUNT_TO_WIN = 4

def bits_to_int(bits):
    res = 0
    for b in bits:
        res *= 2
        res += b
    return res


def int_to_bits(num, bits):
    res = []
    for _ in range(bits):
        res.append(num % 2)
        num //= 2
    return res[::-1]


def encode_lists(field_lists):
    """
    Encode lists representation into the binary numbers
    :param field_lists: list of GAME_COLS lists with 0s and 1s
    :return: integer number with encoded game state
    """
    assert isinstance(field_lists, list)
    assert len(field_lists) == GAME_COLS

    bits = []
    len_bits = []
    for col in field_lists:
        bits.extend(col)
        free_len = GAME_ROWS-len(col)
        bits.extend([0] * free_len)
        len_bits.extend(int_to_bits(free_len, bits=BITS_IN_LEN))
    bits.extend(len_bits)
    return bits_to_int(bits)


INITIAL_STATE = encode_lists([[]] * GAME_COLS)


def decode_binary(state_int):
    """
    Decode binary representation into the list view
    :param state_int: integer representing the field
    :return: list of GAME_COLS lists
    """
    assert isinstance(state_int, int)
    bits = int_to_bits(state_int, bits=GAME_COLS*GAME_ROWS + GAME_COLS*BITS_IN_LEN)
    res = []
    len_bits = bits[GAME_COLS*GAME_ROWS:]
    for col in range(GAME_COLS):
        vals = bits[col*GAME_ROWS:(col+1)*GAME_ROWS]
        lens = bits_to_int(len_bits[col*BITS_IN_LEN:(col+1)*BITS_IN_LEN])
        if lens > 0:
            vals = vals[:-lens]
        res.append(vals)
    return res

#Takes a binary representation and maps it to an nparray in the representation required by the neural networks.
def bin_to_batch(bin):

    field = decode_binary(bin)
    state = np.zeros((2, GAME_ROWS, GAME_COLS), dtype=np.int8)

    for col_idx, col in enumerate(field):
        for height, val in enumerate(col):
            row = GAME_ROWS - height - 1
            if val==1:
                state[val, row, col_idx] = val
            elif val==0:
                    state[val,row,col_idx]=1-val

    return state

'''Given an observation for the networks, find the valid mask.'''
def get_mask(state_obs):
    top_row = state_obs[:, :, 0, :]      # shape (B, 2, GAME_COLS)
    mask = (top_row[:, 0, :] == 0) & (top_row[:, 1, :]==0)
    return mask


def possible_moves(state_int):
    """
    :param state_int: field representation
    :return: the list of columns which we can make a move
    """
    assert isinstance(state_int, int)
    field = decode_binary(state_int)
    return [idx for idx, col in enumerate(field) if len(col) < GAME_ROWS]



def _check_won(field, col, delta_row):
    """
    Check for horisontal/diagonal win condition for the last player moved in the column
    :param field: list of lists
    :param col: column index
    :param delta_row: if 0, checks for horisonal won, 1 for rising diagonal, -1 for falling
    :return: True if won, False if not
    """
    player = field[col][-1]
    coord = len(field[col])-1
    total = 1
    # negative dir
    cur_coord = coord - delta_row
    for c in range(col-1, -1, -1):
        if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= GAME_ROWS:
            break
        if field[c][cur_coord] != player:
            break
        total += 1
        if total == COUNT_TO_WIN:
            return True
        cur_coord -= delta_row
    # positive dir
    cur_coord = coord + delta_row
    for c in range(col+1, GAME_COLS):
        if len(field[c]) <= cur_coord or cur_coord < 0 or cur_coord >= GAME_ROWS:
            break
        if field[c][cur_coord] != player:
            break
        total += 1
        if total == COUNT_TO_WIN:
            return True
        cur_coord += delta_row
    return False


def move(state_int, col, player):
    """
    Perform move into given column. Assume the move could be performed, otherwise, assertion will be raised
    :param state_int: current state
    :param col: column to make a move
    :param player: player index (PLAYER_WHITE or PLAYER_BLACK
    :return: tuple of (state_new, won). Value won is bool, True if this move lead
    to victory or False otherwise (but it could be a draw)
    """
    assert isinstance(state_int, int)
    assert isinstance(col, int)
    assert 0 <= col < GAME_COLS
    assert player == PLAYER_BLACK or player == PLAYER_WHITE
    field = decode_binary(state_int)
    assert len(field[col]) < GAME_ROWS
    field[col].append(player)
    # check for victory: the simplest vertical case
    suff = field[col][-COUNT_TO_WIN:]
    won = suff == [player] * COUNT_TO_WIN
    if not won:
        won = _check_won(field, col, 0) or _check_won(field, col, 1) or _check_won(field, col, -1)
    state_new = encode_lists(field)
    return state_new, won


def render(state_int):
    state_list = decode_binary(state_int)
    data = [[' '] * GAME_COLS for _ in range(GAME_ROWS)]
    for col_idx, col in enumerate(state_list):
        for rev_row_idx, cell in enumerate(col):
            row_idx = GAME_ROWS - rev_row_idx - 1
            data[row_idx][col_idx] = str(cell)
    return [''.join(row) for row in data]


def update_counts(counts_dict, key, counts):
    v = counts_dict.get(key, (0, 0, 0))
    res = (v[0] + counts[0], v[1] + counts[1], v[2] + counts[2])
    counts_dict[key] = res