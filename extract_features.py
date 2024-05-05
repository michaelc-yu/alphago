
import numpy as np
import go



def stone_color_feature(position): # returns 3 feature channels
    board = position['board']
    features = np.zeros([3, position['board_size'], position['board_size']], dtype=np.uint8)
    if position['to_play'] == go.BLACK:
        features[0, :, :] = (board == go.BLACK).astype(np.uint8)  # Black stones in the first channel
        features[1, :, :] = (board == go.WHITE).astype(np.uint8)  # White stones in the second channel
    else:
        features[0, :, :] = (board == go.WHITE).astype(np.uint8)  # White stones in the first channel
        features[1, :, :] = (board == go.BLACK).astype(np.uint8)  # Black stones in the second channel

    features[2, :, :] = (board == go.EMPTY).astype(np.uint8) # Empty in the third channel
    return features

def ones_feature(position): # returns 1 feature channel
    return np.ones([1, position['board_size'], position['board_size']], dtype=np.uint8)

def liberty_feature(position): # returns 8 feature channels
    feature_map = np.zeros([8, position['board_size'], position['board_size']], dtype=np.int8)
    visited = set()
    board = position['board']
    for x in range(position['board_size']):
        for y in range(position['board_size']):
            if (x, y) in visited or board[x][y] == go.EMPTY:
                continue
            group, liberties = go.get_liberties(x, y, board)
            visited.update(group)
            liberty_index = min(liberties-1, 7)
            for stone_x, stone_y in group:
                feature_map[liberty_index][stone_x][stone_y] = 1
    return feature_map

def recent_moves_feature(position): # returns 8 feature channels
    features = [position['recency'][i] for i in range(1, 9)]
    features = np.stack(features, axis=0)
    return features

