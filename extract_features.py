
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

def num_captures_feature(position): # returns 8 feature channels
    board_size = position['board_size']
    color_to_play = position['to_play']
    feature_map = np.zeros([8, board_size, board_size], dtype=np.int8)
    board = position['board']
    dirs = ((-1,0), (1,0), (0,-1), (0,1))
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] != go.EMPTY:
                continue
            for dir in dirs:
                x, y = i + dir[0], j + dir[1]
                if 0 <= x < board_size and 0 <= y < board_size:
                    if board[x][y] == (-1 * color_to_play): # opponent stone
                        group, liberties = go.get_liberties(x, y, board)
                        if liberties == 1:
                            would_capture_index = len(group)-1
                            feature_map[would_capture_index][i][j] = 1
    return feature_map

FEATURE_LIST = [
    stone_color_feature,
    ones_feature,
    liberty_feature,
    recent_moves_feature,
    num_captures_feature
]

def extract_features(position):
    return np.concatenate([feature_extractor(position) for feature_extractor in FEATURE_LIST], axis=0)

