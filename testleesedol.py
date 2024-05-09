
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import load_data
import policy
import value


test = "./unit-tests/leesedol"
policy_data, value_data = load_data.get_data(test)

feats, moves = zip(*policy_data)

assert isinstance(feats, tuple), "Expected features to be a tuple"
assert isinstance(feats[0], np.ndarray), "Expected each feature to be a numpy.ndarray"
assert isinstance(moves, tuple), "Expected moves to be a tuple"
assert isinstance(moves[0], list), "Expected each move to be a list"

assert(len(feats) == 211), "Expected there to be 211 moves in the game"

assert(len(feats[0])) == 28, "Expected there to be 28 features per board position"

for i in range(28):
    assert feats[0][i].shape == (19, 19), f"Expected feature shape at index {i} to be (19, 19), but got {feats[0][i].shape}"


# Check ones
assert np.all(feats[0][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[15][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[32][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[84][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[209][3] == 1), "All elements in the ones feature plane must be 1"


# Check stone color feature

combined = np.logical_or(np.logical_or(feats[0][0], feats[0][1]), feats[0][2])
assert np.all(combined), "Not all positions on the board are filled with 1s when combining the features."

sum_of_features = feats[0][0] + feats[0][1] + feats[0][2]
assert np.all(sum_of_features == 1), "Each position must have exactly one '1' across the three boards."

assert np.all(feats[0][2] == 1), "All board should be empty in first position."


# print("after first move")
# print(feats[1][0])
assert np.all(feats[1][0] == 0), "After black makes first move, white's feature plane should be all empty"
# print("")
# print(feats[1][1])
assert np.count_nonzero(feats[1][1] == 1) == 1, "Black should have only one stone on the board after first move"
# print("")
# print(feats[1][2])
assert np.count_nonzero(feats[1][2] == 0) == 1, "Board should be all empty except for 1 stone after first move"

# print("after 2nd move")
# print(feats[2][0])
assert np.count_nonzero(feats[2][0] == 1) == 1, "After white makes second move, black should have only one stone on the board"
# print("")
# print(feats[2][1])
assert np.count_nonzero(feats[2][1] == 1) == 1, "White should have only one stone on the board after second move"
# print("")
# print(feats[2][2])
assert np.count_nonzero(feats[2][2] == 0) == 2, "Board should be all 1's except two 0s"

# print("after 211 move")
# print("black stones:")
# print(feats[210][0])
# print("white stones:")
# print(feats[210][1])
# print("empty:")
# print(feats[210][2])

combined = np.logical_or(np.logical_or(feats[210][0], feats[210][1]), feats[210][2])
assert np.all(combined), "Not all positions on the board are filled with 1s when combining the features."

sum_of_features = feats[210][0] + feats[210][1] + feats[210][2]
assert np.all(sum_of_features == 1), "Each position must have exactly one '1' across the three boards."


# print("after 22 move")
# print("black stones:")
# print(feats[22][0])
# print("white stones:")
# print(feats[22][1])
# print("empty:")
# print(feats[22][2])

# print("after 23 move")
# print("white stones:")
# print(feats[23][0])
# print("black stones:")
# print(feats[23][1])
# print("empty:")
# print(feats[23][2])

# print("")

# for i in range(1, 9):
#     print(f"after 2nd move, stones with {i} liberty(s):")
#     print(feats[2][3+i])
#     print("")


# print("")

# for i in range(1, 9):
#     print(f"after 2nd move, recent {i} moves:")
#     print(feats[2][11+i])
#     print("")




# value data

feats, winners = zip(*value_data)

assert isinstance(feats, tuple), "Expected features to be a tuple"
assert isinstance(feats[0], np.ndarray), "Expected each feature to be a numpy.ndarray"
assert isinstance(winners, tuple), "Expected winners to be a tuple"
assert isinstance(winners[0], int), "Expected each winner to be an int"

assert(len(feats) == 211), "Expected there to be 211 moves in the game"

assert(len(feats[0])) == 28, "Expected there to be 28 features per board position"

for i in range(28):
    assert feats[0][i].shape == (19, 19), f"Expected feature shape at index {i} to be (19, 19), but got {feats[0][i].shape}"


# Check ones
assert np.all(feats[0][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[15][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[32][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[84][3] == 1), "All elements in the ones feature plane must be 1"
assert np.all(feats[209][3] == 1), "All elements in the ones feature plane must be 1"


# Check stone color feature

combined = np.logical_or(np.logical_or(feats[0][0], feats[0][1]), feats[0][2])
assert np.all(combined), "Not all positions on the board are filled with 1s when combining the features."

sum_of_features = feats[0][0] + feats[0][1] + feats[0][2]
assert np.all(sum_of_features == 1), "Each position must have exactly one '1' across the three boards."

assert np.all(feats[0][2] == 1), "All board should be empty in first position."


assert np.all(feats[1][0] == 0), "After black makes first move, white's feature plane should be all empty"
assert np.count_nonzero(feats[1][1] == 1) == 1, "Black should have only one stone on the board after first move"
assert np.count_nonzero(feats[1][2] == 0) == 1, "Board should be all empty except for 1 stone after first move"
assert np.count_nonzero(feats[2][0] == 1) == 1, "After white makes second move, black should have only one stone on the board"
assert np.count_nonzero(feats[2][1] == 1) == 1, "White should have only one stone on the board after second move"
assert np.count_nonzero(feats[2][2] == 0) == 2, "Board should be all 1's except two 0s"

combined = np.logical_or(np.logical_or(feats[210][0], feats[210][1]), feats[210][2])
assert np.all(combined), "Not all positions on the board are filled with 1s when combining the features."

sum_of_features = feats[210][0] + feats[210][1] + feats[210][2]
assert np.all(sum_of_features == 1), "Each position must have exactly one '1' across the three boards."

