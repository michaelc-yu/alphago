
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import load_data
import policy
import value


test = "./unit-tests/game"
policy_data, value_data = load_data.get_data(test)

feats, moves = zip(*policy_data)

assert isinstance(feats, tuple), "Expected features to be a tuple"
assert isinstance(feats[0], np.ndarray), "Expected each feature to be a numpy.ndarray"
assert isinstance(moves, tuple), "Expected moves to be a tuple"
assert isinstance(moves[0], list), "Expected each move to be a list"

# print(len(feats))
assert(len(feats) == 97), "Expected there to be 97 moves in the game"

assert(len(feats[0])) == 28, "Expected there to be 28 features per board position"

for i in range(28):
    assert feats[0][i].shape == (19, 19), f"Expected feature shape at index {i} to be (19, 19), but got {feats[0][i].shape}"


# Check ones
assert np.all(feats[0][3] == 1), "All elements in the ones feature plane must be 1"


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

# print("after 18 move")
# print("black stones:")
# print(feats[18][0])
# print("white stones:")
# print(feats[18][1])
# print("empty:")
# print(feats[18][2])
assert (feats[18][0][3][6] == 0), "Black should no longer have stone here"
assert (feats[18][1][3][6] == 0), "White should not have stone here"
assert (feats[18][2][3][6] == 1), "There should be an empty spot here after capture"


# num captures feature

# print("after 17 move")
# print("white stones:")
# print(feats[17][0])
# print("black stones:")
# print(feats[17][1])
# print("empty:")
# print(feats[17][2])

assert (feats[17][20][4][6] == 1), "White should be able to capture 1 if playing here"
assert np.count_nonzero(feats[17][20] == 1) == 1, "White should only be able to capture 1"

for i in range(1, 8):
    assert np.all(feats[17][20+i] == 0), "White should not be able to capture any other amount of stones"



# value data
# print("checking value data")
feats, winners = zip(*value_data)

assert isinstance(feats, tuple), "Expected features to be a tuple"
assert isinstance(feats[0], np.ndarray), "Expected each feature to be a numpy.ndarray"
assert isinstance(winners, tuple), "Expected winners to be a tuple"
assert isinstance(winners[0], int), "Expected each winner to be an int"

assert (winners[0] == 1), "Black (1) is the winner"

assert(len(feats) == 97), "Expected there to be 97 moves in the game"

assert(len(feats[0])) == 28, "Expected there to be 28 features per board position"

for i in range(28):
    assert feats[0][i].shape == (19, 19), f"Expected feature shape at index {i} to be (19, 19), but got {feats[0][i].shape}"


# Check ones
assert np.all(feats[0][3] == 1), "All elements in the ones feature plane must be 1"


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


assert (feats[18][0][3][6] == 0), "Black should no longer have stone here"
assert (feats[18][1][3][6] == 0), "White should not have stone here"
assert (feats[18][2][3][6] == 1), "There should be an empty spot here after capture"

assert (feats[17][20][4][6] == 1), "White should be able to capture 1 if playing here"
assert np.count_nonzero(feats[17][20] == 1) == 1, "White should only be able to capture 1"

for i in range(1, 8):
    assert np.all(feats[17][20+i] == 0), "White should not be able to capture any other amount of stones"


