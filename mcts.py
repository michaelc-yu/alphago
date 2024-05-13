
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import sqrt, log
import go

import extract_features
import value
import policy

import copy


"""
policy_network should influence which nodes to explore during expansion
value_network replaces traditional rollout simulations
"""


def index_to_move(index, board_size=19):
    """Convert a flat index to 2D board coordinates."""
    return (index // board_size, index % board_size)


class Node():
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = copy.deepcopy(game_state)
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.move_history = []

    def select_child(self):
        res = max(self.children, key=lambda c: c.wins / c.visits + np.sqrt(2 * np.log(self.visits) / c.visits))
        return res

    def expand(self):
        """Expand the node by making a move, using policy network to suggest moves."""
        feats = extract_features.get_features_from_position(self.position)
        features_tensor = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            logits = policyNN(features_tensor)
            move_probabilities = torch.softmax(logits, dim=1)

        # move = get_next_move(move_probabilities)
        move = (1, 5) # hardcode for now

        self.game_state.make_move(move)

        self.move_history.append(move)
        new_state = copy.deepcopy(self.game_state)
        child_node = Node(new_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def revert_last_move(self):
        if self.move_history:
            last_move = self.move_history.pop()
            self.game_state.undo_move(last_move)

    def simulate(self):
        """Use value network to estimate the value of the new board state."""
        feats = extract_features.get_features_from_position(self.position)
        features_tensor = torch.tensor(feats, dtype=torch.float32)
        with torch.no_grad():
            output = valueNN(features_tensor)
        return output

    def update(self, result):
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent

class MCTS():
    def __init__(self, root_game_state, iterations=10):
        self.root = Node(root_game_state)
        self.iterations = iterations

    def tree_search(self):
        for _ in range(self.iterations):
            node = self.root
            # Selection
            print("select")
            while node.children:
                node = node.select_child()
            # Expansion
            print("expand")
            node = node.expand()
            # Simulation
            print("simulation")
            result = node.simulate()
            # Backpropagation
            print("backprop")
            while node is not None:
                node.update(result)
                node.revert_last_move()
                node = node.parent

    def get_best_move(self):
        return max(self.root.children, key=lambda child: child.visits).move

    def reset_tree(self):
        self.root = Node(copy.deepcopy(self.root_game_state))



valueNN = value.ValueNetwork(input_channels=28, k=48)
valueNN.load_state_dict(torch.load('value_network_model.pth'))
valueNN.eval()

policyNN = policy.PolicyNetwork(input_channels=28, k=48)
policyNN.load_state_dict(torch.load('policy_network_model.pth'))
policyNN.eval()


