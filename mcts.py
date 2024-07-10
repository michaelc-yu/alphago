
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

# Create the position info for a current game state
# this will be used when extracting features for this state
def create_position_dict(game, last_move):
    board_size = game.board_size
    position = {
        'board': game.board,
        'to_play': game.to_play,
        'board_size': board_size,
        'recency': {i: [[0 for _ in range(board_size)] for _ in range(board_size)] for i in range(1, 9)}
    }
    if last_move:
        lx, ly = last_move
        # Update the recency layers
        for i in range(board_size):
            for j in range(board_size):
                if position['recency'][7][i][j] == 1:
                    position['recency'][8][i][j] = 1
        for i in range(7, 1, -1):
            position['recency'][i] = position['recency'][i-1]

        position['recency'][1] = [[0 for _ in range(board_size)] for _ in range(board_size)]
        position['recency'][1][lx][ly] = 1
        for i in range(2, 9):
            position['recency'][i][lx][ly] = 0

    return position

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
        self.position = create_position_dict(game_state, move) if move else create_position_dict(game_state, None)
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

        # print(f"type move probabilities: {type(move_probabilities)}")
        top_k = 5
        top_moves_indices = torch.topk(move_probabilities, top_k).indices.squeeze().tolist()
        move_idx = random.choice(top_moves_indices)
        # print(f"move: {index_to_move(move_idx)}")

        while not self.game_state.move_is_legal(index_to_move(move_idx)):
            top_moves_indices.remove(move_idx)
            if len(top_moves_indices) == 0:
                top_k += 5
                top_moves_indices = top_moves_indices = torch.topk(move_probabilities, top_k).indices.squeeze().tolist()
            move_idx = random.choice(top_moves_indices)

        move = index_to_move(move_idx)
        # print(f"move: {move}")
        self.game_state.make_move(move)
        # print(f"game state to play: {self.game_state.to_play}")
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
            # print("select")
            while node.children:
                node = node.select_child()
            # Expansion
            # print("expand")
            node = node.expand()
            # Simulation
            # print("simulation")
            result = node.simulate()
            # Backpropagation
            # print("backprop")
            while node is not None:
                node.update(result)
                node.revert_last_move()
                node = node.parent

    def get_best_move(self):
        return max(self.root.children, key=lambda child: child.visits).move

    def reset_tree(self):
        self.root = Node(copy.deepcopy(self.root_game_state))

# Handle user making move
def user_move(game):
    valid = False
    while not valid:
        try:
            move = input("Enter your move (row, col): ")
            row, col = map(int, move.split(","))
            if game.move_is_legal((row, col)):
                valid = True
            else:
                print("Illegal move, try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter in the format row,col (e.g., 2,5).")
    return (row, col)


# Load the pretrained value and policy networks
valueNN = value.ValueNetwork(input_channels=28, k=48)
valueNN.load_state_dict(torch.load('value_network_model.pth'))
valueNN.eval()

policyNN = policy.PolicyNetwork(input_channels=28, k=48)
policyNN.load_state_dict(torch.load('policy_network_model.pth'))
policyNN.eval()

# Initialize the GoGame class and the Monte Carlo Tree Search class
game = go.GoGame(board_size=19, to_play=go.GoGame.BLACK)
mcts = MCTS(root_game_state=game, iterations=50)


# Game runs for 250 moves
for _ in range(250):
    if game.to_play == go.GoGame.BLACK:
        mcts.tree_search()
        move = mcts.get_best_move()
        print(f"AI plays at: {move}")
    else:
        move = user_move(game)
    game.make_move(move) # left to right, top to bottom. ex. (2, 5) is 3rd column, 6th row
    game.print_board()

    if game.to_play == go.GoGame.BLACK: # reset MCTS with the new game state if AI turn next
        mcts.root_game_state = game
        mcts.reset_tree()


# Get the final score and board state
(black_score, white_score), final_board = game.get_score()
print("Final board:")
game.print_final_board(final_board)
print(f"Black score: {black_score}, White score: {white_score}")

winner = go.GoGame.determine_winner(black_score, white_score)
print(f"{winner} wins by {abs(black_score - white_score)}")

