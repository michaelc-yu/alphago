
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sgfmill import sgf

import go


# for each move in each sgf file
# we want the (position, subsequent move) pair as (input, output) for our policy network
# position: we need a single instance of GoGame, and in each move we save the board size, board position, to play, etc.
# move: the move is used as the ground truth label for our policy network

# for each move in each sgf file
# we want (position, who won) pair as input, output for our value network
# position: we need a single instance of GoGame, and in each move we save the board size, board position, to play, etc.
# winner: the winner is used as the ground truth label for our value network


def process_sgf_file(filepath):
    with open(filepath, "rb") as file:
        sgf_content = sgf.Sgf_game.from_bytes(file.read())

    root_node = sgf_content.get_root()
    board_size = root_node.get_size()

    game = go.GoGame(board_size=board_size, to_play=go.GoGame.BLACK)

    dirs = [(1,0),(-1,0),(0,1),(0,-1)]

    positions = []
    position = {}

    # Iterate through the game moves
    for node in sgf_content.main_sequence_iter():
        move = node.get_move()
        if move[0] is None or move[1] is None:
            # print("[WARNING] skipping")
            continue
        player, (x, y) = move
        x = 18-x
        player_int = go.BLACK if player == 'b' else go.WHITE

        position['x'] = x
        position['y'] = y
        position['to_play'] = player_int
        position['board'] = game.get_board()
        position['board_size'] = board_size

        positions.append(position)

        print(f"{player} plays at {(x, y)}")
        game.board[x, y] = player_int

        for dir in dirs:
            dx, dy = x+dir[0], y+dir[1]
            if 0 <= dx < board_size and 0 <= dy < board_size:
                if game.board[dx][dy] == (-1 * player_int): # opponent stone
                    group, liberties = go.get_liberties(dx, dy, game.board)
                    if liberties == 0:
                        print("captured group")
                        print(group)
                        print(f"board: {game.board}")
                        for gx, gy in group:
                            game.board[gx][gy] = go.EMPTY

    game_result = root_node.get("RE")
    return positions, game_result


