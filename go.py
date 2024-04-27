
import numpy as np
import copy

BOARD = None

EMPTY = 0
BLACK = 1
WHITE = -1

class GoGame():
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    UNKNOWN = 2
    KOMI = 6.5

    def __init__(self, board_size=19, to_play=BLACK):
        self.board = np.zeros([board_size, board_size], dtype=np.int8)
        self.to_play = to_play
        self.board_size = board_size

    def make_move(self, move):
        x, y = move
        # print(f"board in make_move: {self.board}")
        # print(f"move: {move}")
        if self.board[x][y] != 0:
            raise ValueError("Invalid move: position occupied")
        self.board[x][y] = self.to_play
        # check if capture
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        for dir in dirs:
            dx, dy = x+dir[0], y+dir[1]
            if 0 <= dx < self.board_size and 0 <= dy < self.board_size:
                if self.board[dx][dy] == -self.to_play:
                    group, liberties = get_liberties(dx, dy, self.board)
                    if liberties == 0:
                        for i, j in group:
                            self.board[i][j] = 0
        self.to_play = -self.to_play
        return self

    def undo_move(self, move):
        x, y = move
        if self.board[x][y] == 0:
            raise ValueError("Cannot undo a move - position is empty")
        self.board[x][y] = 0

    def move_is_legal(self, move):
        x, y = move
        if move is None:
            return True
        if self.board[x][y] != self.EMPTY or self.is_suicide(move):
            return False
        return True

    def is_suicide(self, move):
        x, y = move
        self.board[x][y] = self.to_play
        _, liberties = get_liberties(x, y, self.board)
        self.board[x][y] = self.EMPTY
        opponent_color = self.BLACK if self.to_play == self.WHITE else self.WHITE
        for i, j in [(1,0),(-1,0),(0,1),(0,-1)]:
            dx, dy = x + i, y + j
            if 0 <= dx < self.board_size and 0 <= dy < self.board_size:
                if self.board[dx][dy] == opponent_color:
                    _, opp_liberties = get_liberties(dx, dy, self.board)
                    if opp_liberties == 1:
                        return False
        return liberties == 0

    def find_connected(self, start, board_):
        stack = [start]
        connected_empty = set()
        border_colors = set()
        while stack:
            x, y = stack.pop()
            if (x,y) in connected_empty:
                continue
            connected_empty.add((x,y))
            for i, j in [(1,0), (-1,0), (0,1), (0,-1)]:
                dx, dy = x+i, y+j
                if 0 <= dx < self.board_size and 0 <= dy < self.board_size:
                    if board_[dx][dy] == self.EMPTY:
                        stack.append((dx,dy))
                    else:
                        border_colors.add(board_[dx][dy])
        return connected_empty, border_colors

    def assign_territories(self, territories, border_colors, board_):
        territory_color = self.UNKNOWN
        if len(border_colors) == 1:
            territory_color = border_colors.pop()
        for x, y in territories:
            board_[x][y] = territory_color

    def score(self):
        board_copy = np.copy(self.board)
        for x in range(len(self.board)):
            for y in range(len(self.board)):
                if board_copy[x][y] == self.EMPTY:
                    territories, border_colors = self.find_connected((x,y), board_copy)
                    self.assign_territories(territories, border_colors, board_copy)

        black_score = np.count_nonzero(board_copy == self.BLACK)
        white_score = np.count_nonzero(board_copy == self.WHITE) + self.KOMI
        return (black_score, white_score), board_copy

    def get_board(self):
        return copy.deepcopy(self.board)

    def print_final_board(self, board):
        board_size = len(board)
        header = " "
        for i in range(board_size):
            header += f"{i:3}"
        print(header)

        for i in range(board_size):
            row_format = f"{i:2} "
            for cell in board[i]:
                marker = 'B' if cell == 1 else 'W' if cell == -1 else '.'
                row_format += f"{marker:3}"
            print(row_format)

    def print_board(self, board = None):
        board = self.board
        board_size = len(board)
        header = "    "
        for i in range(board_size):
            header += f"{i:<2}"
        print(header)
        
        for i in range(board_size):
            row_format = f"{i:<3} "
            for cell in board[i]:
                if cell == 1:
                    marker = '●' # black stone
                elif cell == -1:
                    marker = '○'  # white stone
                else:
                    marker = '.'
                row_format += f"{marker} "
            print(row_format)

def get_liberties(x, y, board):
    if board[x][y] == EMPTY:
        return [], -1
    board_size = len(board)
    color = board[x][y]
    visited = set()
    stack = [(x,y)]
    group = []
    liberties = 0
    while stack:
        cur_x, cur_y = stack.pop()
        if (cur_x, cur_y) in visited:
            continue
        visited.add((cur_x, cur_y))
        group.append((cur_x, cur_y))
        for i, j in [(1,0),(-1,0),(0,1),(0,-1)]:
            dx, dy = cur_x+i, cur_y+j
            if 0 <= dx < board_size and 0 <= dy < board_size:
                if board[dx][dy] == color and (dx, dy) not in visited:
                    stack.append((dx, dy))
                elif board[dx][dy] == EMPTY:
                    liberties += 1
    return group, liberties

