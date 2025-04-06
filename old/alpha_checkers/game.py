import numpy as np
from typing import List, Tuple, Optional
from copy import deepcopy

# TODO: edit this to make it numpy and similar to https://github.com/foersterrobert/AlphaZeroFromScratch/blob/main/4.AlphaMCTS.ipynb
# https://www.youtube.com/watch?v=wuSQpLinRB4&ab_channel=freeCodeCamp.org


class Checkers:
    """
    Methods: 
    - [x] init 
    - [x] get initial state 
    - [ ] get next state (make move): state, action, player
    - [ ] get valid moves: state
    - [ ] check for win: state, action 
    - [ ] terminated / end of game: state
    - [ ] change perspective to opponent: state, player
    - [ ] return encoded state: state
    """

    def __init__(self) -> None:
        self.row_count = 8
        self.column_count = 8
        self.action_size = self.row_count * self.column_count

        self.piece = 1
        self.king = 2
        self.opponent_piece = -1
        self.opponent_king = -2

    def get_initial_state(self) -> np.ndarray:
        """Returns base state of the board as 8x8 numpy array."""

        state = np.zeros((self.row_count, self.column_count))
        # Black pieces (top)
        state[0:3:2, 1::2] = self.opponent_piece
        state[1, 0::2] = self.opponent_piece

        # White pieces (bottom)
        state[5::, 0::2] = self.piece
        state[6::2, 1::2] = self.piece

        return state

    def check_legal_move(self, action) -> bool:
        """Checks whether the action is within the game board."""
        return

    def get_next_state(self, state, action, player) -> np.ndarray:
        """Returns next state of the board given a action."""
        start_pos, end_pos = action
        intermediate_pos = ((end_pos[0] - start_pos[0]) // 2 + start_pos[0],
                            (end_pos[1] - start_pos[1]) // 2 + start_pos[1])

        if state[intermediate_pos] != 0:  # Capture


class CheckersGame:
    """
    A class representing a game of Checkers.

    Attributes:
        - board (np.ndarray): An 8x8 numpy array representing the board state.
        - current_player (int): The current player (1 for black, -1 for white).
        - moves_without_capture (int): Number of consecutive moves without a capture.

    Methods:
        - get_valid_moves() -> List[Tuple[Tuple[int, int], List[Tuple[int, int]]]]: Returns a list of valid moves.
        - make_move(start_pos: Tuple[int, int], moves: List[Tuple[int, int]]) -> None: Executes a move or a sequence of captures.
        - is_game_over() -> bool: Determines if the game is over.
        - get_winner() -> Optional[int]: Returns the winner (-1 for white, 1 for black, None if game is ongoing).
        - get_state() -> np.ndarray: Returns a deep copy of the board state.
    """

    # Board representation:
    # 0 = empty
    # 1 = black piece
    # 2 = black king
    # 3 = white piece
    # 4 = white king

    board: np.ndarray
    current_player: int
    moves_without_capture: int

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.current_player = 1  # 1 for black, -1 for white
        self.moves_without_capture = 0  # Counter for moves without capture

        self._initialize_board()

    def _initialize_board(self):
        """Initializes 8x8 board"""

        # Set up black pieces (top of board)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = 1

        # Set up white pieces (bottom of board)
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    self.board[row][col] = 3

    def get_valid_moves(self) -> List[Tuple[Tuple[int, int], List[Tuple[int, int]]]]:
        """Returns list of valid moves in format (start_pos, [capture_positions])"""
        moves = []
        capture_moves = []  # Jumps are mandatory

        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                # Check if piece belongs to current player
                if self._is_current_players_piece(piece):
                    piece_captures = self._get_capture_moves(
                        row, col)  # Check for possible captures
                    if piece_captures:
                        capture_moves.extend(((row, col), captures)
                                             for captures in piece_captures)
                    else:
                        piece_moves = self._get_normal_moves(row, col)
                        moves.extend(((row, col), [move])
                                     for move in piece_moves)

        return capture_moves if capture_moves else moves

    def _is_current_players_piece(self, piece: int) -> bool:
        if self.current_player == 1:  # Black's turn
            return piece in [1, 2]  # Black piece or king
        else:  # White's turn
            return piece in [3, 4]  # White piece or king

    def _get_normal_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Finds all pieces belonging to self.current_player and returns list of all possible non-capture moves"""
        moves = []
        piece = self.board[row][col]

        directions = []
        if piece in [1, 2]:  # Black piece or king
            directions.extend([(1, -1), (1, 1)])
        if piece in [3, 4]:  # White piece or king
            directions.extend([(-1, -1), (-1, 1)])
        if piece in [2, 4]:  # Kings can move backwards
            directions.extend([(-1, -1), (-1, 1)] if piece ==
                              2 else [(1, -1), (1, 1)])

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position(new_row, new_col) and self.board[new_row][new_col] == 0:
                moves.append((new_row, new_col))

        return moves

    def _get_capture_moves(self, row: int, col: int) -> List[List[Tuple[int, int]]]:
        """Returns list of capture moves."""

        captures = []
        self._find_capture_sequences(row, col, [], captures, ())
        return captures

    def _find_capture_sequences(self, row: int, col: int, current_sequence: List[Tuple[int, int]],
                                all_sequences: List[List[Tuple[int, int]]], last_visited: Tuple[int, int]) -> None:
        """Takes in the current position and sequence of jumps and finds all possible capture sequences. Used recursively, directly appends to sequences instead of returning anything."""

        directions = []
        piece = self.board[row][col]
        if piece in [2, 4]:  # Kings
            directions.extend([(2, -2), (2, 2), (-2, -2), (-2, 2)])
        elif piece == 1:  # Black piece
            directions.extend([(2, -2), (2, 2)])
        elif piece == 3:  # White piece
            directions.extend([(-2, -2), (-2, 2)])

        found_capture = False
        for dx, dy in directions:
            new_row, new_col = row + dx, col + dy  # End position after jump
            jumped_row, jumped_col = row + dx//2, col + dy//2  # Position jumped over

            # print(
            #     f"Current: ({row}, {col}), New: ({new_row}, {new_col}), Last Visited: {last_visited}, "
            #     f"Checks: {[(new_row, new_col) != last_visited, self._is_valid_position(new_row, new_col)]}"
            # )

            if ((new_row, new_col) != last_visited and  # Check if direction was not last move
                # Check if new space is on game board
                self._is_valid_position(new_row, new_col) and
                # Check if new space is empty
                self.board[new_row][new_col] == 0 and
                # Check if jumping over opponent's piece
                    self._is_opponent_piece(self.board[jumped_row][jumped_col])):

                # Add single jump directly
                all_sequences.append([(new_row, new_col)])

        # ===== DEPRECATED MULTICAPTURE LOGIC =====
        # directions = []
        # if piece in [2, 4]:  # Kings
        #     directions.extend([(2, -2), (2, 2), (-2, -2), (-2, 2)])
        # elif piece == 1:  # Black piece
        #     # Bottom left, bottom right respectively
        #     directions.extend([(2, -2), (2, 2)])
        # elif piece == 3:  # White piece
        #     # Top left, top right respectively
        #     directions.extend([(-2, -2), (-2, 2)])

        # found_capture = False
        # for dx, dy in directions:
        #     new_row, new_col = row + dx, col + dy  # End position after jump
        #     jumped_row, jumped_col = row + dx//2, col + dy//2  # Position jumped over

        #     print(
        #         f"Current: ({row}, {col}), New: ({new_row}, {new_col}), Last Visited: {last_visited}, "
        #         f"Checks: {[(new_row, new_col) != last_visited, self._is_valid_position(new_row, new_col)]}"
        #     )

        #     if ((new_row, new_col) != last_visited and  # Check if direction was not last move
        #         # Check if new space is on game board
        #         self._is_valid_position(new_row, new_col) and
        #         # Check if new space is empty
        #         self.board[new_row][new_col] == 0 and
        #         # Check if jumping over opponent's piece
        #             self._is_opponent_piece(self.board[jumped_row][jumped_col])):

        #         found_capture = True
        #         last_visited = (row, col)
        #         new_sequence = current_sequence + \
        #             [(new_row, new_col)]  # Add jump to sequence
        #         print("Found capture!", new_row, new_col, new_sequence,
        #               all_sequences, last_visited)
        #         self._find_capture_sequences(
        #             new_row, new_col, new_sequence, all_sequences, last_visited, piece)

        # if not found_capture and current_sequence:
        #     all_sequences.append(current_sequence)

    def _is_opponent_piece(self, piece: int) -> bool:
        if self.current_player == 1:  # Black's turn
            return piece in [3, 4]  # White pieces
        else:  # White's turn
            return piece in [1, 2]  # Black pieces

    def _is_valid_position(self, row: int, col: int) -> bool:
        return 0 <= row < 8 and 0 <= col < 8

    def make_move(self, start_pos: Tuple[int, int], moves: List[Tuple[int, int]]) -> None:
        """Make a move or sequence of captures"""
        row, col = start_pos
        piece = self.board[row][col]
        capture_made = False

        # Move the piece through the sequence
        self.board[row][col] = 0
        for i, (new_row, new_col) in enumerate(moves):

            # CAPTURE
            if abs(new_row - row) == 2:
                jumped_row = (new_row + row) // 2
                jumped_col = (new_col + col) // 2
                self.board[jumped_row][jumped_col] = 0
                capture_made = True

            # LAST MOVE
            if i == len(moves) - 1:  # Final position
                # Check if piece should be kinged
                if piece == 1 and new_row == 7:  # Black piece reaches bottom
                    piece = 2
                elif piece == 3 and new_row == 0:  # White piece reaches top
                    piece = 4
                self.board[new_row][new_col] = piece
            row, col = new_row, new_col

        # Update moves without capture counter
        if capture_made:
            self.moves_without_capture = 0
        else:
            self.moves_without_capture += 1

        self.current_player *= -1  # Switch players

    def is_game_over(self) -> bool:
        return len(self.get_valid_moves()) == 0 or self.moves_without_capture >= 50

    def get_winner(self) -> Optional[int]:
        if not self.is_game_over():
            return None
        if self.moves_without_capture >= 50:
            return 0  # Draw
        return -self.current_player  # Previous player won

    def get_state(self) -> np.ndarray:
        return self.board.deepcopy()

    def __str__(self) -> str:
        symbols = {0: ".", 1: "b", 2: "B", 3: "w", 4: "W"}
        board_str = "  0 1 2 3 4 5 6 7\n"
        for i in range(8):
            board_str += f"{i} "
            for j in range(8):
                board_str += symbols[self.board[i][j]] + " "
            board_str += "\n"
        return board_str
