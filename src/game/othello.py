import numpy as np
from typing import Optional, Tuple, List


class Othello:
    """
    Othello class for action and observation space.

    Note:
    - action defined as int from 0 to 63
    """

    def __init__(self) -> None:
        self.row_count = 8
        self.column_count = 8
        self.action_size = self.row_count * self.column_count

        self.white_piece = 1
        self.black_piece = -1

    def get_initial_state(self) -> np.ndarray:
        """Returns base state of the board as 8x8 numpy array."""

        state = np.zeros((self.row_count, self.column_count))
        state[3, 3] = state[4, 4] = self.white_piece
        state[3, 4] = state[4, 3] = self.black_piece
        return state

    def get_valid_moves(self, state, player) -> List:
        """Returns list of possible moves"""
        moves = []
        for action in range(64):
            row, col = action // 8, action % 8
            if state[row, col] != 0:
                continue
            else:
                if self._is_valid_move(state, action, player):
                    moves.append(action)

        return moves

    def _is_valid_move(self, state: np.ndarray, action: int, player: int) -> bool:
        """Vectorized move validation using numpy operations"""
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        row, col = action // 8, action % 8
        opponent = -player

        # Precompute all possible direction vectors
        max_steps = 7

        for i, (drow, dcol) in enumerate(directions):
            # Calculate maximum steps in this direction
            steps = np.arange(1, max_steps + 1)
            rows = row + drow * steps
            cols = col + dcol * steps

            # Create boundary mask
            valid_mask = (rows >= 0) & (rows < 8) & (cols >= 0) & (cols < 8)
            if not valid_mask.any():
                continue  # No valid positions in this direction

            # Get valid positions
            vrows = rows[valid_mask]
            vcols = cols[valid_mask]
            line = state[vrows, vcols]

            # Vectorized sequence checks
            opponent_mask = line == opponent
            player_mask = line == player

            # Find first non-opponent using cumulative product
            cum_opponent = np.cumprod(opponent_mask)
            if 1 not in (
                cum_opponent[0],
                cum_opponent[-1],
            ):  # As long as the edge tile is not non-opponent..
                continue

            first_non_opponent = (
                np.argmax(cum_opponent == 0) if np.any(cum_opponent == 0) else len(line)
            )

            # Check conditions:
            # 1. At least one opponent piece exists
            # 2. First non-opponent is player
            # 3. No gaps in opponent sequence
            if first_non_opponent > 0 and (
                first_non_opponent < len(line) and player_mask[first_non_opponent]
            ):
                return True

        return False

    def check_legal_move(self, state, action, player) -> bool:
        """Checks whether the action is within the game board."""

        if action < 0 or action > 63:
            return False

        row, col = action // 8, action % 8
        if state[row, col] != 0:
            return False

        return action in self.get_valid_moves(state, player)

    def get_next_state(self, state, action, player) -> Optional[np.ndarray]:
        """Returns new state after placing piece and flipping opponent's tiles. Assumes original state is valid."""

        if not self.check_legal_move(state, action, player):
            return state

        new_state = state.copy()
        action_row, action_col = action // 8, action % 8
        new_state[action_row, action_col] = player
        opponent = -player
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for drow, dcol in directions:
            current_row, current_col = action_row + drow, action_col + dcol
            to_flip = []

            while 0 <= current_row <= 7 and 0 <= current_col <= 7:
                if state[current_row, current_col] == opponent:
                    to_flip.append((current_row, current_col))
                    current_row += drow
                    current_col += dcol
                elif state[current_row, current_col] == player:
                    for row, col in to_flip:
                        new_state[row, col] = player
                    break
                else:  # Empty square
                    break

        return new_state

    def check_terminated(self, state) -> bool:
        """Returns if no possible moves for either player."""
        return (
            len(self.get_valid_moves(state, self.white_piece)) == 0
            and len(self.get_valid_moves(state, self.black_piece)) == 0
        )

    def check_for_win(self, state, player) -> Optional[int]:
        """Checks if player 1 won. 1 for win, -1 for lose, 0 for draw."""
        if self.check_terminated(state):
            count = state.sum() * player  # Calculates count for the player / opponent
            return np.sign(count)
        else:
            return None  # Game not over yet

    def get_encoded_state(self, state) -> np.ndarray:
        """Returns encoded state for the model."""

        encoded_state = np.stack((state == -1, state == 0, state == 1)).astype(
            np.float32
        )

        return encoded_state

    def change_perspective(self, state, player) -> np.ndarray:
        """Flips the perspective of the board for the other player."""
        return state * player

    def get_player(self, state, action) -> int:
        """Returns which player this move belongs to."""
        if not action:
            return 1
        else:
            row, col = action // 8, action % 8
            return state[row, col]


if __name__ == "__main__":
    board = Othello()
    state = board.get_initial_state()
    moves = board.get_valid_moves(state, 1)
    print(moves)
