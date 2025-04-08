from src.game.othello import Othello
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class TestOthello:
    @pytest.fixture
    def game(self):
        return Othello()

    @pytest.fixture
    def initial_state(self, game):
        return game.get_initial_state()

    def test_initial_state(self, game, initial_state):
        """Test that initial state is correctly set up"""
        expected = np.zeros((8, 8))
        expected[3, 3] = expected[4, 4] = 1  # white pieces
        expected[3, 4] = expected[4, 3] = -1  # black pieces
        assert np.array_equal(initial_state, expected)

    def test_valid_moves_initial_state(self, game, initial_state):
        """Test valid moves for black player in initial state"""
        valid_moves = game.get_valid_moves(initial_state, -1)  # black's turn
        # (row*8 + col) for valid initial moves
        expected_moves = [19, 26, 37, 44]
        assert set(valid_moves) == set(expected_moves)

    def test_invalid_move(self, game, initial_state):
        """Test that invalid moves are rejected"""
        assert not game.check_legal_move(
            initial_state, 0, -1)  # corner is invalid
        assert not game.check_legal_move(
            initial_state, 28, -1)  # existing piece

    def test_move_execution(self, game, initial_state):
        """Test that a move properly flips opponent pieces"""
        # Make move at position 19 (row 2, col 3)
        new_state = game.get_next_state(initial_state, 19, -1)

        # Check piece was placed
        assert new_state[2, 3] == -1

        # Check piece was flipped (3,3 should now be black)
        assert new_state[3, 3] == -1

    def test_game_termination(self, game):
        """Test game termination detection"""
        # Create a nearly full board
        state = np.full((8, 8), 1)  # all white
        state[0, 1] = -1  # one black piece
        state[0, 0] = 0  # one empty space

        # Game shouldn't be over yet
        assert not game.check_terminated(state)

        # Fill the last space
        state[0, 1] = 1
        assert game.check_terminated(state)

    def test_win_detection(self, game):
        """Test win/lose/draw detection"""
        # White wins
        white_win = np.full((8, 8), 1)
        assert game.check_for_win(white_win, 1) == 1
        assert game.check_for_win(white_win, -1) == -1

        # Black wins
        black_win = np.full((8, 8), -1)
        assert game.check_for_win(black_win, 1) == -1
        assert game.check_for_win(black_win, -1) == 1

        # Draw
        draw_state = np.zeros((8, 8))
        assert game.check_for_win(draw_state, 1) == 0
        assert game.check_for_win(draw_state, -1) == 0

    def test_edge_case_moves(self, game):
        """Test moves at edges of board"""
        state = np.zeros((8, 8))
        state[0, 0] = -1  # black in corner
        state[0, 1] = 1   # white adjacent

        # Placing black at (0,2) should flip white piece
        new_state = game.get_next_state(state, 2, -1)
        assert new_state[0, 1] == -1
