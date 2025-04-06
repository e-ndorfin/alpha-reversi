import pytest
from pure_mcts.game import CheckersGame
import numpy as np


class TestCheckersGame:
    @pytest.fixture
    def game(self):
        return CheckersGame()

    def test_king_capture(self, game):
        game.board = np.zeros((8, 8), dtype=int)
        game.board[4][4] = 4
        game.board[3][5] = 1
        game.board[3][3] = 1
        game.board[5][3] = 1
        game.board[5][5] = 1
        game.current_player = -1

        assert np.array_equal(game.board, np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]))

        expected = [[(6, 6)], [(2, 2)], [(6, 2)], [(2, 6)]]
        assert game._get_capture_moves(4, 4).sort() == expected.sort()

    def test_normal_capture(self, game):
        game.board = np.zeros((8, 8), dtype=int)
        game.board[4][4] = 3
        game.board[3][5] = 1
        game.board[3][3] = 1
        game.current_player = -1

        assert np.array_equal(game.board, np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]
        ]))

        expected = [[(2, 2)], [(2, 6)]]
        assert game._get_capture_moves(4, 4).sort() == expected.sort()

    # def test_triple_alternating_capture(self, game):
    #     game.board = np.zeros((8, 8), dtype=int)
    #     game.board[6][6] = 3
    #     game.board[5][5] = 1
    #     game.board[3][5] = 1
    #     game.board[1][5] = 1
    #     game.current_player = -1

    #     assert np.array_equal(game.board, np.array([
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 3, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0]
    #     ]))

    #     expected = [[(4, 4), (2, 6), (0, 4)]]
    #     assert game._get_capture_moves(6, 6) == expected

    # def test_triple_line_capture(self, game):
    #     game.board = np.zeros((8, 8), dtype=int)
    #     game.board[6][6] = 3
    #     game.board[5][5] = 1
    #     game.board[3][3] = 1
    #     game.board[1][1] = 1
    #     game.current_player = -1

    #     assert np.array_equal(game.board, np.array([
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 3, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0]
    #     ]))

    #     expected = [[(4, 4), (2, 2), (0, 0)]]
    #     assert game._get_capture_moves(6, 6) == expected

    # def test_king_forward_backward(self, game):
    #     game.board = np.zeros((8, 8), dtype=int)
    #     game.board[6][6] = 4
    #     game.board[5][5] = 1
    #     game.board[5][3] = 1
    #     game.current_player = -1

    #     assert np.array_equal(game.board, np.array([
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 4, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0]
    #     ]))

    #     expected = [[(4, 4), (6, 2)]]
    #     assert game._get_capture_moves(6, 6) == expected

    # def test_king_loop(self, game):
    #     game.board = np.zeros((8, 8), dtype=int)
    #     game.board[5][6] = 4
    #     game.board[4][5] = 1
    #     game.board[4][3] = 1
    #     game.board[6][3] = 1
    #     game.board[6][5] = 1
    #     game.current_player = -1

    #     assert np.array_equal(game.board, np.array([
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 4, 0],
    #         [0, 0, 0, 1, 0, 1, 0, 0],
    #         [0, 0, 0, 0, 0, 0, 0, 0]
    #     ]))

    #     expected = [[(7, 4), (5, 2), (3, 4), (5, 6)], [
    #         (3, 4), (5, 2), (7, 4), (5, 6)]]
    #     assert game._get_capture_moves(5, 6) == expected
