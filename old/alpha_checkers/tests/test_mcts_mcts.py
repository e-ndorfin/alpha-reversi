import pytest
from copy import deepcopy
from pure_mcts.game import CheckersGame
from pure_mcts.mcts import MCTS
from pure_mcts.node import Node
import numpy as np
import math


class TestMCTS:
    @pytest.fixture
    def game(self):
        return CheckersGame()

    @pytest.fixture
    def mcts(self, game):
        return MCTS(game)

    def test_expansion(self, game, mcts):
        node = Node()
        game = CheckersGame()
        node.state = deepcopy(game.board)
        node.player = game.current_player

        _ = mcts._expand(node)

        assert len(node.moves_expanded) == len(game.get_valid_moves())

    def test_select_unvisited_node_first(self, mcts):
        root = Node()
        root.children = [Node() for _ in range(3)]
        for i in range(len(root.children)):
            root.children[i].visits = i  # Assigns 0, 1 and 2 as node.visits

        # All children have 0 visits initially
        selected = mcts._select(root)
        assert selected.visits == 0

    def test_expansion_creates_all_children(self, game, mcts):
        # Checks if expansion creates all children
        root = Node()
        root.state = deepcopy(game.board)
        root.player = game.current_player

        valid_moves = game.get_valid_moves()
        mcts._expand(root)

        assert len(root.children) == len(valid_moves)
        assert len(root.moves_expanded) == len(valid_moves)

    def test_expansion_skip_expanded_moves(self, game, mcts):
        # Checks if expansion ignores already expanded moves
        root = Node()
        root.state = deepcopy(game.board)
        root.player = game.current_player
        valid_moves = game.get_valid_moves()
        root.moves_expanded = [valid_moves[0]]  # Mark first move as expanded

        mcts._expand(root)
        print(root.children, valid_moves)
        assert len(root.children) == len(valid_moves) - 1

    def test_simulation_preserves_original_state(self, mcts):
        node = Node()
        game = CheckersGame()
        node.state = deepcopy(game.board)
        node.player = game.current_player
        original_state = deepcopy(node.state)
        mcts._simulate(node)
        assert np.array_equal(node.state, original_state)

    def test_backprop_updates_visits(self, mcts):
        root = Node()
        child = Node(parent=root)
        grandchild = Node(parent=child)

        mcts._backpropagate(grandchild, 1.0)
        assert root.visits == 1
        assert child.visits == 1
        assert grandchild.visits == 1

    def test_backprop_value_propagation(self, game, mcts):
        root = Node()  # Defaults to being black player
        root.state = deepcopy(game.board)
        root.player = game.current_player
        child = mcts._expand(root)
        grandchild = mcts._expand(child)

        # Original player perspective: 1.0 win
        mcts._backpropagate(grandchild, 1.0)
        assert root.value == 1.0  # Root sees win
        assert child.value == 0.0  # Child (opponent) sees loss
        assert grandchild.value == 1.0  # Grandchild should be a win for current player

    def test_terminal_node_expansion(self, mcts):
        # Create whether expanding terminal state should do nothing
        terminal_game = CheckersGame()
        terminal_game.board = np.zeros((8, 8), dtype=int)  # Empty board
        node = Node()
        node.state = terminal_game.board
        node.player = terminal_game.current_player

        assert mcts.is_terminal(node)
        mcts._expand(node)
        assert len(node.children) == 0
