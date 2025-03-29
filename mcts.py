from typing import Tuple, List, Optional, Union
import numpy as np
import random
from copy import deepcopy
from game import CheckersGame
from node import Node
from mcts_debug import *


class MCTS:

    """
    Attributes:
        - game: CheckersGame class
        - exploration_arg: float value that determines exploration/exploitation balance
    """

    game: CheckersGame
    exploration_arg: float

    def __init__(self, game: CheckersGame):
        self.game = game  # NumPy checkers board defined in game.py
        self.exploration_arg = 1.414
        self.state = '' 

    def _select(self, node: Node) -> Node:
        """
        Returns a child of the current node to explore, based on UCB score. 

        Args: 
            - node: current node to explore
        """
        self.state = 'select'

        if not node.children:
            raise ValueError("No children found in the current node")

        max_ucb = float('-inf')

        for child in node.children:
            if child.visits == 0:
                return child

        selected_child, max_ucb = node.children[0], node.children[0].calculate_ucb_score(
            self.exploration_arg)

        for child in node.children[1:]:
            ucb_score = child.calculate_ucb_score(self.exploration_arg)
            if ucb_score > max_ucb:
                max_ucb = ucb_score
                selected_child = child

        return selected_child

    def _expand(self, node: Node) -> Optional[Node]:
        """
        Takes in a leaf node and adds a new child node with a random valid move.

        Args:
            - node: The leaf node to expand

        Returns:
            - A newly created child node, or None if no expansion is possible
        """
        self.state = 'expand'
        # Create a temporary game state from the node's state
        temp_game = CheckersGame()
        temp_game.board = deepcopy(node.state)
        temp_game.current_player = node.player

        # Get valid moves from this state
        valid_moves = temp_game.get_valid_moves()

        # Iterate through valid moves and create child nodes for unexpanded moves
        for move in valid_moves:
            if move not in node.moves_expanded:
                start_pos, moves = move

                # Create a new game state by applying the move
                new_game = CheckersGame()
                new_game.board = deepcopy(temp_game.board)
                new_game.make_move(start_pos, moves)

                # Create a new child node
                child = Node(parent=node)
                child.state = new_game.board
                child.player = new_game.current_player
                # Store the move that led to this state
                child.move = (start_pos, moves)

                # Add the child to the parent's children
                node.add_child(child)
                node.moves_expanded.append(move) # Add move to expanded moves

                # print('chosen', start_pos, moves)

                return child # Return the newly created child

        # If all moves have been expanded, return random child
        return random.choice(node.children)

    def _simulate(self, node: Node) -> float:
        """
        Simulates a random playout from the given node until a terminal state is reached.

        Args:
            - node: The node to start simulation from

        Returns:
            - 1.0 if the current player wins, 0.0 if they lose, 0.5 for a draw
        """

        self.state = 'simulate'
        # Create a temporary game for simulation
        temp_game = CheckersGame()
        temp_game.board = deepcopy(node.state)
        temp_game.current_player = node.player

        # Store the original player to determine the winner
        original_player = node.player

        # Simulate random moves until game is over
        while not temp_game.is_game_over():
            valid_moves = temp_game.get_valid_moves()
            if not valid_moves:
                break

            # Choose a random move
            start_pos, moves = random.choice(valid_moves)
            temp_game.make_move(start_pos, moves)

        # Determine the result
        winner = temp_game.get_winner()

        if winner is None:
            return 0.5  # Draw
        elif winner == original_player:
            return 1.0  # Win
        else:
            return 0.0  # Loss

    def _backpropagate(self, node: Node, result: float) -> None:
        """
        Updates the node and all its ancestors with the simulation result.

        Args:
            - node: The node to start backpropagation from
            - result: The simulation result (1.0 for win, 0.0 for loss, 0.5 for draw)
        """
        self.state = 'backpropagate'
        current = node
        while current is not None:
            current.update(result)
            # Flip the result for the parent node (opponent's perspective)
            result = 1.0 - result
            current = current.parent

    def get_best_move(self, iterations: int = 1000) -> Tuple[Tuple[int, int], List[Tuple[int, int]]]:
        """
        Runs the MCTS algorithm for a specified number of iterations and returns the best move.

        Args:
            - iterations: Number of MCTS iterations to run

        Returns:
            - The best move as (start_pos, moves)
        """
        # Create root node with current game state
        root = Node()
        root.state = deepcopy(self.game.board)
        root.player = self.game.current_player

        if len(self.game.get_valid_moves()) == 1:  # If only one move available return that one
            return self.game.get_valid_moves()[0]

        # Run MCTS for specified iterations on current game state
        for _ in range(iterations):
            self.simulation_loop(root)

        log_tree(root)
        save_tree_visualization(root, filename='mcts_tree')

        # If no children, return a random valid move
        if not root.children:
            return random.choice(self.game.get_valid_moves())

        # Choose child with highest visits == choosing child with highest UCT score
        # for node in root.children:
        #     print(node)

        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

    def simulation_loop(self, root: Node) -> None:
        """
        Performs one iteration of the MCTS algorithm: selection, expansion, simulation, and backpropagation.

        Args:
            - root: The root node to start the simulation from
        """
        # Selection: traverse the tree to find a leaf node
        node = root
        while node.children:
            node = self._select(node)

        # Expansion: if the leaf node is not terminal, expand it
        if not self.is_terminal(node) and len(node.moves_expanded) < len(self.game.get_valid_moves()):
            child = self._expand(node)
            if child:
                node = child

        # Simulation: perform a random playout from the leaf node
        result = self._simulate(node)

        # Backpropagation: update the node and its ancestors with the result
        self._backpropagate(node, result)

    def is_terminal(self, node: Node) -> bool:
        """
        Checks if a node represents a terminal state (game over).

        Args:
            - node: The node to check

        Returns:
            - True if the node is terminal, False otherwise
        """
        temp_game = CheckersGame()
        temp_game.board = deepcopy(node.state)
        temp_game.current_player = node.player
        return temp_game.is_game_over()
