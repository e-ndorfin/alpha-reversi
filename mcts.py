from typing import Tuple, List
import numpy as np
from game import CheckersGame
from node import Node


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

    def _select(self, node: Node) -> Node:
        """
        Returns a child of the current node to explore, based on UCB score. 

        Args: 
            - node: current node to explore
        """

        if not node.children:
            raise ValueError("No children found in the current node")

        max_ucb = float('-inf')

        for child in node.children:
            if child.visits == 0:
                return child

        selected_child, max_ucb = node.children[0], node.children[0].ucb_score(self.exploration_arg)

        for child in node.children[1:]:
            ucb_score = child.ucb_score(child, self.exploration_arg)
            if ucb_score > max_ucb:
                max_ucb = ucb_score
                selected_child = child

        return selected_child

    def _expand(self, node: Node) -> Node:
        """
        Takes in leaf node (has no children) and adds new node to node.children with random move
        """

        if 

        parent = node.parent
        board_state = self.game.get_state
        moves = self.game.get_valid_moves()






        # TODO: Think about how to implement this. might need to add func to game.py to see next state given a current state, action and player

    def _simulate(self, state: np.ndarray) -> float:
        # TODO: Implement simulation/rollout phase
        pass

    def _backpropagate(self, node: Node, result: float) -> None:
        # TODO: Implement backpropagation phase
        pass


    def simulation_loop(self) -> None:

        # Unless reached leaf node, continue choosing 
        pass 
