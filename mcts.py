from typing import Tuple, List
import numpy as np
from game import CheckersGame
from node import Node


def ucb_score(self, parent: Node, child: Node) -> float: 
    """
    Calculates the UCB score for a child node.
    """
    if child.visits == 0: 
        return float('inf')
    return child.value / child.visits + np.sqrt(2 * np.log(parent.visits) / child.visits)

class MCTS:
    def __init__(self, game: CheckersGame):
        self.game = game
        # self.model = model
        # self.args = args
        # TODO: Add any additional initialization parameters
        
    def _select(self, node: Node) -> Node:
        
        max_ucb = float('-inf')
        
        for child in node.children: 
            if child.visits == 0: 
                return child 
            else: 
                if child.ucb_score(node, child) > max_ucb: 
                    max_ucb = child.ucb_score(node, child)
                    selected = child
        
        return selected 
        
    def _expand(self, node: Node) -> Node:
        parent = node.parent 
        state = node.state

        # TODO: Think about how to implement this. might need to add func to game.py to see next state given a current state, action and player
        
    def _simulate(self, state: np.ndarray) -> float:
        # TODO: Implement simulation/rollout phase
        pass
        
    def _backpropagate(self, node: Node, result: float) -> None:
        # TODO: Implement backpropagation phase
        pass 