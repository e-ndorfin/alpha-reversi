from typing import Optional, List, Tuple
import numpy as np


class Node:
    def __init__(self, parent: Optional['Node'] = None):
        self.state = None      # Board state of this node
        self.parent = parent   # Parent node
        self.children = []     # Legal child positions
        self.visits = 0        # Number of times this node has been visited
        self.value = 0.0       # Total value of this state
        # Player at this node (1 for black, -1 for white)
        self.player = None
        self.move = None       # Move that led to this state (start_pos, moves)
        self.moves_expanded = []  # Set of moves expanded so far
        # self.prior = prior  # Added prior probability from policy network

    def calculate_ucb_score(self, exploration_arg: float) -> float:
        """
        Calculates the UCB score for a child node.

        Args: 
            - exploration_arg: value that determines exploration/exploitation tradeoff 
        """

        if not self.parent:
            raise ValueError("No parent found for this child")

        if self.visits == 0:
            # Higher UCB is better; returning inf incentivizes the agent to choose this for exploration
            # Usually this will not be run however as _select will directly select children with visits == 0
            return float('inf')
        else:
            return self.value / self.visits \
                + exploration_arg * \
                np.sqrt(np.log(self.parent.visits) / self.visits)

    def add_child(self, child: 'Node') -> None:
        """Adds new child to children array"""

        self.children.append(child)
        # print(len(self.children))
        # for child in self.children:
        #     print(child.move)
        # print(self.children)
        if not child.parent:
            child.parent = self

    def update(self, game_value: float) -> None:
        """Updates the visit and value count of the node while backpropagating.

        Args:
            - game_value: 1.0 if win, 0.0 if loss, 0.5 if draw
        """

        self.visits += 1
        self.value += game_value

    # def __repr__(self):
    #     """
    #     Debugger pretty print node info
    #     """
    #     if hasattr(self, 'prior'):
    #         prior = "{0:.2f}".format(self.prior)
    #         return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visits, self.value)
    #     else:
    #         return "Count: {} Value: {}".format(self.visits, self.value)
