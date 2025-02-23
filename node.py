from typing import Optional, List, Tuple
import numpy as np


class Node:
    def __init__(self, parent: Optional['Node'] = None):
        self.state = None     # Board state as this node
        self.parent = parent   # Parent node
        self.children = []     # Legal child positions
        self.visits = 0        # Number of times this node has been visited
        self.value = 0.0       # Total value of this state
        # self.prior = prior  # Added prior probability from policy network

    def value_calculation(self) -> Optional[float]:
        """ 
        Calculates the value of this node.
        """
        if self.visits == 0:
            return 0
        else:
            return self.value / self.visits

    def select_move(self, temperature: float) -> Tuple[int, int]:
        """
        Selects the move with the highest UCB score.
        """
        visits = [child.visits for child in self.children]

        if temperature == 0:  # Greedy selection
            action = self.children[np.argmax(visits)]
        elif temperature == float('inf'):  # Random (exploratory) selection
            action = np.random.choice(self.children)
        else:  # Temperature-based selection
            visit_counts = np.array([child.visits for child in self.children])
            distribution = visit_counts ** (1 / temperature)
            distribution = distribution / distribution.sum()
            action = np.random.choice(len(self.children), p=distribution)

        return self.children[action]

    def best_child(self) -> 'Node':
        """
        Select child with highest UCB score.
        """
        if not self.children:
            raise ValueError("Node has no children")
        return max(self.children, key=lambda c: c.ucb_score(self, c))

    # def expand(self, state: np.ndarray, to_play: int, action_probs: List[float]) -> 'Node':
    #     """
    #     We expand a node and keep track of the prior policy probability given by neural network
    #     """
    #     self.to_play = to_play
    #     self.state = state
    #     for a, prob in enumerate(action_probs):
    #         if prob != 0:
    #             self.children[a] = Node(prior=prob, to_play=self.to_play * -1)

    def __repr__(self):
        """
        Debugger pretty print node info
        """
        prior = "{0:.2f}".format(self.prior)
        return "{} Prior: {} Count: {} Value: {}".format(self.state.__str__(), prior, self.visit_count, self.value())
