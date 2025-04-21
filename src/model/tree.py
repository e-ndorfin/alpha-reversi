from ..game.othello import Othello
import numpy as np

import math
from typing import Optional
import torch


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = (
            prior  # Probability given by policy network when child is initiated
        )

        self.children = []

        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def calculate_ucb_score(self, child) -> int:
        """
        Calculates the UCB score for a child node.

        Args:
                        - exploration_arg: value that determines exploration/exploitation tradeoff
        """

        if child.visits == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value / child.visits) + 1) / 2
        return (
            q_value
            + self.args["C"]
            * (math.sqrt(self.visits) / (child.visits + 1))
            * child.prior
        )

    def select(self) -> Optional["Node"]:
        """
        Returns a child of the current node to explore, based on UCB score.
        """

        if not self.children:
            raise ValueError("No children found in the current node")
        else:
            best_child = None
            best_ucb = float("-inf")

            for child in self.children:
                ucb = self.calculate_ucb_score(child)
                if ucb > best_ucb:
                    best_child = child
                    best_ucb = ucb

            return best_child

    def expand(self, policy) -> Optional["Node"]:
        """
        Expands all possible moves as outlined in the policy

        Returns:
                - A randomly selected newly created child node, or None if no expansion is possible
        """

        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(
                    state=child_state, action=action, player=1
                )
                child_state = self.game.change_perspective(state=child_state, player=-1)

                child = Node(
                    self.game,
                    self.args,
                    child_state,
                    parent=self,
                    action_taken=action,
                    prior=prob,
                )
                self.children.append(child)

    def backpropagate(self, value) -> None:
        """
        Updates all values up until root node.
        """
        self.value += value
        self.visits += 1

        value = -value
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()  # Don't want to store gradients of these tensors
    def search(self, state) -> None:
        root = Node(self.game, self.args, state)

        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            # print(node.action_taken)

            value = self.game.check_for_win(
                node.state, self.game.get_player(node.state, node.action_taken)
            )

            if value:
                value = -value

            if not self.game.check_terminated(node.state):
                encoded_state = encoded_state = torch.tensor(
                    self.game.get_encoded_state(node.state)
                ).unsqueeze(
                    0
                )  # unsqueeze to make batch dimension

                policy, value = self.model(encoded_state)

                policy = (
                    torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                )  # Apply softmax to 64 neurons
                valid_moves = self.game.get_valid_moves(
                    node.state, self.game.get_player(node.state, node.action_taken)
                )
                valid_mask = np.zeros(64)
                valid_mask[valid_moves] = 1
                policy *= valid_mask
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visits
        action_probs /= np.sum(action_probs)  # Normalizing probabilities
        return action_probs
