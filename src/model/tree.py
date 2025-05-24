from ..game.othello import Othello
from ..utils.constants import *

import numpy as np
from typing import Optional
import math
import torch
import logging
import os
from tqdm import trange

if LOGGING:
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(base_path, "logs")
    highest_num = 0

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Find the highest debug log number
    highest_num = 0
    for filename in os.listdir(log_dir):
        if filename.startswith("debug") and filename.endswith(".log"):
            try:
                num = int(filename[5:-4])  # Extract number between "debug" and ".log"
                highest_num = max(highest_num, num)
            except ValueError:
                continue

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"debug{highest_num + 1}.log")),
        ]
    )

    # Create a logger for this module
    logger = logging.getLogger(__name__)

    print(f"Logging to {log_dir}/debug{highest_num + 1}.log")
else:
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state  # Note this is state as seen by the player (meaning 1 is the current player, -1 is the opponent as opposed to P1 / P2)
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

    def expand(self, policy) -> None:
        """
        Expands all possible moves as outlined in the policy. Returns nothing.
        Note that this stores the state AS SEEN BY THE PLAYER.
        """

        if np.sum(policy) == 0:
            child_state = self.state.copy()
            child = Node(
                self.game,
                self.args,
                child_state,
                parent = self,
                action_taken=None
            )
            self.children.append(child)
        else:
            for action, prob in enumerate(policy):
                if prob > 0:
                    child_state = self.state.copy()
                    child_state = self.game.get_next_state(
                        state=child_state, action=action, player=1
                    )
                    child_state = self.game.change_perspective(
                        state=child_state, player=-1)

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
        self.model = model.to(torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"))

    @torch.no_grad()  # Don't want to store gradients of these tensors
    def search(self, state) -> None:
        logger.debug(f"=== STARTING SEARCH ===")
        root = Node(self.game, self.args, state)

        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu")
        
        encoded_state = torch.tensor(
            self.game.get_encoded_state(state),
            device=device
        ).unsqueeze(0)  # unsqueeze to make batch dimension

        logger.debug(f"Encoded state shape: {encoded_state.shape}")
        logger.debug(f"Encoded state device: {encoded_state.device}")

        policy, value = self.model(encoded_state)

        logger.debug(f"Raw policy from model: {policy}")
        logger.debug(f"Raw policy shape: {policy.shape}")
        logger.debug(f"Raw policy min/max: {policy.min()}/{policy.max()}")
        
        logger.debug(f"Raw value from model: {value}")
        logger.debug(f"Raw value contains NaN: {torch.isnan(value).any()}")

        policy = (
            torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        )  # Apply softmax to 64 neurons

        logger.debug(f"Policy after softmax: {policy}")
        logger.debug(f"Policy after softmax contains NaN: {np.isnan(policy).any()}")
        logger.debug(f"Policy after softmax contains Inf: {np.isinf(policy).any()}")
        logger.debug(f"Policy after softmax sum: {np.sum(policy)}")
        
        # Dirichlet noise
        dirichlet_noise = np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        logger.debug(f"Dirichlet noise: {dirichlet_noise}")
        logger.debug(f"Dirichlet noise contains NaN: {np.isnan(dirichlet_noise).any()}")
        
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * dirichlet_noise
        
        logger.debug(f"Policy after dirichlet: {policy}")
        logger.debug(f"Policy after dirichlet contains NaN: {np.isnan(policy).any()}")
        
        valid_moves = self.game.get_valid_moves(state, 1)

        logger.debug(f"Valid moves: {valid_moves}")
        logger.debug(f"Number of valid moves: {len(valid_moves)}")

        valid_mask = np.zeros(64)
        valid_mask[valid_moves] = 1
        logger.debug(f"Valid mask: {valid_mask}")
        
        policy *= valid_mask
        logger.debug(f"Policy after masking: {policy}")
        logger.debug(f"Policy after masking contains NaN: {np.isnan(policy).any()}")
        logger.debug(f"Policy after masking sum: {np.sum(policy)}")
        
        policy_sum = np.sum(policy)
        logger.debug(f"Policy sum before normalization: {policy_sum}")
        
        if policy_sum != 0:
            policy /= policy_sum
            logger.debug(f"Policy after normalization: {policy}")
        else:
            logger.debug("WARNING: Policy sum is zero, setting to zeros")
            policy = np.zeros_like(policy)

        logger.debug(f"Final policy for root expansion: {policy}")
        logger.debug(f"Final policy contains NaN: {np.isnan(policy).any()}")
        logger.debug(f"Final policy sum: {np.sum(policy)}")

        root.expand(policy)
        
        logger.debug(f"Policy: {policy}")

        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():  # Progress down tree until reaching leaf node
                node = node.select()

            value = self.game.check_for_win(
                node.state, 1
            )

            if value:
                value = -value

            valid_moves = self.game.get_valid_moves(
                node.state, 1
            )

            if not self.game.check_terminated(node.state):
                if not valid_moves:  # Edge case where opponent still has moves but we don't
                    logger.warning(
                        f"==================== NO VALID MOVES ====================")
                    logger.warning(node.state)
                    logger.warning(valid_moves)
                    policy = np.zeros(64)
                    value = 0
                else: 
                    encoded_state = torch.tensor(
                        self.game.get_encoded_state(node.state),
                        device=device
                    ).unsqueeze(0)

                    policy, value = self.model(encoded_state)
                    policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()


                    logger.debug(f"Valid moves: {valid_moves}")
                    valid_mask = np.zeros(64)
                    valid_mask[valid_moves] = 1
                    policy *= valid_mask
                    policy_sum = np.sum(policy)
                    if policy_sum != 0:
                        policy /= policy_sum
                    else:
                        policy = np.zeros_like(policy)
                    logger.debug(f"Policy after masking: {policy}")

                    value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visits
            
        logger.debug(f"Raw action_probs (visit counts): {action_probs}")
        logger.debug(f"Raw action_probs sum: {np.sum(action_probs)}")
        logger.debug(f"Raw action_probs contains NaN: {np.isnan(action_probs).any()}")
        
        total_visits = np.sum(action_probs)
        logger.debug(f"Total visits: {total_visits}")
        
        if total_visits > 0:
            action_probs /= total_visits
        else:
            logger.debug("ERROR: No visits recorded, returning zeros")
            action_probs = np.zeros_like(action_probs)
            
        logger.debug(f"Final action_probs: {action_probs}")
        logger.debug(f"Final action_probs contains NaN: {np.isnan(action_probs).any()}")
        logger.debug(f"Final action_probs sum: {np.sum(action_probs)}")
        logger.debug(f"=== ENDING SEARCH ===")
        
        return action_probs
