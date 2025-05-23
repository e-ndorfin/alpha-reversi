import torch
import torch.nn.functional as F
from tqdm import trange
import random

from src.model.tree import MCTS
import numpy as np


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        print(f"Using device {self.model.device}")
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        """
        Self play loop. Plays through one game and returns a list of tuples (state, action_probs, value).
        """
        memory = []
        player = 1
        state = self.game.get_initial_state()

        with torch.no_grad():
            while True:
                neutral_state = self.game.change_perspective(state, player)
                action_probs = self.mcts.search(neutral_state)

                memory.append((neutral_state, action_probs, player))

                try:
                    action_probs = action_probs ** (1 / self.args['temperature'])  # Higher temperature = more exploration (random)
                    action = np.random.choice(
                        self.game.action_size, p=action_probs)
                    state = self.game.get_next_state(state, action, player)
                except ValueError:
                    pass

                is_terminal = self.game.check_terminated(state)
                value = self.game.check_for_win(
                    state, player)

                if is_terminal:
                    returnMemory = []
                    for hist_neutral_state, hist_action_probs, hist_player in memory:
                        hist_outcome = value if hist_player == player else -value
                        returnMemory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    return returnMemory

                player = -player

    def train(self, memory):
        """
        Training loop. 
        """
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            sample = memory[batchIdx:min(
                len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(
                policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()  # Set model to eval mode so we don't do batch norms 
            for selfPlay_iteration in trange(self.args['num_self_play_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(),
                       f"optimizer_{iteration}.pt")
