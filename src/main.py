from src.game.othello import Othello
from src.model.model import ResNet, ResBlock
from src.model.tree import MCTS

import torch
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # print(np.__version__)
    # print(torch.__version__)

    # othello = Othello()

    # state = othello.get_initial_state()
    # moves = othello.get_valid_moves(state, 1)
    # state = othello.get_next_state(state, moves[0], 1)

    # encoded = othello.get_encoded_state(state)

    # tensor_state = torch.tensor(encoded).unsqueeze(0)  # Unsqueeze makes another dimension for a batch

    # args = {
    # 	'C': 2,
    # 	'num_searches': 1000
    # }

    # mcts = MCTS(othello, args)

    # model = ResNet(othello, 4, 64)

    # policy, value = model(tensor_state)
    # value = value.item()
    # policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()  # Not on the batch axis but on the neurons

    # plt.bar(range(othello.action_size), policy)
    # plt.show()
    # print(value, policy)

    othello = Othello()
    player = 1

    args = {"C": 2, "num_searches": 100}

    model = ResNet(othello, 4, 64)

    mcts = MCTS(othello, args, model)

    state = othello.get_initial_state()

    while True:
        print(state)

        if player == 1:
            valid_moves = othello.get_valid_moves(state, player)
            print(valid_moves)
            action = int(input(f"{player}:"))

            if action not in valid_moves:
                print("action not valid")
                continue

        else:
            neutral_state = othello.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

        state = othello.get_next_state(state, action, player)

        if othello.check_terminated(state):
            print(state)
            if othello.check_for_win(state, 1) == 1:
                print("player 1 won")
            elif othello.check_for_win(state, 1) == -1:
                print("player 2 won")
            else:
                print("draw")

        player = -player
