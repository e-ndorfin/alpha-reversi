from src.game.othello import Othello
from src.model.model import ResNet
from src.model.selfplay import AlphaZero

import torch

if __name__ == "__main__":

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    othello = Othello()

    model = ResNet(othello, 4, 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    args = {
        "C": 2,
        "num_searches": 60,
        "num_iterations": 3,
        "num_self_play_iterations": 500,
        "num_epochs": 4,
        "batch_size": 256,
        "temperature": 1.25
    }

    alphaZero = AlphaZero(model, optimizer, othello, args)
    alphaZero.learn()
