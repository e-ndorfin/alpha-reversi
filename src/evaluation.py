import numpy as np
import os
import datetime
from src.game.othello import Othello
from src.model.tree import MCTS
from src.model.model import ResNet
import torch


def format_board(state):
    """Format the board state into a human-readable string"""
    symbols = {0: ".", 1: "X", -1: "O"}
    rows = []
    for row in state:
        formatted_row = " ".join(symbols[cell] for cell in row)
        rows.append(formatted_row)
    return "\n".join(rows)


def evaluate_model_vs_random(num_episodes=10):
    game = Othello()
    args = {"C": 2, "num_searches": 100}
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    print(device)

    # Initialize model and MCTS
    model = ResNet(game, 4, 64).to(device)
    model.eval()
    mcts = MCTS(game, args, model)

    results = {
        "model_wins": 0,
        "random_wins": 0,
        "draws": 0
    }

    os.makedirs('runs', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"runs/evaluation_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write(f"Evaluation Run ({timestamp})\n")
        f.write(f"Episodes: {num_episodes}\n\n")

        for episode in range(num_episodes):
            # Randomly assign players
            model_player = np.random.choice([1, -1])
            random_player = -model_player

            state = game.get_initial_state()
            current_player = 1

            while True:
                if current_player == model_player:
                    # Model's turn - ensure tensor is on correct device
                    neutral_state = game.change_perspective(
                        state, model_player)
                    mcts_probs = mcts.search(neutral_state)
                    action = np.argmax(mcts_probs)
                else:
                    # Random agent's turn
                    valid_moves = game.get_valid_moves(state, random_player)
                    action = np.random.choice(
                        valid_moves) if valid_moves else None

                if action is None:
                    break  # No valid moves, game continues

                state = game.get_next_state(state, action, current_player)

                if game.check_terminated(state):
                    winner = game.check_for_win(state, model_player)

                    # Write to single file
                    f.write(f"Episode {episode+1}\n")
                    f.write(f"Model played as: {model_player}\n")
                    f.write(f"Random played as: {random_player}\n")

                    # Determine result text
                    if winner == model_player:
                        result_text = "Model win"
                        results["model_wins"] += 1
                    elif winner == random_player:
                        result_text = "Random win"
                        results["random_wins"] += 1
                    else:
                        result_text = "Draw"
                        results["draws"] += 1

                    f.write(f"Result: {result_text}\n")
                    f.write("Final board state:\n")
                    f.write(format_board(state))
                    f.write("\n\n")  # Separate episodes with blank lines

                    print(f"Episode {episode+1} complete")
                    break

                current_player = -current_player

            print(f"Episode {episode+1}/{num_episodes} complete")

    print("\nEvaluation results:")
    print(
        f"Model wins: {results['model_wins']}/{num_episodes} ({results['model_wins']/num_episodes:.1%})")
    print(
        f"Random wins: {results['random_wins']}/{num_episodes} ({results['random_wins']/num_episodes:.1%})")
    print(
        f"Draws: {results['draws']}/{num_episodes} ({results['draws']/num_episodes:.1%})")
    print(f"\nFull results saved to: {filename}")


if __name__ == "__main__":
    evaluate_model_vs_random(num_episodes=10)
