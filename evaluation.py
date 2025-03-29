import numpy as np
import random
from game import CheckersGame
from mcts import MCTS
from timeit import default_timer as timer

class RandomAgent:
    """A simple random agent for playing Checkers."""
    
    def __init__(self, game: CheckersGame):
        self.game = game

    def get_move(self):
        """Selects a random valid move."""
        valid_moves = self.game.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None


def simulate_game(mcts_iterations: int = 1000) -> int:
    """Simulates a game between the MCTS agent and a random agent.

    Args:
        mcts_iterations: Number of iterations for the MCTS agent.

    Returns:
        1 if MCTS wins, -1 if random agent wins, 0 for draw.
    """
    game = CheckersGame()
    mcts_agent = MCTS(game)
    random_agent = RandomAgent(game)

    while not game.is_game_over():
        if game.current_player == 1:  # MCTS's turn (Black)
            start_pos, moves = mcts_agent.get_best_move(iterations=mcts_iterations)
            game.make_move(start_pos, moves)
        else:  # Random agent's turn (White)
            start_pos, moves = random_agent.get_move()
            if start_pos is None:
                break  # No valid moves, game over
            game.make_move(start_pos, moves)

    winner = game.get_winner()
    return winner  # 1 for MCTS win, -1 for random win, 0 for draw


def evaluate_mcts_vs_random(num_simulations: int, mcts_iterations: int = 1000) -> None:
    """Evaluates the MCTS agent against a random agent over multiple simulations.

    Args:
        num_simulations: Number of games to simulate.
        mcts_iterations: Number of iterations for the MCTS agent.
    """
    mcts_wins = 0
    random_wins = 0
    draws = 0

    for _ in range(num_simulations):
        result = simulate_game(mcts_iterations)
        if result == 1:
            mcts_wins += 1
        elif result == -1:
            random_wins += 1
        else:
            draws += 1

    print(f"Results after {num_simulations} simulations:")
    print(f"MCTS Wins: {mcts_wins} ({(mcts_wins / num_simulations) * 100:.2f}%)")
    print(f"Random Wins: {random_wins} ({(random_wins / num_simulations) * 100:.2f}%)")
    print(f"Draws: {draws} ({(draws / num_simulations) * 100:.2f}%)")


if __name__ == "__main__":
    num_simulations = 100  # Set the number of simulations
    evaluate_mcts_vs_random(num_simulations, mcts_iterations=1000)
