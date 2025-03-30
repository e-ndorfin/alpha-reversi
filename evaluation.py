import numpy as np
import random
from game import CheckersGame
from mcts import MCTS
from timeit import default_timer as timer
from tqdm import tqdm
from datetime import datetime

class RandomAgent:
    """A simple random agent for playing Checkers."""
    
    def __init__(self, game: CheckersGame):
        self.game = game

    def get_move(self):
        """Selects a random valid move."""
        valid_moves = self.game.get_valid_moves()
        return random.choice(valid_moves) if valid_moves else None


def simulate_game(filepath: str, mcts_iterations: int = 1000) -> int:
    """Simulates a game between the MCTS agent and a random agent.

    Args:
        mcts_iterations: Number of iterations for the MCTS agent.

    Returns:
        1 if MCTS wins, -1 if random agent wins, 0 for draw.
    """
    game = CheckersGame()
    mcts_agent = MCTS(game)
    random_agent = RandomAgent(game)

    agent = random.choice([1, -1])  # 1 for MCTS, -1 for random agent

    while not game.is_game_over():
        if agent == 1:
            start_pos, moves = mcts_agent.get_best_move(iterations=mcts_iterations)
            game.make_move(start_pos, moves)
            agent *= -1
        else:  # Random agent's turn (White)
            start_pos, moves = random_agent.get_move()
            if start_pos is None:
                break  # No valid moves, game over
            game.make_move(start_pos, moves)
            agent *= -1

    winner = game.get_winner()

    with open(f"{filepath}.txt", 'a') as file:
        # Save the final board state
        file.write("Final Board State:\n")
        file.write(str(game))
        file.write("\n")
        
        # Save the game result
        result_text = "MCTS Win" if winner == 1 else "Random Win" if winner == -1 else "Draw"
        file.write(f"Game Result: {result_text}\n")

    return winner  # 1 for MCTS win, -1 for random win, 0 for draw


def evaluate_mcts_vs_random(filepath: str, num_simulations: int, mcts_iterations: int = 1000) -> None:
    """Evaluates the MCTS agent against a random agent over multiple simulations.

    Args:
        num_simulations: Number of games to simulate.
        mcts_iterations: Number of iterations for the MCTS agent.
    """
    mcts_wins = 0
    random_wins = 0
    draws = 0

    curr_date = datetime.now()

    with open(f"{filepath}.txt", 'a') as file:
        file.write("======================================\n")
        file.write(f"{curr_date}\n")

    for num_games_so_far in tqdm(range(num_simulations), desc="Simulating games"):
        result = simulate_game(filepath, mcts_iterations)
        if result == 1:
            mcts_wins += 1
        elif result == -1:
            random_wins += 1
        else:
            draws += 1

        with open(f"{filepath}.txt", 'a') as file:
            file.write(f"MCTS wins: {mcts_wins} ({(mcts_wins / (num_games_so_far + 1)) * 100:.2f}%) \n")

    print(f"Results after {num_simulations} simulations:")
    print(f"MCTS Wins: {mcts_wins} ({(mcts_wins / num_simulations) * 100:.2f}%)")
    print(f"Random Wins: {random_wins} ({(random_wins / num_simulations) * 100:.2f}%)")
    print(f"Draws: {draws} ({(draws / num_simulations) * 100:.2f}%)")

    with open(f"{filepath}.txt", "a") as file:
        file.write(f"\n \n {curr_date} to {datetime.now()}\n")
        file.write(f"Results after {num_simulations} simulations:\n")
        file.write(f"MCTS Wins: {mcts_wins} ({(mcts_wins / num_simulations) * 100:.2f}%)\n")
        file.write(f"Random Wins: {random_wins} ({(random_wins / num_simulations) * 100:.2f}%)\n")
        file.write(f"Draws: {draws} ({(draws / num_simulations) * 100:.2f}%)\n")



if __name__ == "__main__":
    num_simulations = 10  # Set the number of simulations
    evaluate_mcts_vs_random('exp1', num_simulations, mcts_iterations=100)
