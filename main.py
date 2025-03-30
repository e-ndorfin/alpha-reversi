from game import CheckersGame
from mcts import MCTS
from timeit import default_timer as timer


def get_human_move(game: CheckersGame) -> tuple:
    """Get a move from human player"""
    print("\nValid moves:")
    valid_moves = game.get_valid_moves()
    for i, (start, moves) in enumerate(valid_moves):
        print(f"{i}: {start} -> {moves}")

    while True:
        try:
            choice = int(input("\nEnter move number: "))
            if 0 <= choice < len(valid_moves):
                return valid_moves[choice]
        except ValueError:
            pass
        print("Invalid choice, try again")


def main():
    game = CheckersGame()
    mcts = MCTS(game)

    self_play = (input("Self play? Y/N").lower() == 'y')

    ai1_iterations, ai2_iterations = 1000, 1000

    if self_play:
        ai1_iterations = int(input('How many iterations for first AI?'))
        ai2_iterations = int(input('How many iterations for second AI?'))

    curr = 1

    # Game loop
    while not game.is_game_over():
        print("\nCurrent board:")
        print(game)
        print(f"Current player: {'Black' if game.current_player == 1 else 'White'}")

        if not self_play:
            if game.current_player == 1:  # Human plays as black
                start_pos, moves = get_human_move(game)
            else:  # AI plays as white
                start = timer()
                print("AI is thinking...")
                start_pos, moves = mcts.get_best_move(iterations=1000)
                end = timer()
                print(f"AI move: {start_pos} -> {moves}")
                print(f"Took {round(end - start, 2)} seconds.")
        else:
            # Both players are AI
            start = timer()
            if curr == 1:
                print(f"{mcts.game.current_player} is thinking...")
                start_pos, moves = mcts.get_best_move(iterations=ai1_iterations)
                print(f"AI move: {start_pos} -> {moves}")
                curr *= -1
            elif curr == -1:
                print(f"{mcts.game.current_player} is thinking...")
                start_pos, moves = mcts.get_best_move(iterations=ai2_iterations)
                print(f"AI move: {start_pos} -> {moves}")
                curr *= -1
            end = timer()
            print(f"Took {round(end - start, 2)} seconds.")

        game.make_move(start_pos, moves)

    # Game over
    winner = game.get_winner()
    if winner == 1:
        print("\nBlack wins!")
    elif winner == -1:
        print("\nWhite wins!")
    else:
        print("\nDraw!")


if __name__ == "__main__":
    main()
