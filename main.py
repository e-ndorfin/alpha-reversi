from game import CheckersGame
from mcts import MCTS


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

    self_play = (input("Self play? Y/N") == 'Y')

    # Game loop
    while not game.is_game_over():
        print("\nCurrent board:")
        print(game)
        print(f"Current player: {'Black' if game.current_player == 1 else 'White'}")

        if not self_play:
            if game.current_player == 1:  # Human plays as black
                start_pos, moves = get_human_move(game)
            else:  # AI plays as white
                print("AI is thinking...")
                start_pos, moves = mcts.get_best_move(iterations=1000)
                print(f"AI move: {start_pos} -> {moves}")
        else:
            # Both players are AI
            print("AI is thinking...")
            start_pos, moves = mcts.get_best_move(iterations=1000)
            print(f"AI move: {start_pos} -> {moves}")

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
