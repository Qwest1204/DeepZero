import numpy as np
from games.tictactoe import TickTackToe
from models.mcts import MCTS

tictactoe = TickTackToe()
player = 1

args = {
    'C': 1.41,
    'num_search': 100,
}

mcts = MCTS(tictactoe, args)

state = tictactoe.get_initial_state()

while True:
    print(state)
    if player == 1:
        valid_moves = tictactoe.get_valid_moves(state)
        print("val_movies", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}: "))

        if valid_moves[action] == 0:
            print("action not val")
            continue
    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)

    state = tictactoe.get_next_state(state, action, player)

    value, is_terminate = tictactoe.get_value_and_terminated(state, action)

    if is_terminate:
        if value == 1:
            print(player, "win")
        else:
            print(player, "lose")
        break

    player = tictactoe.get_opponent(player)