import numpy as np
from games.checkers import Checkers
from models.mcts import MCTS
from models.resnet import ResNet
import torch

game = Checkers()
player = -1
device = torch.device("cpu")
args = {
    'C': 1.5,
    'num_searches': 600,
    'num_iterations': 60,
    'num_parallel_games': 128,
    'batch_size': 128,
    'num_selfPlay_iterations': 2000,
    'num_epochs': 10,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

model = ResNet(game, 24, 256, device=device)
model.load_state_dict(torch.load("weights/model.pt", map_location=device))
model.eval()
mcts = MCTS(game, args, model)
state = game.get_initial_state()

while True:
    game.print_board(state)
    if player == 1:
        valid_moves = game.get_valid_moves(state).flatten()
        print("val_movies", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}: "))

        if valid_moves[action] == 0:
            print("action not val")
            continue
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs, net_win_value = mcts.search(neutral_state)
        print("expected win rate", net_win_value)
        valid_moves = game.get_valid_moves(neutral_state)
        mcts_probs = mcts_probs * valid_moves  # Mask invalid moves to zero
        action = np.argmax(mcts_probs)
        action = game.flip_action(action)
    state = game.get_next_state(state, action, player)

    value, is_terminate = game.get_value_and_terminated(state, action)

    if is_terminate:
        if value == 1:
            print(player, "win")
        else:
             print(player, "lose")
        break

    player = game.get_opponent(player)