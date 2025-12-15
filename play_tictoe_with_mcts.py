import numpy as np
from games.tictactoe import TicTacToe
from models.mcts import MCTS
from models.resnet import ResNet
import torch

game = TicTacToe()
player = -1
device = torch.device("cpu")

args = {
    'C': 2,
    'num_search': 100,
    'num_iterations': 3,
    'num_parallel_games': 100,
    'batch_size': 16,
    'num_selfplay_iterations': 350,
    'num_epochs': 4,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}


model = ResNet(game, 4, 32, device=device)
model.load_state_dict(torch.load("weights/model_2_TicTacToe.pt", map_location=device))
model.eval()
mcts = MCTS(game, args, model)
state = game.get_initial_state()

while True:
    print(state)
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("val_movies", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}: "))
        if valid_moves[action] == 0:
            print("action not val")
            continue
    else:
        valid_moves = game.get_valid_moves(state)
        neutral_state = game.change_perspective(state, player)
        mcts_probs, net_win_value = mcts.search(neutral_state)
        print("expected win rate", net_win_value)
        mcts_probs = mcts_probs * valid_moves  # Mask invalid moves to zero
        action = np.argmax(mcts_probs)
        # Optional: Add a check for no valid moves, though this should not occur in a proper game state
        if valid_moves[action] == 0:
            raise ValueError("No valid moves available; game state may be invalid.")

    state = game.get_next_state(state, action, player)
    value, is_terminate = game.get_value_and_terminated(state, action)
    if is_terminate:
        if value == 1:
            print(player, "win")
        else:
            print(player, "lose")
        break
    player = game.get_opponent(player)