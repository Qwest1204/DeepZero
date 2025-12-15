from games.tictactoe import TicTacToe
from models.resnet import ResNet
from models.deepzero import DeepZeroParallel
import torch

game = TicTacToe()

device = torch.device("cpu")

model = ResNet(game, 4, 32, device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

player = 1

args = {
    'C': 2,
    'num_searches': 100,
    'num_iterations': 3,
    'num_parallel_games': 100,
    'batch_size': 16,
    'num_selfPlay_iterations': 350,
    'num_epochs': 4,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1
}

deepzero = DeepZeroParallel(model, optimizer, game, args)
deepzero.learn()
