from games.tictactoe import TicTacToe
from models.resnet import ResNet
from models.deepzero import DeepZero
import torch

game = TicTacToe()

device = torch.device("cpu")

model = ResNet(game, 4, 64, device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

player = 1

args = {
    'C': 2,
    'num_search': 300,
    'num_iterations': 5,
    'batch_size': 64,
    'num_selfplay_iterations': 600,
    'num_epochs': 5,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

deepzero = DeepZero(model, optimizer, game, args)
deepzero.learn()
