from games.connectfour import ConnectFour
from models.resnet import ResNet
from models.deepzero import DeepZeroParallel
import torch

game = ConnectFour()

device = torch.device("cpu")

model = ResNet(game, 9, 128, device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

player = 1

args = {
    'C': 2,
    'num_search': 500,
    'num_iterations': 10,
    'num_parallel_games': 300,
    'batch_size': 32,
    'num_selfplay_iterations': 500,
    'num_epochs': 10,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1
}

deepzero = DeepZeroParallel(model, optimizer, game, args)
deepzero.learn()
