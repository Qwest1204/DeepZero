from games.checkers import Checkers
from models.resnet import ResNet
from models.deepzero import DeepZeroParallel
import torch

game = Checkers()

device = torch.device("cuda")

model = ResNet(game, 24, 256, device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

player = 1


args = {
    'C': 2,
    'num_searches': 400,
    'num_iterations': 10,
    'num_parallel_games': 200,
    'batch_size': 128,
    'num_selfPlay_iterations': 1000,
    'num_epochs': 10,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 1
}

deepzero = DeepZeroParallel(model, optimizer, game, args)
deepzero.learn()
