from games.chess import Chess
from models.resnet import ResNet
from models.deepzero import DeepZeroParallel
import torch

game = Chess()

device = torch.device("cuda")

model = ResNet(game, 24, 512, device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

args = {
    'C': 1.5,
    'num_searches': 800,
    'num_iterations': 60,
    'num_parallel_games': 128,
    'batch_size': 128,
    'num_selfPlay_iterations': 2000,
    'num_epochs': 10,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

deepzero = DeepZeroParallel(model, optimizer, game, args)
deepzero.learn()
