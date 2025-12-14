from models.mcts import MCTS
from models.resnet import ResNet
import torch
import torch.nn as nn
from tqdm import trange
import torch.nn.functional as F
import random
import numpy as np

class DeepZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)

    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1/self.args['temperature'])
            action = np.random.choice(self.game.action_size, p = action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminate = self.game.get_value_and_terminated(state, action)

            if is_terminate:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = (value if hist_player == player else self.game.get_opponent_value(value))
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))

                return returnMemory

            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) -1, batchIdx + self.args['batch_size'])]
            state, policy_target, value_target = zip(*sample)

            state, policy_target, value_target = np.array(state), np.array(policy_target), np.array(value_target).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_target = torch.tensor(policy_target, dtype=torch.float32, device=self.model.device)
            value_target = torch.tensor(value_target, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_target)
            value_loss = F.mse_loss(out_value, value_target)

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfplay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"weights/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"weights/optimizer_{iteration}_{self.game}..pt")