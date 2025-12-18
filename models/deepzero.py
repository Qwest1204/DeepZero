from models.mcts import MCTS, MCTSParallel
from datetime import datetime
from models.resnet import ResNet
import os
import torch
import torch.nn as nn
from tqdm import trange
import torch.nn.functional as F
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def mask_and_normalize(policy: np.ndarray, valid_moves: np.ndarray) -> np.ndarray:
    """
    Маскирует policy по valid_moves и нормализует.
    Гарантирует отсутствие NaN/Inf и деления на ноль.
    """
    policy = policy.astype(np.float64) * valid_moves.astype(np.float64)

    # Обнуляем NaN/Inf
    policy[~np.isfinite(policy)] = 0.0

    s = policy.sum()
    if s > 0.0:
        policy /= s
        return policy

    # Fallback: равномерно по валидным ходам
    v = valid_moves.astype(np.float64)
    v[~np.isfinite(v)] = 0.0
    s = v.sum()
    if s > 0.0:
        v /= s
        return v

    # Совсем нет ходов — вернём нули (терминальное состояние)
    return v


def normalize_probs(p: np.ndarray) -> np.ndarray:
    """
    Нормализация произвольного вектора вероятностей.
    1) NaN/Inf -> 0
    2) отрицательные -> 0
    3) если сумма <= 0 -> равномерное распределение
    4) гарантируем sum(p) == 1.0 (с поправкой на округление)
    """
    p = np.asarray(p, dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    p[p < 0] = 0.0

    s = p.sum()
    if s <= 0.0:
        p[:] = 1.0 / len(p)
        return p

    p /= s
    # Коррекция накопленной погрешности, чтобы сумма была ровно 1.0
    diff = 1.0 - p.sum()
    if abs(diff) > 1e-12:
        p[np.argmax(p)] += diff

    return p

class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None

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

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs = temperature_action_probs / temperature_action_probs.sum()
            action = np.random.choice(self.game.action_size,
                                      p=temperature_action_probs)  # Divide temperature_action_probs with its sum in case of an error

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
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
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args[
                'batch_size'])]  # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")


class DeepZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            # Собираем ОРИГИНАЛЬНЫЕ states
            states = np.stack([spg.state for spg in spGames])

            # Меняем перспективу для MCTS
            # neutral_states: текущий игрок видит свои фигуры как положительные
            neutral_states = self.game.change_perspective(states, player)

            # MCTS работает с neutral_states (всегда как будто ходит player=1)
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames) - 1, -1, -1):
                spg = spGames[i]

                # action_probs из MCTS - в координатах neutral_state
                action_probs = np.zeros(self.game.action_size, dtype=np.float64)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count

                visits_sum = action_probs.sum()
                neutral_state_i = neutral_states[i]

                if visits_sum > 0:
                    action_probs /= visits_sum
                else:
                    valid_moves = self.game.get_valid_moves(neutral_state_i)
                    if valid_moves.sum() == 0:
                        value, is_terminal = self.game.get_value_and_terminated(
                            neutral_state_i, None
                        )
                        if not is_terminal:
                            value, is_terminal = 0, True

                        for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                            hist_outcome = value if hist_player == player else \
                                self.game.get_opponent_value(value)
                            return_memory.append((
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome
                            ))
                        del spGames[i]
                        continue

                    action_probs = valid_moves.astype(np.float64)
                    action_probs = normalize_probs(action_probs)

                # Сохраняем neutral_state и action_probs (в neutral координатах)
                # spg.root.state = neutral_states[i] (из MCTS)
                spg.memory.append((neutral_state_i.copy(), action_probs.copy(), player))

                # Температурная выборка
                temperature_action_probs = action_probs ** (1.0 / self.args['temperature'])
                temperature_action_probs = normalize_probs(temperature_action_probs)

                # Выбираем action в NEUTRAL координатах
                neutral_action = np.random.choice(
                    self.game.action_size,
                    p=temperature_action_probs
                )

                # ===============================================
                # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ!
                # neutral_action - это ход в координатах neutral_state
                # Для применения к ОРИГИНАЛЬНОМУ state нужно:
                # - если player=1 (белые): action = neutral_action (без изменений)
                # - если player=-1 (чёрные): action = flip(neutral_action)
                # ===============================================
                if player == 1:
                    action = neutral_action
                else:
                    action = self.game.flip_action(neutral_action)

                # Применяем к ОРИГИНАЛЬНОМУ state
                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else \
                            self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])]
            if len(sample) == 0:
                continue

            state, policy_targets, value_targets = zip(*sample)

            state = np.array(state)
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(
                    self.args['num_selfPlay_iterations'] // self.args['num_parallel_games'],
                    desc="Self-play"
            ):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs'], desc="Training"):
                self.train(memory)

            torch.save(self.model.state_dict(), f"weights/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"weights/optimizer_{iteration}_{self.game}.pt")

