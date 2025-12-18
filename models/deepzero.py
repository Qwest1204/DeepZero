from models.mcts import MCTS, MCTSParallel
from models.resnet import ResNet
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
    def __init__(self, model, optimizer, game, args, log_dir=None):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

        # TensorBoard setup
        if log_dir is None:
            log_dir = f"runs/{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.train_step = 0

        # Метрики для отслеживания
        self.game_lengths = []
        self.game_outcomes = []  # 1 = белые выиграли, -1 = чёрные, 0 = ничья

        # Логируем гиперпараметры
        self.writer.add_text('hyperparameters', str(args), 0)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args['num_parallel_games'])]
        move_counts = [0] * len(spGames)

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames) - 1, -1, -1):
                spg = spGames[i]
                move_counts[i] += 1

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

                        # Записываем метрики игры
                        self.game_lengths.append(move_counts[i])
                        self.game_outcomes.append(value if player == 1 else -value)

                        for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                            hist_outcome = value if hist_player == player else \
                                self.game.get_opponent_value(value)
                            return_memory.append((
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome
                            ))
                        del spGames[i]
                        del move_counts[i]
                        continue

                    action_probs = valid_moves.astype(np.float64)
                    action_probs = normalize_probs(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1.0 / self.args['temperature'])
                temperature_action_probs = normalize_probs(temperature_action_probs)

                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    # Записываем метрики игры
                    self.game_lengths.append(move_counts[i])
                    self.game_outcomes.append(value if player == 1 else -value)

                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else \
                            self.game.get_opponent_value(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]
                    del move_counts[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        num_batches = 0

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
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

            # Gradient clipping (опционально, но полезно)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Накапливаем метрики
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

            # Логируем каждый батч
            self.writer.add_scalar('Train/PolicyLoss_batch', policy_loss.item(), self.train_step)
            self.writer.add_scalar('Train/ValueLoss_batch', value_loss.item(), self.train_step)
            self.writer.add_scalar('Train/TotalLoss_batch', loss.item(), self.train_step)
            self.train_step += 1

        # Возвращаем средние значения
        if num_batches > 0:
            return (total_policy_loss / num_batches,
                    total_value_loss / num_batches,
                    total_loss / num_batches)
        return 0, 0, 0

    def log_selfplay_metrics(self, iteration):
        """Логирует метрики self-play"""
        if len(self.game_lengths) > 0:
            avg_length = np.mean(self.game_lengths)
            min_length = np.min(self.game_lengths)
            max_length = np.max(self.game_lengths)

            self.writer.add_scalar('SelfPlay/AvgGameLength', avg_length, iteration)
            self.writer.add_scalar('SelfPlay/MinGameLength', min_length, iteration)
            self.writer.add_scalar('SelfPlay/MaxGameLength', max_length, iteration)
            self.writer.add_histogram('SelfPlay/GameLengthDist', np.array(self.game_lengths), iteration)

        if len(self.game_outcomes) > 0:
            outcomes = np.array(self.game_outcomes)
            white_wins = np.sum(outcomes == 1) / len(outcomes)
            black_wins = np.sum(outcomes == -1) / len(outcomes)
            draws = np.sum(outcomes == 0) / len(outcomes)

            self.writer.add_scalar('SelfPlay/WhiteWinRate', white_wins, iteration)
            self.writer.add_scalar('SelfPlay/BlackWinRate', black_wins, iteration)
            self.writer.add_scalar('SelfPlay/DrawRate', draws, iteration)

        # Очищаем метрики для следующей итерации
        self.game_lengths = []
        self.game_outcomes = []

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            print(f"\n{'=' * 50}")
            print(f"Iteration {iteration + 1}/{self.args['num_iterations']}")
            print(f"{'=' * 50}")

            memory = []

            # Self-play фаза
            self.model.eval()
            print("\nSelf-play phase:")
            for selfPlay_iteration in trange(
                    self.args['num_selfPlay_iterations'] // self.args['num_parallel_games'],
                    desc="Self-play"
            ):
                memory += self.selfPlay()

            # Логируем метрики self-play
            self.log_selfplay_metrics(iteration)
            self.writer.add_scalar('SelfPlay/MemorySize', len(memory), iteration)

            # Анализируем распределение outcomes в памяти
            if len(memory) > 0:
                outcomes = [m[2] for m in memory]
                self.writer.add_histogram('Memory/OutcomeDistribution', np.array(outcomes), iteration)
                self.writer.add_scalar('Memory/AvgOutcome', np.mean(outcomes), iteration)

            # Training фаза
            self.model.train()
            print("\nTraining phase:")
            epoch_losses = []

            for epoch in trange(self.args['num_epochs'], desc="Training"):
                policy_loss, value_loss, total_loss = self.train(memory)
                epoch_losses.append((policy_loss, value_loss, total_loss))

                # Логируем по эпохам
                global_epoch = iteration * self.args['num_epochs'] + epoch
                self.writer.add_scalar('Train/PolicyLoss_epoch', policy_loss, global_epoch)
                self.writer.add_scalar('Train/ValueLoss_epoch', value_loss, global_epoch)
                self.writer.add_scalar('Train/TotalLoss_epoch', total_loss, global_epoch)

            # Средние потери за итерацию
            avg_policy_loss = np.mean([l[0] for l in epoch_losses])
            avg_value_loss = np.mean([l[1] for l in epoch_losses])
            avg_total_loss = np.mean([l[2] for l in epoch_losses])

            self.writer.add_scalar('Train/PolicyLoss_iteration', avg_policy_loss, iteration)
            self.writer.add_scalar('Train/ValueLoss_iteration', avg_value_loss, iteration)
            self.writer.add_scalar('Train/TotalLoss_iteration', avg_total_loss, iteration)

            # Логируем learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, iteration)

            # Логируем нормы градиентов и весов
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar('Train/GradientNorm', total_norm, iteration)

            # Сохраняем модель
            os.makedirs("weights", exist_ok=True)
            torch.save(self.model.state_dict(), f"weights/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"weights/optimizer_{iteration}_{self.game}.pt")

            print(f"\nIteration {iteration + 1} completed:")
            print(f"  Policy Loss: {avg_policy_loss:.4f}")
            print(f"  Value Loss: {avg_value_loss:.4f}")
            print(f"  Total Loss: {avg_total_loss:.4f}")

            # Flush tensorboard
            self.writer.flush()

        self.writer.close()
