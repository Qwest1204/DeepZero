import math
import numpy as np
import torch
from tqdm import tqdm

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


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            return self.args['C'] * child.prior * math.sqrt(self.visit_count + 1)
        q_value = child.value_sum / child.visit_count  # value с точки зрения корня
        return q_value + self.args['C'] * child.prior * math.sqrt(self.visit_count) / (child.visit_count + 1)

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                # state уже neutral, action тоже neutral
                # Применяем как player=1 (это neutral perspective)
                child_state = self.game.get_next_state(self.state, action, player=1)
                # Меняем перспективу для следующего хода
                child_state = self.game.change_perspective(child_state, player=-1)
                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(value)




class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        winrate = None
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        valid_moves = self.game.get_valid_moves(state)

        valid_indices = np.where(valid_moves == 1)[0]
        if len(valid_indices) > 0:
            dirichlet_noise = np.zeros(self.game.action_size)
            dirichlet_noise[valid_indices] = np.random.dirichlet(
                [self.args['dirichlet_alpha']] * len(valid_indices)
            )
            policy = (1 - self.args['dirichlet_epsilon']) * policy + \
                     self.args['dirichlet_epsilon'] * dirichlet_noise

        policy = mask_and_normalize(policy, valid_moves)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy = mask_and_normalize(policy, valid_moves)
                value = value.item()
                winrate = value
                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs, winrate

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        # states = neutral_states (текущий игрок положительный)
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, dim=1).cpu().numpy()

        for i, spg in enumerate(spGames):
            spg_policy = policy[i].copy()
            valid_moves = self.game.get_valid_moves(states[i])

            valid_indices = np.where(valid_moves == 1)[0]
            if len(valid_indices) > 0:
                dirichlet_noise = np.zeros(self.game.action_size, dtype=np.float64)
                dirichlet_noise[valid_indices] = np.random.dirichlet(
                    [self.args['dirichlet_alpha']] * len(valid_indices)
                )
                spg_policy = (1 - self.args['dirichlet_epsilon']) * spg_policy + \
                             self.args['dirichlet_epsilon'] * dirichlet_noise

            spg_policy = mask_and_normalize(spg_policy, valid_moves)

            # Root state = neutral_state
            spg.root = Node(self.game, self.args, states[i].copy(), visit_count=1)
            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, node.action_taken
                )

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_spGames = [idx for idx in range(len(spGames))
                                  if spGames[idx].node is not None]

            if len(expandable_spGames) > 0:
                states_batch = np.stack(
                    [spGames[idx].node.state for idx in expandable_spGames]
                )

                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states_batch),
                                 device=self.model.device)
                )
                policy = torch.softmax(policy, dim=1).cpu().numpy()
                value = value.cpu().numpy()

                for i, mappingIdx in enumerate(expandable_spGames):
                    node = spGames[mappingIdx].node
                    spg_policy, spg_value = policy[i].copy(), value[i]

                    valid_moves = self.game.get_valid_moves(node.state)
                    spg_policy = mask_and_normalize(spg_policy, valid_moves)

                    node.expand(spg_policy)
                    node.backpropagate(spg_value)
