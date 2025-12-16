# 🎮 DeepZero

**DeepZero** — реализация алгоритма обучения с подкреплением на основе AlphaZero для настольных игр. Нейросеть обучается играть в игры исключительно через самостоятельную игру (self-play), без использования человеческих знаний или заранее подготовленных баз данных.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## 🎯 Поддерживаемые игры

| Игра | Размер доски | Действия | Каналы | Сложность |
|------|--------------|----------|--------|-----------|
| ❌⭕ Крестики-нолики | 3×3 | 9 | 3 | ⭐ |
| 🔴🟡 Четыре в ряд | 6×7 | 7 | 3 | ⭐⭐ |
| ⚫⚪ Шашки | 8×8 | 4096 | 5 | ⭐⭐⭐ |
| ♟️♚ Шахматы | 8×8 | 4096 | 13 | ⭐⭐⭐⭐ |

## 🧠 Алгоритм

DeepZero использует комбинацию **глубокой нейронной сети** и **поиска по дереву Монте-Карло (MCTS)**.

### Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                      DeepZero                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐      ┌─────────────┐      ┌────────────┐  │
│   │   Игровая   │ ───▶ │   ResNet    │ ───▶ │   MCTS     │  │
│   │    среда    │      │  (policy,   │      │  (поиск)   │  │
│   │             │ ◀─── │   value)    │ ◀─── │            │  │
│   └─────────────┘      └─────────────┘      └────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Компоненты

#### 1. 🎲 Игровые среды (`games/`)
Каждая игра реализует единый интерфейс:

```python
class Game:
    def get_initial_state(self)           # Начальное состояние
    def get_next_state(state, action, player)  # Применить ход
    def get_valid_moves(state)            # Маска допустимых ходов
    def check_win(state, action)          # Проверка победы
    def get_value_and_terminated(state, action)  # Значение и терминальность
    def change_perspective(state, player) # Смена перспективы
    def get_encoded_state(state)          # Кодирование для нейросети
```

#### 2. 🧬 Нейросеть ResNet (`models/resnet.py`)
Остаточная нейронная сеть с двумя головами:

```
Input: encoded_state [channels × height × width]
          │
          ▼
    ┌─────────────┐
    │ Conv Block  │
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │  ResBlocks  │ × N
    └─────────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌────────┐ ┌────────┐
│ Policy │ │ Value  │
│  Head  │ │  Head  │
└────────┘ └────────┘
    │           │
    ▼           ▼
 π(s,a)       v(s)
```

- **Policy Head** `π(s,a)`: Вероятности действий
- **Value Head** `v(s)`: Оценка позиции [-1, 1]

#### 3. 🌳 MCTS (`models/mcts.py`)
Поиск по дереву Монте-Карло улучшает политику нейросети:

```
          Selection          Expansion         Simulation        Backpropagation
              │                  │                  │                   │
              ▼                  ▼                  ▼                   ▼
           ┌───┐              ┌───┐              ┌───┐              ┌───┐
           │ ● │──────────────│ ● │──────────────│ ● │──────────────│ ● │
           └─┬─┘              └─┬─┘              └─┬─┘              └─┬─┘
           ┌─┴─┐              ┌─┴─┐              ┌─┴─┐              ┌─┴─┐
           │   │              │   │              │   │              │   │
          ●   ●              ●   ●              ●   ●──▶NN        ●   ●
                                  │                  │                 ▲
                                  ▼                  ▼                 │
                                  ○              v=0.7 ────────────────┘
```

**UCB формула для выбора узла:**
```
UCB(s,a) = Q(s,a) + C × π(s,a) × √(N(s)) / (1 + N(s,a))
```

#### 4. 🔄 Self-Play (`models/deepzero.py`)
Цикл обучения:

```
┌─────────────────────────────────────────────────────────┐
│                    Итерация обучения                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Self-Play (генерация данных)                        │
│     ┌──────────────────────────────────────────┐        │
│     │  for game in parallel_games:             │        │
│     │      state = initial_state               │        │
│     │      while not terminated:               │        │
│     │          π = MCTS.search(state)          │        │
│     │          action = sample(π)              │        │
│     │          memory.append(state, π)         │        │
│     │          state = next_state(action)      │        │
│     │      assign_values(memory, winner)       │        │
│     └──────────────────────────────────────────┘        │
│                          │                               │
│                          ▼                               │
│  2. Training (обучение нейросети)                       │
│     ┌──────────────────────────────────────────┐        │
│     │  for epoch in epochs:                    │        │
│     │      for batch in memory:                │        │
│     │          π_pred, v_pred = model(states)  │        │
│     │          loss = CE(π_pred, π_target)     │        │
│     │                + MSE(v_pred, v_target)   │        │
│     │          optimizer.step()                │        │
│     └──────────────────────────────────────────┘        │
│                          │                               │
│                          ▼                               │
│  3. Save checkpoint                                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 📁 Структура проекта

```
DeepZero/
├── games/
│   ├── tictactoe.py      # Крестики-нолики
│   ├── connectfour.py    # Четыре в ряд
│   ├── checkers.py       # Шашки
│   └── chess.py          # Шахматы
├── models/
│   ├── resnet.py         # Нейросеть
│   ├── mcts.py           # Поиск Монте-Карло
│   └── deepzero.py       # Self-play обучение
├── train_tictactoe.py    # Обучение крестиков-ноликов
├── train_checkers.py     # Обучение шашек
├── train_chess.py        # Обучение шахмат
├── play_vs_ai.py         # Игра против ИИ
└── README.md
```

## 🚀 Быстрый старт

### Зависимости

```txt
numpy>=1.21.0
torch>=2.0.0
tqdm>=4.60.0
```

## 🎓 Обучение

### Крестики-нолики (быстро, ~5 минут)

```python
from games.tictactoe import TicTacToe
from models.resnet import ResNet
from models.deepzero import DeepZeroParallel
import torch

game = TicTacToe()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, num_resBlocks=4, num_hidden=64, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

args = {
    'C': 2,                        # UCB константа
    'num_searches': 60,            # MCTS симуляций
    'num_iterations': 3,           # Итераций обучения
    'num_parallel_games': 100,     # Параллельных игр
    'num_selfPlay_iterations': 500,# Self-play игр за итерацию
    'num_epochs': 4,               # Эпох обучения
    'batch_size': 64,
    'temperature': 1.25,           # Температура выбора действия
    'dirichlet_epsilon': 0.25,     # Шум исследования
    'dirichlet_alpha': 0.3
}

deepzero = DeepZeroParallel(model, optimizer, game, args)
deepzero.learn()
```

### Шашки (средне, ~2-4 часа на GPU)

```python
from games.checkers import Checkers
from models.resnet import ResNet
from models.deepzero import DeepZeroParallel
import torch

game = Checkers()
device = torch.device("cuda")

model = ResNet(game, num_resBlocks=9, num_hidden=128, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

args = {
    'C': 2,
    'num_searches': 100,
    'num_iterations': 8,
    'num_parallel_games': 32,
    'num_selfPlay_iterations': 100,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.5
}

deepzero = DeepZeroParallel(model, optimizer, game, args)
deepzero.learn()
```

### Шахматы (долго, ~24-48 часов на GPU)

```python
from games.chess import Chess
from models.resnet import ResNet
from models.deepzero import DeepZeroParallel
import torch

game = Chess()
device = torch.device("cuda")

model = ResNet(game, num_resBlocks=19, num_hidden=256, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

args = {
    'C': 2,
    'num_searches': 400,
    'num_iterations': 20,
    'num_parallel_games': 64,
    'num_selfPlay_iterations': 200,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

deepzero = DeepZeroParallel(model, optimizer, game, args)
deepzero.learn()
```

## 🎮 Игра против ИИ

```python
from games.checkers import Checkers
from models.resnet import ResNet
from models.mcts import MCTS
import torch
import numpy as np

game = Checkers()
device = torch.device("cpu")

# Загрузка обученной модели
model = ResNet(game, 9, 128, device=device)
model.load_state_dict(torch.load("model_Checkers_7.pt", map_location=device))
model.eval()

args = {'C': 2, 'num_searches': 600, 'dirichlet_epsilon': 0, 'dirichlet_alpha': 0.3}
mcts = MCTS(game, args, model)

state = game.get_initial_state()
player = 1  # Вы играете белыми

while True:
    game.print_board(state)
    
    if player == 1:
        # Ход человека
        valid_moves = game.get_valid_moves(state)
        valid_actions = np.where(valid_moves == 1)[0]
        
        print("Доступные ходы:")
        for i, action in enumerate(valid_actions):
            fr, fc, tr, tc = game.action_to_coords(action)
            print(f"  {i}: ({fr},{fc}) -> ({tr},{tc})")
        
        choice = int(input("Ваш ход: "))
        action = valid_actions[choice]
    else:
        # Ход ИИ
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        action = game.flip_action(action)
        print(f"ИИ ходит: {game.action_to_coords(action)}")
    
    state = game.get_next_state(state, action, player)
    value, terminated = game.get_value_and_terminated(state, action)
    
    if terminated:
        game.print_board(state)
        print("Белые победили!" if value == 1 and player == 1 else "Чёрные победили!")
        break
    
    player = game.get_opponent(player)
```

## 📊 Параметры

| Параметр | Описание | TicTacToe | Checkers | Chess |
|----------|----------|-----------|----------|-------|
| `num_resBlocks` | Количество residual блоков | 4 | 9 | 19 |
| `num_hidden` | Размер скрытого слоя | 64 | 128 | 256 |
| `num_searches` | MCTS симуляций за ход | 60 | 100 | 400 |
| `num_iterations` | Итераций обучения | 3 | 8 | 20 |
| `num_parallel_games` | Параллельных self-play игр | 100 | 32 | 64 |
| `dirichlet_alpha` | Параметр шума Дирихле | 0.3 | 0.5 | 0.3 |

## 📈 Результаты обучения

После обучения модель сохраняется в файлы:
- `model_{Game}_{iteration}.pt` — веса модели
- `optimizer_{Game}_{iteration}.pt` — состояние оптимизатора

## 🔧 API игровых сред

Все игры реализуют единый интерфейс:

```python
class Game:
    row_count: int          # Высота доски
    column_count: int       # Ширина доски
    action_size: int        # Размер пространства действий
    shape_obs: int          # Количество каналов для нейросети
    
    def __repr__(self) -> str
    def get_initial_state(self) -> np.ndarray
    def get_next_state(self, state, action, player) -> np.ndarray
    def get_valid_moves(self, state) -> np.ndarray
    def check_win(self, state, action) -> bool
    def get_value_and_terminated(self, state, action) -> Tuple[int, bool]
    def get_opponent(self, player) -> int
    def get_opponent_value(self, value) -> int
    def change_perspective(self, state, player) -> np.ndarray
    def get_encoded_state(self, state) -> np.ndarray
    def flip_action(self, action) -> int
```

## 📚 Литература

- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) — AlphaGo Zero
- [A general reinforcement learning algorithm that masters chess, shogi, and Go](https://www.science.org/doi/10.1126/science.aar6404) — AlphaZero
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) — MuZero

## 📝 Лицензия

MIT License

## 🤝 Вклад

Pull requests приветствуются! Для крупных изменений сначала откройте issue.

---

<p align="center">
  Made with ❤️ and 🧠
</p>