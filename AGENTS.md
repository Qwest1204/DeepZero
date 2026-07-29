# AGENTS.md

# Правила общения

- **Всегда отвечай исключительно на русском языке**, независимо от языка запроса пользователя или языка исходного кода.
- Все сообщения в чате, print-выводы и Markdown-пояснения в ноутбуках — на русском.
- Docstring в Python-коде и комментарии внутри кода — **на английском**.
- Если пользователь явно не попросил ответить на другом языке, русский язык является строгим приоритетом по умолчанию.

## Проект
DeepZero — deep RL world-model агент для CarRacing-v3 и ViZDoom. Трёхэтапный пайплайн: **embed** (VAE) → **predict** (MDN Transformer) → **control** (линейная политика, оптимизация CMA-ES). Разработка через Jupyter-ноутбуки, предобученные веса закоммичены прямо в репозиторий.

## Команды
```bash
python record_human.py [1|2]     # 1=CarRacing, 2=ViZDoom — запись игры человеком
python play_in_dream.py          # авторегрессивная симуляция мира из случайного латентного вектора
python vae_play.py [car|doom]    # игра через VAE-бутылку (encode→decode перед отображением)
```

Обучение происходит в Jupyter-ноутбуках:
- `embedder/embedder.ipynb` — обучение VAE
- `predictor/predictor.ipynb` — обучение предиктора-трансформера
- `CME-ES/main.ipynb` — оптимизация контроллера через CMA-ES

Для управления виртуальным окружением используется `uv` (`.venv/`, in-project). Pygame необходим для запуска, но **отсутствует** в `pyproject.toml` — при необходимости установи вручную.

## Архитектура
| Этап | Модуль | Вход → Выход |
|---|---|---|
| Embed | `embedder/vae.py:VAE` | `(B,3,96,96)` изображение → латентный вектор `z` (размерность 32) |
| Predict | `predictor/model.py:PredictorTransformer` | `(z_seq, action_seq)` → MDN над `z_next` (n=4 гауссианы, размерность 32) |
| Control | `controller/model.py:Controller` | `[z_current, z_target]` → `tanh(action)` в `[-1,1]^3` |

VAE (`embedder/vae.py`) — две независимые ветки развития:
- **`master`** — оригинальный VAE на `Conv2d+ReLU` и `ConvTranspose2d`.
- **`ResVAE`** — улучшенная архитектура: `ResBlock2D (GroupNorm+ReLU+Conv3×3×2)`, `Upsample(nearest)+Conv2d`, поддержка LPIPS и PatchGAN.

SelfAttention2D реализован в `embedder/attention.py` (MultiHeadAttention через `F.scaled_dot_product_attention`).
Сохранение/загрузка: `VAE.save_pretrained(path)` / `VAE.from_pretrained(path)` — JSON-конфиг + safetensors.
Веса лежат в `weights/{car,doom}/` (config.json + model.safetensors).

**Loss-метрики** (`embedder/losses.py`):
| Loss | Назначение |
|---|---|
| MSE | Попиксельная реконструкция |
| β·KL | β-VAE регуляризация (по умолч. β=0.1) |
| LPIPS | Перцептивная loss через VGG16 (сохраняет NPC/монстры) |
| Adversarial | PatchGAN (70×70), LSGAN loss — резкие текстуры |

## Подводные камни
- **Нет тестов, линтера, CI** — проверяй всё вручную.
- **`controller/model.py:16` содержит баг бесконечной рекурсии** — строка `mean, log_std, value = self.forward(...)` вызывает `forward` внутри самого `forward` и недостижима (мёртвый код после `return` на строке 15). Не используй код ниже строки 15; это заброшенная PPO-голова. Реальный контроллер детерминированный (`torch.tanh`).
- **Pygame — необъявленная зависимость** — `record_human.py` и `play_in_dream.py` импортируют его.
- **Смешанный русский/английский** в комментариях и print-сообщениях.