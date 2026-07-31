# Handoff #1 — DeepZero VAE (июль 2026)

## Проект

DeepZero — deep RL world-model агент. Трёхэтапный пайплайн:

1. **Embed** (VAE) — сжатие изображения в латентный вектор (1 токен на timestep)
2. **Predict** (MDN Transformer) — авторегрессионная модель мира
3. **Control** (линейная политика / CMA-ES) — планирование в латенте

## Текущее состояние VAE

### Архитектура: единый класс `VAE` (`embedder/vae.py`)

Параметризован — один класс покрывает все режимы:

| Режим | `flat_latent` | `use_resblocks` | `use_attention` | Params |
|---|---|---|---|---|
| Конфиг сейчас | `False` | `False` | `False` | **1.07M** |
| ResVAE | `False` | `True` | `True` | ~8.2M |
| Legacy (car/doom) | `True` | `False` | `False` | ~12.7M |

Текущий конфиг (обучение Doom):
```
in_channels=3
latent_dim=4            # latent_channels (квадратный латент)
img_size=256
flat_latent=False       # conv_mu/logvar вместо fc bridge
encoder_channels=[32,64,128,256]
use_resblocks=False
use_attention=False
final_activation='sigmoid'
```

Размер латента: `(B, 4, 16, 16)` → flatten `(B, 1024)`.
Сжатие: `196608 / 1024 = 192:1`.

### bridge
- `flat_latent=False`: `conv_mu(1×1)` / `conv_logvar(1×1)` — spatial mu/logvar `(B, C, H, W)`
- `flat_latent=True` (legacy): flatten → `fc_mu` / `fc_logvar` — backward compat

### encode / decode
- `encode(x) → mu, logvar` — spatial (flat_latent=False) или flat (flat_latent=True)
- `decode(z)` — `z.dim()==2` → reshape; `z.dim()==4` → прямой проход
- `forward(x) → recon, mu, logvar`

### Восстановленные классы (из старого ResVAE)
- `ResBlock2D` — GroupNorm+ReLU+Conv3×3×2 + skip
- `DownsampleBlock` — Conv2d(k=4,s=2,p=1)
- `UpsampleBlock` — Upsample(nearest)+Conv2d(k=3)

## Обучение

### `embedder/train_model.py`

```python
LR = 3e-5                # единый lr для VAE и D
BATCH_SIZE = 32
EPOCHS = 80
FREE_NATS = 0.5          # free bits KL threshold
W_WAVELET = 0.3
W_GAUSS = 0.1
W_ADV = 0.001
```

### Loss-функции (`embedder/losses.py`)

- **MSE** — `reduction="sum"`, основной loss
- **wavelet_loss** — Haar DWT 3 уровня, L1 на высокочастотных поддиапазонах (LH, HL, HH)
- **gaussian_pyramid_loss** — avg_pool2d × 3, MSE на каждой шкале
- **free_bits_kl** — `sum(max(KL_per_dim - 0.5, 0))` вместо `beta*KL`
  - Внутри делает `.flatten(1)` — работает и с flat, и spatial
- **PatchGAN**(LSGAN) — 70×70 discriminator, `W_ADV=0.001`
- LPIPS — удалён (жрёт ~500MB VRAM, заменён wavelet+gauss)

### Валидация

В конце каждой эпохи — решётка 3×4:
- **Row 0**: Original (RGB 256×256)
- **Row 1**: Latent (4ch, stacked 64×16, gray, per-channel norm)
- **Row 2**: Recon (RGB 256×256)

Сохраняется в `weights/doom_xl/val_{epoch:03d}.png`.

## Проблемы

### 1. NPC не видны в латенте
Сжатие 192:1 — NPC занимают ~16×16 пикселей (0.4% кадра). 4 канала на
16×16 не хватает ёмкости для мелких деталей.

**План исправления (согласован с пользователем):**
- `latent_channels=8` (сжатие 96:1) — сразу +ёмкость
- `use_resblocks=True` — GroupNorm сохраняет границы через skip
- `use_attention=True` — SelfAttention2D на stage 3 (16×16→256 токенов)

Это даст ~2.5M params (всё ещё ×5 меньше старого ~12.7M).

### 2. Низкая утилизация GPU
CPU загружен на 60%, GPU простаивает. Причина:
- `DataLoader` без `num_workers=0` (по умолчанию)
- `F.interpolate(x, size=256)` на каждый батч

**План:** `num_workers=4`, `pin_memory=True`, перенести resize в датасет.

## Файловая структура (embedder)

```
embedder/
  vae.py            — VAE, ResBlock2D, DownsampleBlock, UpsampleBlock
  losses.py         — LPIPS, PatchGAN, VAECombinedLoss, wavelet_loss, gauss_loss, free_bits_kl
  attention.py      — MultiHeadAttention, SelfAttention2D
  __init__.py       — экспорт всех классов/функций
  train_model.py  — активный скрипт обучения (Doom, 256², квадратный латент)
  embedder.ipynb    — старый ноутбук (не обновлялся)
  train_.py         — удалён (был для старого ResVAE)
```

Внешние зависимости:
```
dataset.py          — RecordingDataset (vae/predictor modes)
record_human.py     — запись игры человеком
play_in_dream.py    — авторегрессивная симуляция
vae_play.py          — VAE encode→decode реального времени
predictor/          — PredictorTransformer (не трогали)
controller/         — Controller (CMA-ES, не трогали)
```

## Команды

```bash
uv run python embedder/train_model.py    # обучение VAE
uv run python record_human.py 2            # запись Doom
uv run python vae_play.py doom             # VAE encode→decode
uv run python play_in_dream.py             # dream rollout
```

## Git (вехи)

```
6dd5813 Add latent visualisation
b44034b Simplify training, delete old train_.py and checkpoints
463d369 Unified VAE: square latent + ConvVAE/ResVAE toggle
6b585b3 Replace LPIPS+dynamic balancing with wavelet+gaussian+free_bits
a622593 Replace ResVAE with lightweight ConvVAE
2ab6fcc Old ResVAE notebook (last before lightweight replacement)
```

Ветка: `master` (1 ahead origin).

## Правила общения

- Чат, print, Markdown — **русский**
- Docstring в Python, комменты в коде — **английский**
- Если пользователь не просил другой язык — русский строгий приоритет

## Подводные камни

- **Нет тестов, линтера, CI** — проверять вручную
- **`controller/model.py:16`** — баг бесконечной рекурсии (dead code после return)
- **Pygame** — не объявлен в `pyproject.toml`, но нужен для record/play
- **Предобученные веса**: car/doom используют `flat_latent=True` (старый формат).
  `from_pretrained` автоопределяет: если в config.json нет `flat_latent` → `True`
- Свежие checkpoint-и (`weights/doom_vae-xl1024`, `train_.py`) — удалены
