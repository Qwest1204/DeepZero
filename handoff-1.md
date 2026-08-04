# Handoff #1 — DeepZero (август 2026)

## Проект

DeepZero — deep RL world-model агент для CarRacing-v3, ViZDoom и MetaWorld. Трёхэтапный пайплайн:

1. **Embed** (VAE) — сжатие изображения в латентное представление (1 токен на timestep)
2. **Predict** (MDN Transformer) — авторегрессионная модель мира
3. **Control** (линейная политика / CMA-ES) — планирование в латенте

**Статус: этап embed (VAE) ЗАВЕРШЁН и закоммичен в `master`. Рабочее дерево чистое. Следующий — predict (предиктор Doom).**

## Этап embed (VAE) — итог

### Архитектура: единый класс `VAE` (`embedder/vae.py`)

Один класс покрывает все режимы через параметры конструктора:

| Параметр | По умолчанию | Doom обучен (`doom_sq_mid_49`) |
|---|---|---|
| `flat_latent` | `False` | `False` |
| `use_resblocks` | `False` | `True` |
| `use_attention` | `False` | `True` |
| `attention_layers` | `[]` | `[2]` (внимание на 2-й ступени) |
| `latent_dim` | 4 (каналы) | 4 |
| `encoder_channels` | [32,64,128,256] | [64,128,256] |
| `hidden_activation` | `"relu"` | `"relu"` |
| `res_blocks_per_stage` | 1 | 1 |
| `norm_groups` | 32 | 32 |
| Params | ~1.1M | 4.9M |
| Латент | `(B,4,32,32)` = 4096 | `(B,4,32,32)` = 4096 |
| Сжатие | 48:1 | 48:1 |

⚠️ **`use_attention=True` без непустого `attention_layers` молча отключает внимание** — всегда передавай `attention_layers=[N]`.

💡 `hidden_activation` (`"relu"`/`"silu"`/`"gelu"`/`"leaky_relu"`/`"elu"`) применяется в `ResBlock2D` и plain encoder/decoder; skip-путь и `final_activation` декодера (`"sigmoid"`) не трогаются. Старые чекпоинты без «поля» в `config.json` наследуют `"relu"` (фолбэк в `from_pretrained`). Новая активация требует переобучения — обученные модели остаются на `relu`.

### bridge
- `flat_latent=False`: `conv_mu(1×1)` / `conv_logvar(1×1)` — spatial mu/logvar `(B, C, H, W)`
- `flat_latent=True` (legacy): flatten → `fc_mu` / `fc_logvar` — backward compat (старые car/doom чекпоинты)

### encode / decode
- `encode(x) → mu, logvar` — spatial (flat_latent=False) или flat (flat_latent=True)
- `decode(z)` — `z.dim()==2` → reshape; `z.dim()==4` → прямой проход
- `forward(x) → recon, mu, logvar`

### Блоки (res-режим)
- `ResBlock2D` — GroupNorm + `hidden_activation` + Conv3×3×2 + skip
- `DownsampleBlock` — Conv2d(k=4,s=2,p=1)
- `UpsampleBlock` — Upsample(nearest)+Conv2d(k=3), без checkerboard
- `SelfAttention2D` (`attention.py`) — MHA по (B, H·W, C), pre-norm + residual

## Обучение Doom (A10, удалённо)

### Фактический конфиг прогона

```
in_channels=3, latent_dim=4, img_size=256
encoder_channels=[64,128,256]        # 3 ступени → 32×32 латент
use_resblocks=True, use_attention=True, attention_layers=[2]
hidden_activation='relu', final_activation='sigmoid'
LR=3e-5, BATCH_SIZE=32, 50 эпох, AMP
DataLoader: num_workers=4, pin_memory=True, prefetch_factor=4
```

- Чекпоинты: пер-эпохальные `../weights/doom_sq_mid_*` (финальный — `doom_sq_mid_49`), локальная реплика — `weights/doom_vae_sd`
- `embedder/train_model.py` — дефолтные значения на диске могут отличаться от реального прогона; конфиг воспроизводим через блокнот
- Дискриминатор НЕ сохранялся — при resume безопасно: `W_ADV=0.001` даёт ~0.1% вклада, D восстанавливается за 1–2 эпохи
- AMP: `torch.amp` (autocast + GradScaler на CUDA), `USE_AMP = DEVICE == "cuda"` — на MPS/CPU выключена

### Loss-функции (`embedder/losses.py`)

| Функция | Назначение |
|---|---|
| MSE (`reduction="sum"`) | основной loss |
| `wavelet_loss` | Haar DWT 3 уровня, L1 по высокочастотным — резкость |
| `gaussian_pyramid_loss` | avg_pool2d ×3, MSE по шкалам — структура |
| `free_bits_kl` | `sum(max(KL_per_dim - 0.5, 0))`; работает flat и spatial |
| PatchGAN + `discriminator_loss` | 70×70 LSGAN, `W_ADV=0.001` |

Итог: `mse + 0.3·wav + 0.1·gauss + free_bits_kl + 0.001·adv`. LPIPS/VAECombinedLoss — legacy, не используются.

## Диагностика латента (PCA, `embedder.ipynb`)

Выборка 20% (seed 42), z = reparameterize(mu, logvar), flatten 4096.

**Результаты (все положительные):**
- **100% живых димов** — ёмкость задействована полностью, PCA-сжатие НЕ применять
- **3 чётких кластера с переходами** — сцены записи; переходы — динамика, которую учит предиктор
- **Ближайшие соседи семантически похожи** — латент гладкий
- PCA: n90=n95=51, PC1=2.3%, PC2=1.5% — информация «рыхлая», но это не баг (структура в полном 4096-мерном пространстве)

**Вывод: латент 4096 используется целиком, VAE закрыт.**

## Записи и датасет

- Запись единый CLI: `games/record.py` (car/doom), `mw/record.py` (MetaWorld)
- `try/CarRacing/`: car-act (3-dim: steer/gas/brake), car-obs (RGB 192²), car-reward; n_obs = n_act+1; сохранение на границе эпизода и при Esc
- `try/Doom/`: doom-act (`np.int32` 1-dim, 7 действия), doom-obs (RGB 192²); reward без записи
- `try/MW/`: obs RGB 192² каждый 2-й кадр, joints — полный obs-вектор каждый шаг, act=4-dim, reward=dense float32, success-флаги
- `dataset/dataset.py`: единый `RecordingDataset` (car/doom/mw; z-файлы; done-фильтр; хвостовое окно len(obs)-seq_len-1)
- Модель Doom обучалась на 256²; записи 192² → `ResizedVAEDataset` (resize) в ноутбуке

## Этап predict — план

### Факты
- **Действия Doom**: 7 дискретных (0–6: idle, F, L, R, A, F+L, F+R), `np.int32` индексы в `doom-act*.npy` → one-hot, `act_space=7`
- `PredictorTransformer` (`predictor/model.py`) уже есть: `z_dim`, `act_space`, MDN-головы, `save_pretrained`/`from_pretrained`
- Данные записываются локально (есть запись фреймов) и/или на удалённой машине
- Старый `predictor/predictor.ipynb` устарел (d_model=128, seq_len=340, старый API)

### Согласованные решения
1. **Прекомпьют латентов**: закодировать все фреймы замороженным VAE (mean, без семплинга) → `try/z-doom{session}.npy` (fp16), параллельно `doom-act*.npy`
2. **Без нормализации** z — предсказывать как есть
3. **Формат**: скрипт `predictor/train_predictor_doom.py` + переработка `predictor/predictor.ipynb`
4. Hold-out сессия для валидации NLL (`mdn_loss`), `save_pretrained` → `../weights/predictor_doom_*`

### TODO (обсуждается)
- **Параметры предиктора**: d_model / n_layer / n_head / n_gaussians / seq_len (предложение: 1024/6/8/4/32, lr 1e-4, ~110M params — уместится на A10)
- Resume-логика для VAE: сохранять `train_state.pt` (D, оптимизаторы, epoch)
- Fine-tune VAE при необходимости: W_WAVELET 0.3→0.15, FREE_NATS 0.5→0.2, LR→1e-5; если соседи в латенте НЕ похожи — ретрейн с FREE_NATS=0.1
- Идея на будущее: 8×RTX4090, SDXL VAE + предиктор 10M–0.5B, ~100–1000 ч игрового времени; SDXL KL-F8 = 48:1 при любом разрешении (наш 32×32×4 = тот же 48:1)

## VAE-вьюверы (перенесены из корня)

- `games/vae_doom.py` — Doom в реальном времени: оригинал (192) + латент (4 канала) + реконструкция
- `games/vae_car.py` — CarRacing: тот же композит
- `mw/vae_mw.py` — MetaWorld: только предсказание/латент (без реального окна записи)
- Общие хелперы в `games/common.py`: `latent_to_rgb` (первые 3 канала → RGB, 4+ → серые карты), `build_vae_view`
- `vae_play.py` (старый корневой, Doom-only) удалён — заменён на `games/vae_doom.py`
- `play_in_dream.py` — dream rollout (car; требует веса VAE кара) → `games/play_in_dream.py`

## Файловая структура

```
games/                — запись и вьюеры
  record.py           — CLI: -m games.record car|doom
  carracing.py, doom.py, common.py (pygame-хелперы, save_session, латент-композит)
  vae_car.py, vae_doom.py, play_in_dream.py
mw/                   — MetaWorld
  record.py (play_metaworld), env.json (50 задач), vae_mw.py
dataset/
  dataset.py          — RecordingDataset (car/doom/mw, z-файлы, done-фильтр, reward)
embedder/
  vae.py              — VAE, ResBlock2D, Downsample/Upstream, hidden_activation
  losses.py           — wavelet/gaussian_pyramid/free_bits_kl/PatchGAN (+legacy LPIPS, VAECombinedLoss)
  attention.py        — MultiHeadAttention, SelfAttention2D
  __init__.py         — экспорт
  train_model.py      — обучение VAE (AMP: torch.amp, USE_AMP)
  embedder.ipynb      — обучение + PCA-диагностика латента + ResizedVAEDataset
  README.md           — отчёт по архитектуре VAE (актуальный)
predictor/            — PredictorTransformer + старые чекпоинты (legacy API)
controller/           — Controller (CMA-ES, не трогали)
weights/              — в .gitignore; не коммитим: doom_vae_sd, CR/model_032, CR/model_033
AGENTS.md             — правила, фокус predict, таблица архитектуры
README.md             — обзор проекта и методы
handoff-1.md          — этот файл
```

## Команды

```bash
uv run python -m games.record car|doom      # запись игры
uv run python -m mw.record                  # запись MetaWorld
uv run python games/vae_doom.py             # Doom: оригинал+латент+реконструкция
uv run python games/vae_car.py              # CarRacing
uv run python mw/vae_mw.py                  # MetaWorld
uv run python games/play_in_dream.py        # dream rollout (car)
uv run python embedder/train_model.py       # обучение VAE
```

## Git

```
a42c6a6 Remove root VAE play scripts (moved into games/, mw/)
1576340 VAE viewers: vae_doom/vae_car/vae_mw with latent RGB composite
7f20dd3 Embedder: align notebook/train with doom_sq_mid_49 and fix AMP (torch.amp, USE_AMP)
60f5788 VAE: configurable hidden activation (default relu)
7749624 Рефакторинг: единый RecordingDataset и CLI записи для car/doom/mw, разбивка по пакетам
...
```

Полный лог прошлых этапов: `git log --oneline`. Ветка `master`; не пушили — не просили.

## Правила общения

- Чат, print, Markdown — **русский**
- Docstring Python, комменты кода — **английский**
- Если пользователь не просил другой язык — русский приоритет

## Подводные камни

- **Нет тестов, линтера, CI** — проверять вручную
- **`controller/model.py:16`** — баг бесконечной рекурсии (dead code после return), не использовать
- **Pygame** не объявлен в `pyproject.toml`, но нужен для record/play (используется через системный)
- **Предобученные веса car**: `flat_latent=True` (старый формат) — `from_pretrained` автоопределяет: нет `flat_latent`→ `True`
- **`use_attention=True` без `attention_layers`** — молча отключает внимание
- **`weights/` gitignore** — чекпоинты большие (safetensors), не коммитим; конфиги чекпоинтов (в т.ч. `hidden_activation`) живут только локально
- **Локально данных мало** (`try/` заполняется свежими записями), большие датасеты и обучение — на удалённой машине
- **`.DS_Store`** — не коммитить