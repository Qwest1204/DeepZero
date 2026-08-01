# Handoff #1 — DeepZero (июль 2026)

## Проект

DeepZero — deep RL world-model агент. Трёхэтапный пайплайн:

1. **Embed** (VAE) — сжатие изображения в латентное представление (1 токен на timestep)
2. **Predict** (MDN Transformer) — авторегрессионная модель мира
3. **Control** (линейная политика / CMA-ES) — планирование в латенте

**Статус: этап embed (VAE для Doom) ЗАВЕРШЁН. Следующий — predict, отложен до появления вычислительных ресурсов.**

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
| `num_attention_heads` | 4 | 4 |
| Params | 1.07M | ~8.2M (оценка) |
| Латент | `(B,4,32,32)` = 4096 | `(B,4,32,32)` = 4096 |
| Сжатие | 48:1 | 48:1 |

⚠️ **`use_attention=True` без непустого `attention_layers` молча отключает внимание** — всегда передавай `attention_layers=[N]`.

### bridge
- `flat_latent=False`: `conv_mu(1×1)` / `conv_logvar(1×1)` — spatial mu/logvar `(B, C, H, W)`
- `flat_latent=True` (legacy): flatten → `fc_mu` / `fc_logvar` — backward compat (старые car/doom чекпоинты)

### encode / decode
- `encode(x) → mu, logvar` — spatial (flat_latent=False) или flat (flat_latent=True)
- `decode(z)` — `z.dim()==2` → reshape; `z.dim()==4` → прямой проход
- `forward(x) → recon, mu, logvar`

### Блоки (res-режим)
- `ResBlock2D` — GroupNorm+ReLU+Conv3×3×2 + skip
- `DownsampleBlock` — Conv2d(k=4,s=2,p=1)
- `UpsampleBlock` — Upsample(nearest)+Conv2d(k=3), без checkerboard
- `SelfAttention2D` (`attention.py`) — MHA по (B, H·W, C), pre-norm + residual

## Обучение (A10, удалённо)

### Фактический конфиг прогона (внимание: отличается от дефолта в скрипте!)

```
in_channels=3, latent_dim=4, img_size=256
encoder_channels=[64,128,256]        # 3 ступени → 32×32 латент
use_resblocks=True, use_attention=True, attention_layers=[2]
final_activation='sigmoid'
LR=3e-5, BATCH_SIZE=32, 50 эпох, AMP
DataLoader: num_workers=4, pin_memory=True, prefetch_factor=4
```

- Чекпоинты: пер-эпохальные `../weights/doom_sq_mid_*` (финальный — `doom_sq_mid_49`)
- `embedder/train_model.py` (дефолт на диске: `[32,64,128,256]`, без resblocks/attention, `SAVE_DIR=../weights/doom_xl`) — **отличается от реального прогона**; перед следующим обучением привести к фактическому конфигу
- Дискриминатор НЕ сохранялся — при resume безопасно: `W_ADV=0.001` даёт ~0.1% вклада, D восстанавливается за 1–2 эпохи

### Loss-функции (`embedder/losses.py`)

| Функция | Назначение |
|---|---|
| MSE (`reduction="sum"`) | основной loss |
| `wavelet_loss` | Haar DWT 3 уровня, L1 на LH/HL/HH — резкость |
| `gaussian_pyramid_loss` | avg_pool2d ×3, MSE по шкалам |
| `free_bits_kl` | `sum(max(KL_per_dim - 0.5, 0))`; `.flatten(1)` — работает flat и spatial |
| PatchGAN + `discriminator_loss` | 70×70 LSGAN, `W_ADV=0.001` |

Итог: `mse + 0.3·wav + 0.1·gauss + kl + 0.001·adv`. LPIPS/VAECombinedLoss — legacy, не используются.

### Динамика в конце обучения (эпохи 44–50)
- avg loss стабилизировался ~1460–1480; вклад: wavelet ~60%, MSE ~33%, KL ~6%, gauss <1%, adv ~0
- последняя эпоха «качнулась» (kl 382→1030, mse 3.07e3→5.48e3)

## Диагностика латента (PCA, `embedder.ipynb` блок 7)

Выборка 20% (seed 42), z = reparameterize(mu, logvar), flattened 4096.

**Результаты (все положительные):**
- **100% живых димов** (порог 1% от макс. дисперсии) — ёмкость используется полностью, PCA-сжатие НЕ применять (потеря инфы)
- **3 чётких кластера с переходами** в 2D-проекции — сцены записи; переходы — динамика, которую учит предиктор
- **Ближайшие соседи семантически похожи** (NearestNeighbors, эвклид) — латент гладкий

Дополнительно: PCA: n90=n95=51 компонент, PC1=2.3%, PC2=1.5% — информация «рыхлая» по димам, но это НЕ баг: структура есть в полном 4096-мерном пространстве. PCA ≠ fc bridge (подтверждено эмпирически).

**Вывод: латент 4096 используется целиком, VAE закрыт.**

## Этап predict — план (отложен)

### Факты
- **Действия Doom**: 7 дискретных (0–6: idle, F, L, R, A, F+L, F+R), `np.int32` индексы в `doom-act*.npy` → one-hot, `act_space=7`
- `PredictorTransformer` (`predictor/model.py`) готов: `z_dim`, `act_space`, MDN-головы, `save_pretrained`/`from_pretrained`
- Данные только на удалённой машине (локально `../try` пуст) — обучение там
- Старый `predictor/predictor.ipynb` устарел (d_model=128, seq_len=340, старый API)

### Согласованные решения
1. **Прекомпьют латентов**: закодировать все фреймы замороженным VAE (mean, без семплирования) → `try/z-doom{session}.npy` (fp16), параллельно `doom-act*.npy`
2. **Без нормализации** z — предсказывать как есть
3. **Формат**: скрипт `predictor/train_predictor_doom.py` + переработка `predictor/predictor.ipynb`
4. Hold-out сессия для валидации NLL (`mdn_loss`), `save_pretrained` → `../weights/predictor_doom_*`

### TODO (пользователь ещё думает)
- **Параметры предиктора**: d_model / n_layer / n_head / n_gaussians / seq_len (предложение: 1024/6/8/4/32, lr 1e-4, ~110M params — уместится на A10)
- Резюм-логика для VAE: сохранять `train_state.pt` (D, оптимизаторы, epoch)
- Fine-tune VAE при необходимости: W_WAVELET 0.3→0.15, FREE_NATS 0.5→0.2, LR→1e-5; если соседи в латенте НЕ похожи — ретрейн с FREE_NATS=0.1
- Идея на будущее: 8×RTX4090, SDXL VAE + предиктор 10M–0.5B, ~100–1000 ч игрового времени приближают большинство игр; SDXL KL-F8 = 48:1 при любом разрешении (наш 32×32×4 = тот же 48:1)

## vae_play.py — переделан (Doom-only)

Композитное окно (`FINAL_SCALE=2` → 768×1024):
- слева: латентные карты `mu` (4 канала, per-channel норм, вертикально 32×128) → масштаб ×4 → 128×512
- справа сверху: оригинал 256×256 (resize 640×480)
- справа снизу (вплотную к нижней границе): реконструкция 256×256
- учёт `final_activation="tanh"`, латент берётся динамически из `mu.shape`
- CarRacing удалён полностью; usage: `python vae_play.py [--vae-weights path] [--record]`

## Файловая структура

```
embedder/
  vae.py            — VAE, ResBlock2D, DownsampleBlock, UpsampleBlock
  losses.py         — wavelet_loss, gaussian_pyramid_loss, free_bits_kl, PatchGAN, (legacy: LPIPS, VAECombinedLoss)
  attention.py      — MultiHeadAttention, SelfAttention2D
  __init__.py       — экспорт всех классов/функций
  train_model.py    — обучение VAE (дефолт не совпадает с A10-прогоном!)
  README.md         — отчёт по архитектуре VAE и результатам (актуальный)
  embedder.ipynb    — ноутбук: обучение + PCA-диагностика (блок 7, 25 ячеек)
```

Внешние:
```
dataset.py          — RecordingDataset (vae/predictor modes), doom-act = int32 индексы
record_human.py     — запись игры человеком (car/doom)
play_in_dream.py    — dream rollout (car, старый API)
vae_play.py         — Doom-only композитное окно
predictor/          — PredictorTransformer + predictor_mdn_car.pt / predictor_doom_mdn.pt (legacy)
controller/         — Controller (CMA-ES, не трогали)
AGENTS.md           — правила + фокус + таблица архитектуры
handoff-1.md        — этот файл
```

## Команды

```bash
uv run python record_human.py [1|2]        # 1=CarRacing, 2=ViZDoom — запись игры
uv run python vae_play.py                  # Doom: оригинал+латент+реконструкция
uv run python embedder/train_model.py      # обучение VAE
uv run python play_in_dream.py             # dream rollout (car)
```

## Git

```
551827c vae_play: doom-only composite window (original + latent maps + recon)
488c3ab Rename train_xlmodel.py to train_model.py, update references
875832b Sync scripts with unified VAE API and new checkpoints
e8d6f75 Add PCA latent diagnostics to embedder notebook (alive dims, clusters, nearest neighbours)
a4ea02e Update docs: VAE report (Doom 48:1, 32x32x4 latent), AGENTS.md focus on predictor, handoff-1 snapshot
6dd5813 Add latent visualisation to validation grid
b44034b Simplify training, delete old train_.py and checkpoints
463d369 Unified VAE: square latent + ConvVAE/ResVAE toggle
6b585b3 Replace LPIPS+dynamic balancing with wavelet+gaussian+free_bits
a622593 Replace ResVAE with lightweight ConvVAE
```

Ветка: `master`, 5 коммитов впереди `origin/master` (не пушили — не просили).

## Правила общения

- Чат, print, Markdown — **русский**
- Docstring в Python, комменты в коде — **английский**
- Если пользователь не просил другой язык — русский строгий приоритет

## Подводные камни

- **Нет тестов, линтера, CI** — проверять вручную
- **`controller/model.py:16`** — баг бесконечной рекурсии (dead code после return), не использовать
- **Pygame** — не объявлен в `pyproject.toml`, но нужен для record/play
- **Предобученные веса**: car/doom используют `flat_latent=True` (старый формат); `from_pretrained` автоопределяет: нет `flat_latent` в config.json → `True`
- **`use_attention=True` без `attention_layers`** — молча отключает внимание
- **Локально нет данных** (`../try` пуст, `try.zip` 1.6GB untracked) — всё обучение/ноутбуки на удалённой машине
- **`.DS_Store`** untracked — не коммитить
- Удалены: `weights/doom_vae-xl1024`, `train_.py` (старое имя train_model.py)
