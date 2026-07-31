# VAE — вариационный автоэнкодер для сжатия игровых кадров

Сжимает RGB-кадры игры в латентное представление `z` для world-model пайплайна DeepZero.
Один класс `VAE` покрывает все режимы (ConvVAE / ResVAE, flat / square latent) — выбор
происходит параметрами конструктора, а не отдельными классами.

## Быстрый старт

```python
from embedder import VAE

vae = VAE(in_channels=3, latent_dim=4, img_size=256,
          encoder_channels=[64, 128, 256],
          use_resblocks=True, use_attention=True, attention_layers=[2])
recon_x, mu, logvar = vae(x)

vae.save_pretrained("../weights/my_model")
vae = VAE.from_pretrained("../weights/my_model")
```

## Конструктор

| Параметр | По умолчанию | Описание |
|---|---|---|
| `in_channels` | 3 | Каналы входного изображения |
| `latent_dim` | 4 | Размерность латента: flat — длина вектора, square — число каналов карты |
| `img_size` | 256 | Сторона квадратного изображения |
| `flat_latent` | `False` | `True` — fc bridge (legacy); `False` — conv bridge → квадратный латент `(B, C, H, W)` |
| `encoder_channels` | [32,64,128,256,256,256] | Каналы каждой ступени энкодера |
| `decoder_channels` | зеркалит энкодер | Каналы декодера |
| `use_resblocks` | `False` | `True` — ResBlock2D+GroupNorm (стабильно, сохраняет детали); `False` — Conv2d+ReLU (лёгкий, CPU) |
| `use_attention` | `False` | Вставить SelfAttention2D на выбранных ступенях |
| `attention_layers` | `[]` | Индексы ступеней с attention (например `[2]`) |
| `num_attention_heads` | 4 | Голов в SelfAttention2D |
| `res_blocks_per_stage` | 1 | ResBlock2D на ступень |
| `norm_groups` | 32 | Группы GroupNorm |
| `final_activation` | "sigmoid" | Финальная активация декодера |

⚠️ **`use_attention=True` без непустого `attention_layers` молча отключает внимание** — всегда передавай `attention_layers=[N]`.

### Мост к латенту

- `flat_latent=True` (legacy): последняя карта энкодера флаттенится → `fc_mu`/`fc_logvar` → вектор `z`. Используется только для старых чекпоинтов (backward-compat).
- `flat_latent=False` (square): `conv_mu`/`conv_logvar` (Conv2d 1×1) → латентная карта `(B, C, H, W)`. Градиент до conv напрямую, без бутылочного горлышка fc.

Каждая ступень энкодера (res-режим): `[ResBlock2D(in, out)] × N → DownsampleBlock(out, out)`.
Декодер зеркалит: `UpsampleBlock(in, out) → [ResBlock2D(out, out)] × N`.

- `ResBlock2D`: GroupNorm → ReLU → Conv3×3 × 2 + residual
- `DownsampleBlock`: Conv2d(k=4, s=2)
- `UpsampleBlock`: `nn.Upsample(nearest, ×2)` + `Conv2d(k=3)` — без checkerboard-артефактов

## Отчёт: обученная модель Doom (`doom_sq_mid_49`)

### Конфигурация

| Параметр | Значение |
|---|---|
| Вход | 256×256×3 |
| `encoder_channels` | [64, 128, 256] (3 ступени) |
| `use_resblocks` | `True` |
| `use_attention` | `True`, `attention_layers=[2]` |
| `latent_dim` | 4 (каналы) |
| Латент | квадратный `(B, 4, 32, 32)` → 4096 димов |
| Сжатие | **48:1** (256²·3 / 4096) — то же, что SDXL KL-F8 |
| Параметры | ~8.2M (оценка) |
| Чекпоинт | `../weights/doom_sq_mid_49` (пер-эпохальные `doom_sq_mid_*`) |

### Обучение

- 50 эпох на A10, batch 32, lr 3e-5, AMP.
- DataLoader: `num_workers=4, pin_memory=True, prefetch_factor=4` (иначе CPU бутылочное горлышко).
- Loss-функции: `wavelet_loss`, `gaussian_pyramid_loss`, `free_bits_kl`, PatchGAN (подробнее ниже).

### Диагностика латента (PCA в `embedder.ipynb`, блок 7)

- **100% живых димов** — ёмкость латента задействована полностью, сжимать PCA-проекцией нельзя.
- **3 чётких кластера с переходами** в 2D-проекции — соответствуют сценам записи; переходы между ними — то, что должен выучить предиктор.
- **Ближайшие соседи семантически похожи** — латент гладкий, динамика в нём обучаема.
- Итог: латент готов к этапу **predict** (MDN Transformer).

## Loss-функции (`embedder/losses.py`)

Используются в `train_xlmodel.py`:

| Функция | Назначение |
|---|---|
| `wavelet_loss` | Haar DWT (3 уровня), L1 по высокочастотным субполосам — резкость текстур |
| `gaussian_pyramid_loss` | avg_pool2d (3 уровня), MSE по пирамиде — многошкальная структура |
| `free_bits_kl` | KL с порогом free bits (0.5 nats), поддерживает flat и квадратные `mu`/`logvar` |
| `PatchGANDiscriminator` + `discriminator_loss` | 70×70 PatchGAN, LSGAN — резкость (вес 0.001, слабое влияние) |

Итоговый loss на шаг: `mse + 0.3·wav + 0.1·gauss + free_bits_kl + 0.001·adv`.

В `losses.py` также остаются неиспользуемые `LPIPS` и `VAECombinedLoss` (legacy — из обученного пайплайна убраны: доминировали и не давали выигрыша).

## Формат весов

Каждая обученная модель — директория с двумя файлами:

```
weights/doom_sq_mid_49/
├── config.json        # JSON-конфиг (все параметры конструктора)
└── model.safetensors  # веса в формате safetensors
```

Сериализация не зависит от pickle/Python-версии: `VAE.save_pretrained(path)` / `VAE.from_pretrained(path)`.

## Зависимости

- `torch>=2.1`
- `safetensors>=0.6.0`
- `torchvision` (только для legacy LPIPS)
