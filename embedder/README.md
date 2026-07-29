# VAE — вариационный автоэнкодер для сжатия игровых кадров

Сжимает RGB-кадры игры в компактный латентный вектор `z` для world-model пайплайна DeepZero.
Архитектура основана на **ResNet-блоках с GroupNorm** (как в SDXL KL-F8):
- `ResBlock2D`: GroupNorm → ReLU → Conv3×3 × 2 + residual
- `DownsampleBlock`: Conv2d(k=4, s=2)
- `UpsampleBlock`: `nn.Upsample(nearest, ×2)` + `Conv2d(k=3)` — без checkerboard-артефактов и без `output_padding`

## Быстрый старт

```python
from embedder import VAE

vae = VAE(in_channels=3, latent_dim=32, img_size=96)
recon_x, mu, logvar = vae(x)
loss, rl, kl = VAE.loss_vae(recon_x, x, mu, logvar)

vae.save_pretrained("../weights/car")
vae = VAE.from_pretrained("../weights/car")
```

## Конструктор

| Параметр | По умолчанию | Описание |
|---|---|---|
| `in_channels` | 3 | Каналы входного изображения |
| `latent_dim` | 128 | Размерность латентного вектора `z` |
| `img_size` | 96 | Размер стороны квадратного изображения |
| `encoder_channels` | [32, 64, 128, 256, 256] | Количество каналов каждой ступени энкодера |
| `decoder_channels` | см. ниже | Каналы декодера (по умолчанию зеркалит энкодер) |
| `attention_layers` | [] | Индексы ступеней, после которых вставить `SelfAttention2D` |
| `num_attention_heads` | 4 | Число голов в `SelfAttention2D` |
| `resnet_blocks_per_stage` | 1 | Количество `ResBlock2D` на ступень (1 или 2) |
| `norm_groups` | 32 | Количество групп в GroupNorm |
| `final_activation` | "sigmoid" | Финальная активация декодера |

Каждая ступень энкодера:
```
[ResBlock2D(in, out)] × N → DownsampleBlock(out, out)
```

Декодер зеркалит энкодер:
```
UpsampleBlock(in, out) → [ResBlock2D(out, out)] × N
```

## Архитектура потока (256×256, 5 ступеней)

```
Encoder:
  3×256×256 → Conv2d(3, 32, 3)  # проекция
  → ResBlock(32,32) → Down(32,64)  → 128×128
  → ResBlock(64,64) → Down(64,128) → 64×64
  → ResBlock(128,128) → Down(128,256) → 32×32
  → ResBlock(256,256) → SelfAttn → Down(256,256) → 16×16
  → ResBlock(256,256) → SelfAttn → Down(256,256) → 8×8
  → flatten → fc → z(256d)

Decoder:
  z → fc → reshape(256, 8, 8)
  → Up(256,256) → ResBlock(256,256) → SelfAttn
  → Up(256,128) → ResBlock(128,128) → SelfAttn
  → Up(128,64) → ResBlock(64,64)
  → Up(64,32) → ResBlock(32,32)
  → Up(32,3) → Sigmoid
```

## SelfAttention2D

Адаптирует `MultiHeadAttention` (из DNN Building Blocks) к 2D-фича-мэпам:

```
(B, C, H, W)  →  reshape  →  (B, H·W, C)  →  MHA  →  (B, H·W, C)  →  reshape  →  (B, C, H, W)
```

Pre-norm через `LayerNorm`, residual connection вокруг MHA.

## Loss-функции

Помимо стандартного `VAE.loss_vae` (MSE + β·KL), модуль `embedder.losses.py` предоставляет:

| Функция/Класс | Назначение |
|---|---|
| `LPIPS` | Перцептивная loss через VGG16 (4 скрытых слоя), L1 в feature space |
| `PatchGANDiscriminator` | 70×70 PatchGAN (4 conv + InstanceNorm + LeakyReLU) |
| `VAECombinedLoss` | Обёртка: MSE + β·KL + LPIPS_weight·LPIPS + adv_weight·fool D |
| `discriminator_loss()` | LSGAN или hinge loss для обучения D |

## Формат весов

Каждая обученная модель — директория с двумя файлами:

```
weights/doom_N/
├── config.json        # JSON-конфиг (все параметры конструктора)
└── model.safetensors  # веса в формате safetensors
```

Сериализация не зависит от pickle/Python-версии:
- `VAE.save_pretrained(path)` — запись конфига + весов
- `VAE.from_pretrained(path)` — загрузка => воссоздание модели

## Зависимости

- `torch>=2.12.0`
- `torchvision>=0.22.0` (для LPIPS)
- `safetensors>=0.6.0`
