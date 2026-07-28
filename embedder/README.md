# VAE — вариационный автоэнкодер для сжатия игровых кадров

Сжимает RGB-кадры игры в компактный латентный вектор `z` для world-model пайплайна DeepZero.

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
| `encoder_channels` | [32, 64, 128, 256] | Количество каналов каждого свёрточного блока |
| `encoder_kernels` | [4, 4, 4, 4] | Размеры ядер свёрток |
| `encoder_strides` | [2, 2, 2, 2] | Страйды свёрток |
| `decoder_channels` | см. ниже | Каналы деконволюций (по умолчанию зеркалит энкодер) |
| `decoder_kernels` | см. ниже | Ядра деконволюций (по умолчанию зеркалит энкодер) |
| `decoder_strides` | см. ниже | Страйды деконволюций (по умолчанию зеркалит энкодер) |
| `attention_layers` | None | Индексы слоёв, после которых вставить `SelfAttention2D` |
| `num_attention_heads` | 4 | Число голов в `SelfAttention2D` |
| `final_activation` | "sigmoid" | Финальная активация декодера |


Значения decoder-параметров по умолчанию вычисляются из энкодера:
- `decoder_channels` = `reversed(encoder_channels[:-1]) + [in_channels]`
- `decoder_kernels` = `reversed(encoder_kernels)`
- `decoder_strides` = `reversed(encoder_strides)`

## SelfAttention2D

Адаптирует `MultiHeadAttention` (из DNN Building Blocks) к 2D-фича-мэпам:

```
(B, C, H, W)  →  reshape  →  (B, H·W, C)  →  MHA  →  (B, H·W, C)  →  reshape  →  (B, C, H, W)
```

Pre-norm через `LayerNorm`, residual connection вокруг MHA.

## Формат весов

Каждая обученная модель — директория с двумя файлами:

```
weights/car/
├── config.json        # JSON-конфиг (все параметры конструктора)
└── model.safetensors  # веса в формате safetensors
```

Сериализация не зависит от pickle/Python-версии:
- `VAE.save_pretrained(path)` — запись конфига + весов
- `VAE.from_pretrained(path)` — загрузка => воссоздание модели

## Зависимости

- `torch>=2.12.0`
- `safetensors>=0.6.0`
