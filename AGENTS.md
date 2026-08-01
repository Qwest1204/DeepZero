# AGENTS.md

## Правила общения

- **Всегда отвечай исключительно на русском языке**, независимо от языка запроса пользователя или языка исходного кода.
- Все сообщения в чате, print-выводы и Markdown-пояснения в ноутбуках — на русском.
- Docstring в Python-коде и комментарии внутри кода — **на английском**.
- Если пользователь явно не попросил ответить на другом языке, русский язык является строгим приоритетом по умолчанию.

## Проект

DeepZero — deep RL world-model агент для CarRacing-v3 и ViZDoom. Трёхэтапный пайплайн: **embed** (VAE) → **predict** (MDN Transformer) → **control** (линейная политика, CMA-ES).

Текущий фокус — предиктор для Doom (этап **predict**).

**Статус этапа embed (VAE для Doom): ЗАВЕРШЁН.**
- Архитектура: resblocks + attention, квадратный латент 32×32×4=4096, сжатие 48:1 (256×256 → 4096).
- Обучен на A10 (50 эпох), чекпоинт `../weights/doom_sq_mid_49`.
- Диагностика латента (PCA в `embedder.ipynb`): 100% живых димов, 3 чётких кластера с переходами, ближайшие соседи семантически похожи — латент готов к предиктору.
- Параметры предиктора ещё обсуждаются (d_model/n_layer/n_gaussians/seq_len в TODO).

## Команды

```bash
uv run python -m games.record car|doom        # запись игры (CarRacing / ViZDoom)
uv run python -m mw.record                    # запись MetaWorld
uv run python play_in_dream.py                # dream rollout (car, требует VAE car)
uv run python vae_play.py                     # Doom: VAE в реальном времени
uv run python embedder/train_model.py         # обучение VAE (Doom, квадратный латент)
```

Записи сохраняются в `try/CarRacing/` (car-act/car-obs/car-reward, obs=RGB 192², act=3-dim, reward каждый шаг, n_obs=n_act+1), `try/Doom/` (doom-act/doom-obs, obs=RGB 192², act=1-dim int32) и `try/MW/` (MetaWorld: obs=RGB 192² каждый 2-й кадр, joints=полный obs-вектор каждый шаг, act=4-dim каждый шаг, reward=dense float32 каждый шаг, success-флаги каждый шаг).

## Ключевые файлы

- `games/record.py` — CLI записи CarRacing/ViZDoom; `games/carracing.py`, `games/doom.py`, `games/common.py` (общие pygame-хелперы и `save_session`)
- `mw/record.py` — запись MetaWorld (play_metaworld), `mw/env.json` — описания 50 задач
- `dataset/dataset.py` — единый RecordingDataset (car/doom/mw, z-файлы, done-фильтр, reward)
- `embedder/vae.py` — единый VAE (ConvVAE/ResVAE, flat/square latent)
- `embedder/losses.py` — wavelet_loss, gaussian_pyramid_loss, free_bits_kl, PatchGAN
- `embedder/train_model.py` — обучение с нуля
- `handoff-1.md` — полный snaphot контекста

## Архитектура VAE

Единый класс, конфигурируется параметрами:

| Параметр | Режим по умолчанию | Doom (обучен, `doom_sq_mid_49`) |
|---|---|---|
| `flat_latent` | `False` | `False` |
| `use_resblocks` | `False` | `True` |
| `use_attention` | `False` | `True` |
| `attention_layers` | `[]` | `[2]` (внимание на 2-й ступени) |
| `latent_dim` | 4 (каналы) | 4 |
| `encoder_channels` | [32,64,128,256] | [64,128,256] |
| Параметры | 1.07M | ~8.2M (оценка) |
| Латент | `(B,4,32,32)` = 4096 | `(B,4,32,32)` = 4096 |
| Сжатие | 48:1 | 48:1 |

⚠️ Внимание: `use_attention=True` без `attention_layers` (непустого списка) молча отключает внимание — всегда передавай `attention_layers=[N]`.

См. `handoff-1.md` для полного контекста.
