# DeepZero

Deep RL world-model агент для **CarRacing-v3**, **ViZDoom** и **MetaWorld**.
Трёхэтапный пайплайн:

1. **Embed** — VAE сжимает RGB-кадр в латент (1 токен на шаг), сжатие 48:1 (256²×3 → 4×32×32)
2. **Predict** — MDN Transformer предсказывает латент авторегрессионно
3. **Control** — линейная политика, оптимизация CMA-ES в пространстве латента

## Статус

| Этап | Статус |
|---|---|
| **Embed** (VAE) | Завершён, закоммичен. Doom: `weights/doom_vae_sd` (реплика `doom_sq_mid_49`); CarRacing: `weights/CR/model_032`, `model_033`. |
| **Predict** | Текущий фокус (предиктор Doom). Параметры в обсуждении. |
| **Control** | Не начат. |

## Быстрый старт

```bash
uv run python -m games.record car|doom        # запись игры (CarRacing / ViZDoom)
uv run python -m mw.record                    # запись MetaWorld
uv run python games/vae_doom.py               # Doom: VAE в реальном времени (ориг+рекон+латент)
uv run python games/vae_car.py                # CarRacing: VAE в реальном времени
uv run python mw/vae_mw.py                    # MetaWorld: VAE в реальном времени
uv run python games/play_in_dream.py          # dream rollout (требует VAE car)
uv run python embedder/train_model.py         # обучение VAE
```

Подробности — в [AGENTS.md](AGENTS.md) (правила + архитектура VAE), [handoff-1.md](handoff-1.md) (история/контекст), [embedder/README.md](embedder/README.md) (отчёт по VAE).