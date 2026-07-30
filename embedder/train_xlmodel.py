# %% [markdown]
# ## train_xlmodel — ConvVAE с нуля (latent_dim=768, 6-stage Conv2d+ReLU)
#
# Лёгкая архитектура в стиле World Models, оптимизированная под CPU-инференс:
# - 6× Conv2d(stride=2)+ReLU → 4×4 → 4096 → z=768
# - Декодер: Upsample(nearest×2)+Conv2d(k=3)+ReLU
# - Нет GroupNorm, ResBlock, SelfAttention — только чистые свёртки
# - Сжатие 256:1 (196K → 768)
# - ~14M параметров (vs 51M в ResVAE)
# - LPIPS+GAN только при обучении, на инференсе не используются

# %%
import sys
sys.path.insert(0, "..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from dataset import RecordingDataset
from embedder import VAE, PatchGANDiscriminator, VAECombinedLoss, discriminator_loss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# %% [markdown]
# ## 1. Гиперпараметры

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 80

LR_VAE = 3e-5
LR_D = 3e-5

LATENT_DIM = 768
IMG_SIZE = 256
ENCODER_CHANNELS = [32, 64, 128, 256, 256, 256]  # 6 стадий: 256→4×4

NDF = 128               # фильтров дискриминатора

# KL annealing
BETA_START = 0.0
BETA_END = 0.05
BETA_WARMUP_EPOCHS = 15

SAVE_DIR = "../weights/doom_xl"

os.makedirs(SAVE_DIR, exist_ok=True)

# %% [markdown]
# ## 2. Dataset

# %%
dataset = RecordingDataset(data_dir="../try", game="doom", mode="vae")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# %% [markdown]
# ## 3. Модели

# %%
vae = VAE(
    in_channels=3,
    latent_dim=LATENT_DIM,
    img_size=IMG_SIZE,
    encoder_channels=ENCODER_CHANNELS,
    final_activation="sigmoid",
).to(DEVICE)

discriminator = PatchGANDiscriminator(in_channels=3, ndf=NDF).to(DEVICE)

# %%
from torchinfo import summary

# %%
summary(vae)

# %% [markdown]
# ## 4. Loss с динамической балансировкой

# %%
criterion_base = VAECombinedLoss(
    lpips_weight=1.0,     # будет пересчитан динамически
    adv_weight=1.0,
    beta=BETA_START,
).to(DEVICE)

lpips_module = criterion_base.lpips   # отдельная ссылка для прямого вызова

opt_vae = optim.AdamW(vae.parameters(), lr=LR_VAE)
opt_d = optim.AdamW(discriminator.parameters(), lr=LR_D)
scaler = GradScaler()

# %% [markdown]
# ## 5. Training loop

# %%
for epoch in range(EPOCHS):
    vae.train()
    discriminator.train()
    total_loss = 0.0
    last_batch = None

    # KL annealing
    if epoch < BETA_WARMUP_EPOCHS:
        beta = BETA_START + (BETA_END - BETA_START) * epoch / (BETA_WARMUP_EPOCHS - 1)
    else:
        beta = BETA_END
    criterion_base.beta = beta

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        x = batch.to(DEVICE)

        with autocast():
            recon_x, mu, logvar = vae(x)
            d_real = discriminator(x)
            d_fake = discriminator(recon_x.detach())
            loss_d, _ = discriminator_loss(d_real, d_fake, loss_type="lsgan")

        opt_d.zero_grad()
        scaler.scale(loss_d).backward()
        scaler.step(opt_d)

        with autocast():
            d_fake_for_vae = discriminator(recon_x)

            # ---- Динамическая балансировка ----
            with torch.no_grad():
                mse_val = F.mse_loss(recon_x, x, reduction="sum")
                lpips_val = lpips_module(recon_x, x).sum()
                adv_val = F.mse_loss(
                    d_fake_for_vae, torch.ones_like(d_fake_for_vae)
                ) * recon_x.size(0)

                lpips_weight = mse_val / (lpips_val + 1e-8)
                adv_weight = mse_val / (adv_val + 1e-8) * 0.5   # 50% силы MSE

            criterion_base.lpips_weight = lpips_weight
            criterion_base.adv_weight = adv_weight

            losses = criterion_base(recon_x, x, mu, logvar, d_fake_for_vae)

        opt_vae.zero_grad()
        scaler.scale(losses["loss"]).backward()
        scaler.step(opt_vae)

        scaler.update()

        total_loss += losses["loss"].item()
        last_batch = x

        pbar.set_postfix(
            loss=f"{losses['loss'].item():.2e}",
            mse=f"{losses['mse'].item():.2e}",
            lpips=f"{losses['lpips'].item():.2e}",
            adv=f"{losses['adv'].item():.2e}",
            beta=f"{beta:.4f}",
        )

    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg loss: {avg_loss:.4f}, β: {beta:.4f}")

    vae.eval()
    with torch.no_grad():
        recon_vae = vae(last_batch)[0]
        n = min(4, len(last_batch))
        orig = last_batch[:n].cpu()
        recon_imgs = recon_vae[:n].cpu()

        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        for i in range(n):
            axes[0, i].imshow(np.transpose(orig[i].numpy(), (1, 2, 0)))
            axes[0, i].axis("off")
            axes[1, i].imshow(np.transpose(recon_imgs[i].numpy(), (1, 2, 0)))
            axes[1, i].axis("off")
        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Reconstruction")
        plt.suptitle(f"Epoch {epoch+1}/{EPOCHS}, β={beta:.4f}")
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/val_epoch_{epoch+1:03d}.png", dpi=150)
        plt.close()

    vae.save_pretrained(f"{SAVE_DIR}_{epoch}")
    print(f"Checkpoint saved to {SAVE_DIR}_{epoch}")
    vae.train()
