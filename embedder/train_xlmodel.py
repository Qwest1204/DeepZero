# %% [markdown]
# ## train_xlmodel — ConvVAE 256²→z=16×16×4 (сжатие 192:1)
#
# - Вход: 256×256 (изображение ресайзится перед VAE)
# - 4× Conv2d(stride=2)+ReLU → 16×16×256 → conv_mu → z=(4,16,16) → 1024
# - Квадратный латент: Spatial mu/logvar вместо fc bridge
# - ~1.1M параметров, быстрый CPU-инференс

# %%
import sys
sys.path.insert(0, "..")
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from dataset import RecordingDataset
from embedder import VAE, PatchGANDiscriminator, discriminator_loss
from embedder.losses import wavelet_loss, gaussian_pyramid_loss, free_bits_kl
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# %% [markdown]
# ## 1. Гиперпараметры

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 80

LR_VAE = 3e-5
LR_D = 3e-5

LATENT_DIM = 4          # latent_channels (квадратный латент 16×16×4)
IMG_SIZE = 256
ENCODER_CHANNELS = [32, 64, 128, 256]  # 4 стадии: 256→16
FLAT_LATENT = False     # conv_mu/logvar вместо fc bridge

NDF = 128
FREE_NATS = 0.5          # per-dim free nats (free_bits_kl)

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
    flat_latent=FLAT_LATENT,
    encoder_channels=ENCODER_CHANNELS,
    final_activation="sigmoid",
).to(DEVICE)

discriminator = PatchGANDiscriminator(in_channels=3, ndf=NDF).to(DEVICE)

# %%
from torchinfo import summary

# %%
summary(vae)

# %% [markdown]
# ## 4. Loss: wavelet + gaussian pyramid + free bits KL + PatchGAN

# %%

W_WAVELET = 0.3          # wavelet loss weight
W_GAUSS = 0.1            # gaussian pyramid loss weight
W_ADV = 0.001            # adversarial loss weight (relative to MSE)

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

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        x = batch.to(DEVICE)
        x = F.interpolate(x, size=IMG_SIZE)

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

            mse_loss = F.mse_loss(recon_x, x, reduction="sum")
            wav_loss = wavelet_loss(recon_x, x, levels=3)
            gauss_loss = gaussian_pyramid_loss(recon_x, x, levels=3)
            kl_loss = free_bits_kl(mu, logvar, free_nats=FREE_NATS)
            adv_loss = F.mse_loss(
                d_fake_for_vae, torch.ones_like(d_fake_for_vae)
            ) * recon_x.size(0)

            total = mse_loss + W_WAVELET * wav_loss + W_GAUSS * gauss_loss + kl_loss + W_ADV * adv_loss

        opt_vae.zero_grad()
        scaler.scale(total).backward()
        scaler.step(opt_vae)

        scaler.update()

        total_loss += total.item()
        last_batch = x

        pbar.set_postfix(
            mse=f"{mse_loss.item():.2e}",
            wav=f"{wav_loss.item():.2e}",
            gauss=f"{gauss_loss.item():.2e}",
            kl=f"{kl_loss.item():.2e}",
            adv=f"{adv_loss.item():.2e}",
        )

    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg loss: {avg_loss:.4f}")

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
        plt.suptitle(f"Epoch {epoch+1}/{EPOCHS}")
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/val_epoch_{epoch+1:03d}.png", dpi=150)
        plt.close()

    vae.save_pretrained(f"{SAVE_DIR}_{epoch}")
    print(f"Checkpoint saved to {SAVE_DIR}_{epoch}")
    vae.train()
    