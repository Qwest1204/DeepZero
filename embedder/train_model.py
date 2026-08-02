import sys; sys.path.insert(0, "..")
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import RecordingDataset
from embedder import VAE, PatchGANDiscriminator, discriminator_loss
from embedder.losses import wavelet_loss, gaussian_pyramid_loss, free_bits_kl

# --- Hyperparameters ---
DEVICE = "mps"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE == "cuda"  # mixed precision (torch.amp); CUDA only
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-5

IMG_SIZE = 96
LATENT_DIM = 2
FLAT_LATENT = False
ENCODER_CHANNELS = [32, 64, 64, 128]
USE_RESBLOCKS = True
USE_ATTENTION = False
ATTENTION_LAYERS = [2]
NUM_ATTENTION_HEADS = 4
RES_BLOCKS_PER_STAGE = 1
NORM_GROUPS = 32
NDF = 128
FREE_NATS = 0.5
W_WAVELET = 0.3
W_GAUSS = 0.1
W_ADV = 0.001

SAVE_DIR = "../weights/CR"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Data ---
dataset = RecordingDataset(data_dir="../try/CarRacing/", game="car", mode="vae")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Models ---
vae = VAE(
    in_channels=3,
    latent_dim=LATENT_DIM,
    img_size=IMG_SIZE,
    flat_latent=FLAT_LATENT,
    encoder_channels=ENCODER_CHANNELS,
    use_resblocks=USE_RESBLOCKS,
    use_attention=USE_ATTENTION,
    attention_layers=ATTENTION_LAYERS,
    num_attention_heads=NUM_ATTENTION_HEADS,
    res_blocks_per_stage=RES_BLOCKS_PER_STAGE,
    norm_groups=NORM_GROUPS,
    final_activation="sigmoid",
).to(DEVICE)

discriminator = PatchGANDiscriminator(in_channels=3, ndf=NDF).to(DEVICE)

opt = optim.AdamW(vae.parameters(), lr=LR)
opt_d = optim.AdamW(discriminator.parameters(), lr=LR)
scaler = GradScaler("cuda") if USE_AMP else None

# --- Train ---
for epoch in range(EPOCHS):
    vae.train()
    discriminator.train()
    total_loss = 0.0
    last_batch = None

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in pbar:
        x = batch.to(DEVICE)
        x = F.interpolate(x, size=IMG_SIZE)

        with autocast("cuda", enabled=USE_AMP):
            recon_x, mu, logvar = vae(x)
            loss_d, _ = discriminator_loss(
                discriminator(x), discriminator(recon_x.detach()), loss_type="lsgan"
            )

        opt_d.zero_grad()
        if scaler is not None:
            scaler.scale(loss_d).backward()
            scaler.step(opt_d)
        else:
            loss_d.backward()
            opt_d.step()

        with autocast("cuda", enabled=USE_AMP):
            d_fake = discriminator(recon_x)
            mse = F.mse_loss(recon_x, x, reduction="sum")
            wav = wavelet_loss(recon_x, x, levels=3)
            gauss = gaussian_pyramid_loss(recon_x, x, levels=3)
            kl = free_bits_kl(mu, logvar, free_nats=FREE_NATS)
            adv = F.mse_loss(d_fake, torch.ones_like(d_fake)) * recon_x.size(0)
            total = mse + W_WAVELET * wav + W_GAUSS * gauss + kl + W_ADV * adv

        opt.zero_grad()
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.step(opt)
        else:
            total.backward()
            opt.step()
        if scaler is not None:
            scaler.update()

        total_loss += total.item()
        last_batch = x
        pbar.set_postfix(mse=f"{mse.item():.2e}", wav=f"{wav.item():.2e}",
                         gauss=f"{gauss.item():.2e}", kl=f"{kl.item():.2e}",
                         adv=f"{adv.item():.2e}")

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg loss: {avg_loss:.4f}")

    # --- Visualisation ---
    vae.eval()
    with torch.no_grad():
        recon, mu, _ = vae(last_batch)
        n = min(4, len(last_batch))
        n_ch = mu.shape[1]
        fig, axes = plt.subplots(3, n, figsize=(3 * n, 8))
        for i in range(n):
            axes[0, i].imshow(np.transpose(last_batch[i].cpu().numpy(), (1, 2, 0)))
            axes[0, i].axis("off")
            latent_maps = mu[i].cpu().numpy()
            ch_imgs = []
            for c in range(n_ch):
                ch = latent_maps[c]
                ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                ch_imgs.append(ch)
            axes[1, i].imshow(np.concatenate(ch_imgs, axis=0), cmap="gray", vmin=0, vmax=1)
            axes[1, i].axis("off")
            axes[2, i].imshow(np.transpose(recon[i].cpu().numpy(), (1, 2, 0)))
            axes[2, i].axis("off")
        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel(f"Latent\n({n_ch}ch)")
        axes[2, 0].set_ylabel("Recon")
        plt.suptitle(f"Epoch {epoch+1}/{EPOCHS}")
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/val_{epoch+1:03d}.png", dpi=150)
        plt.close()
    vae.train()

    vae.save_pretrained(f"{SAVE_DIR}/model_{epoch+1:03d}")
    print(f"Model saved to {SAVE_DIR}/model_{epoch+1:03d}.png")
