"""Play CarRacing through the VAE bottleneck — original, reconstruction and latent.

Usage:
    python vae_car.py [--vae-weights path]

If ``--vae-weights`` is omitted, the script auto-detects a CarRacing VAE
checkpoint in ``weights/`` (e.g. ``weights/CR``). Latent maps are shown as RGB
(first 3 channels) plus the remaining channels as separate grey maps.
"""

import argparse
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import pygame
import torch
from PIL import Image

from embedder import VAE
from games.common import (
    _init_pygame,
    _poll_quit,
    _render,
    build_vae_view,
    latent_to_rgb,
)
from games.carracing import _car_action

DEVICE = "mps"
LAT_SCALE = 4
FINAL_SCALE = 4


def _auto_detect_weights():
    candidates = sorted(glob.glob("weights/CR/*"))
    if not candidates:
        candidates = sorted(glob.glob("weights/car*"))
    if not candidates:
        print("Не найдены веса VAE для CarRacing в weights/ (покажите --vae-weights)")
        sys.exit(1)
    return candidates[-1]


def play_car(vae):
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    img_size = vae.img_size
    screen, clock = _init_pygame(
        (img_size * 2, img_size * 2), scale=FINAL_SCALE,
        title="VAE CarRacing – стрелки: поворот+газ+тормоз, Esc",
    )

    obs, _ = env.reset()
    running = True
    while running:
        frame_rgb = np.array(Image.fromarray(obs).resize((img_size, img_size), Image.LANCZOS))
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            recon, mu, _ = vae(tensor)
        recon_rgb = _recon_to_rgb(recon, vae.final_activation)
        lat_imgs = latent_to_rgb(mu)
        canvas = build_vae_view(lat_imgs, frame_rgb, recon_rgb, lat_scale=LAT_SCALE)
        _render(screen, canvas)

        if _poll_quit():
            break

        steer, gas, brake = _car_action(pygame.key.get_pressed())
        obs, _, terminated, truncated, _ = env.step(np.array([steer, gas, brake], dtype=np.float32))
        if terminated or truncated:
            obs, _ = env.reset()

        clock.tick(30)

    env.close()
    pygame.quit()


def _recon_to_rgb(recon_tensor, activation="sigmoid"):
    arr = recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if activation == "tanh":
        arr = (arr + 1.0) / 2.0
    arr = arr.clip(0, 1)
    return (arr * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Play CarRacing through the VAE bottleneck")
    parser.add_argument("--vae-weights", default=None,
                        help="Path to VAE checkpoint (config.json + model.safetensors)")
    args = parser.parse_args()

    weights_path = args.vae_weights or _auto_detect_weights()
    print(f"Веса VAE: {weights_path}")

    vae = VAE.from_pretrained(weights_path, map_location="cpu").to(DEVICE)
    vae.eval()
    print(f"VAE: latent_dim={vae.latent_dim}, flat_latent={vae.flat_latent}, "
          f"img_size={vae.img_size}")

    play_car(vae)


if __name__ == "__main__":
    main()