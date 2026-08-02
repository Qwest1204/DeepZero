"""Play MetaWorld through the VAE bottleneck — original, reconstruction and latent.

Usage:
    python vae_mw.py [--vae-weights path]

Renders a single camera (topview) through the VAE. Latent maps are shown as RGB
(first 3 channels) plus the remaining channels as separate grey maps.
"""

import argparse
import glob
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
import torch

from embedder import VAE
from games.common import (
    _init_pygame,
    _poll_quit,
    _render,
    build_vae_view,
    latent_to_rgb,
)

DEVICE = "mps"
MW_CAM = 0        # topview camera index (matches mw/record.py cam 1)
LAT_SCALE = 4
FINAL_SCALE = 2


def _auto_detect_weights():
    candidates = sorted(glob.glob("weights/MW/*"))
    if not candidates:
        candidates = sorted(glob.glob("weights/mw*"))
    if not candidates:
        print("Не найдены веса VAE для MetaWorld в weights/ (укажите --vae-weights)")
        sys.exit(1)
    return candidates[-1]


def play_mw(vae):
    import metaworld
    from mujoco import Renderer

    task_names = sorted(metaworld.ALL_V3_ENVIRONMENTS)
    print(f"MetaWorld: доступно {len(task_names)} задач (V3)")

    img_size = vae.img_size
    screen, clock = _init_pygame(
        (img_size * 2, img_size), scale=FINAL_SCALE,
        title="VAE MetaWorld – WASD dx/dy, Q/E dz, пробел = захват, Esc",
    )

    task = random.choice(task_names)
    ml1 = metaworld.ML1(task)
    env = ml1.train_classes[task](render_mode="rgb_array")
    env.set_task(random.choice(ml1.train_tasks))
    env.max_path_length = 2000
    obs, _ = env.reset()
    cam_names = [env.model.cam(i).name for i in range(env.model.ncam)]
    renderer = Renderer(env.model, img_size, img_size)
    print(f"Задача: {task}, камера: {cam_names[MW_CAM]}, obs_dim={obs.shape[0]}")

    gripper_state = 0.0
    running = True

    # physical key states (scancodes are layout-independent)
    MOVE_KEYS = {
        pygame.KSCAN_A: "dx+", pygame.KSCAN_UP: "dx+",
        pygame.KSCAN_D: "dx-", pygame.KSCAN_DOWN: "dx-",
        pygame.KSCAN_S: "dy+", pygame.KSCAN_RIGHT: "dy+",
        pygame.KSCAN_W: "dy-", pygame.KSCAN_LEFT: "dy-",
        pygame.KSCAN_E: "dz+",
        pygame.KSCAN_Q: "dz-",
    }
    key_state = {k: False for k in MOVE_KEYS}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.scancode == pygame.KSCAN_ESCAPE:
                    running = False
                    break
                if event.scancode in key_state:
                    key_state[event.scancode] = True
                if event.scancode == pygame.KSCAN_SPACE:
                    gripper_state = 1.0 - gripper_state
            elif event.type == pygame.KEYUP:
                if event.scancode in key_state:
                    key_state[event.scancode] = False
        if not running:
            break

        renderer.update_scene(env.data, camera=MW_CAM)
        frame_rgb = renderer.render()  # (img_size, img_size, 3) uint8
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            recon, mu, _ = vae(tensor)
        recon_rgb = _recon_to_rgb(recon, vae.final_activation)
        lat_imgs = latent_to_rgb(mu)
        canvas = build_vae_view(lat_imgs, frame_rgb, recon_rgb, lat_scale=LAT_SCALE)
        _render(screen, canvas)

        dx = dy = dz = 0.0
        if key_state[pygame.KSCAN_A] or key_state[pygame.KSCAN_UP]:
            dx = 0.5
        if key_state[pygame.KSCAN_D] or key_state[pygame.KSCAN_DOWN]:
            dx = -0.5
        if key_state[pygame.KSCAN_S] or key_state[pygame.KSCAN_RIGHT]:
            dy = 0.5
        if key_state[pygame.KSCAN_W] or key_state[pygame.KSCAN_LEFT]:
            dy = -0.5
        if key_state[pygame.KSCAN_E]:
            dz = 0.5
        if key_state[pygame.KSCAN_Q]:
            dz = -0.5
        action = np.array([dx, dy, dz, gripper_state], dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

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
    parser = argparse.ArgumentParser(description="Play MetaWorld through the VAE encoder")
    parser.add_argument("--vae-weights", default=None,
                        help="Path to VAE checkpoint (config.json + model.safetensors)")
    args = parser.parse_args()

    weights_path = args.vae_weights or _auto_detect_weights()
    print(f"Веса VAE: {weights_path}")

    vae = VAE.from_pretrained(weights_path, map_location="cpu").to(DEVICE)
    vae.eval()
    print(f"VAE: latent_dim={vae.latent_dim}, flat_latent={vae.flat_latent}, "
          f"img_size={vae.img_size}")

    play_mw(vae)


if __name__ == "__main__":
    main()