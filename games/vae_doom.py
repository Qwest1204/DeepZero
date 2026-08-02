"""Play ViZDoom through the VAE bottleneck — original, reconstruction and latent.

Usage:
    python vae_doom.py [--vae-weights path]

If ``--vae-weights`` is omitted, the script auto-detects a Doom checkpoint
in ``weights/``. Latent maps are shown as RGB (first 3 channels) plus the
remaining channels as separate grey maps.
"""

import argparse
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame
import torch
import vizdoom as vzd
from PIL import Image

from embedder import VAE
from games.common import (
    _init_pygame,
    _poll_quit,
    _render,
    build_vae_view,
    latent_to_rgb,
)
from games.doom import _doom_action_index

DEVICE = "mps"
LAT_SCALE = 4    # latent map upscale
FINAL_SCALE = 2  # final window scale


def _auto_detect_weights():
    pattern = os.path.join("weights", "doom_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        dirs = sorted(glob.glob(os.path.join("weights", "doom")))
    if not dirs:
        print("Не найдены веса VAE для Doom в weights/")
        print("Укажите путь через --vae-weights")
        sys.exit(1)
    return dirs[-1]


def play_doom(vae):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    available_buttons = game.get_available_buttons()
    num_buttons = len(available_buttons)
    button_idx = {button: i for i, button in enumerate(available_buttons)}

    def mask(*buttons):
        m = [0] * num_buttons
        for b in buttons:
            if b in button_idx:
                m[button_idx[b]] = 1
        return m

    F = vzd.Button.MOVE_FORWARD
    L = vzd.Button.TURN_LEFT
    R = vzd.Button.TURN_RIGHT
    A = vzd.Button.ATTACK
    action_masks = {
        0: mask(),
        1: mask(F),
        2: mask(L),
        3: mask(R),
        4: mask(A),
        5: mask(F, L),
        6: mask(F, R),
    }

    img_size = vae.img_size
    screen, clock = _init_pygame(
        (img_size * 2, img_size * 2), scale=FINAL_SCALE,
        title="VAE ViZDoom – стрелки + пробел, Esc",
    )
    game.new_episode()
    running = True

    while running:
        if _poll_quit():
            running = False
            break

        state = game.get_state()
        frame_rgb = np.array(
            Image.fromarray(state.screen_buffer).resize((img_size, img_size), Image.LANCZOS)
        )
        tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            recon, mu, _ = vae(tensor)
        recon_rgb = _recon_to_rgb(recon, vae.final_activation)
        lat_imgs = latent_to_rgb(mu)
        canvas = build_vae_view(lat_imgs, frame_rgb, recon_rgb, lat_scale=LAT_SCALE)
        _render(screen, canvas)

        action = _doom_action_index(pygame.key.get_pressed())
        game.make_action(action_masks[action])

        if game.is_episode_finished():
            game.new_episode()

        clock.tick(30)

    game.close()
    pygame.quit()


def _recon_to_rgb(recon_tensor, activation="sigmoid"):
    arr = recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if activation == "tanh":
        arr = (arr + 1.0) / 2.0
    arr = arr.clip(0, 1)
    return (arr * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Play ViZDoom through the VAE bottleneck")
    parser.add_argument("--vae-weights", default=None,
                        help="Path to VAE checkpoint (config.json + model.safetensors)")
    args = parser.parse_args()

    weights_path = args.vae_weights or _auto_detect_weights()
    print(f"Веса VAE: {weights_path}")

    vae = VAE.from_pretrained(weights_path, map_location="cpu").to(DEVICE)
    vae.eval()
    print(f"VAE: latent_dim={vae.latent_dim}, flat_latent={vae.flat_latent}, "
          f"img_size={vae.img_size}")

    play_doom(vae)


if __name__ == "__main__":
    main()