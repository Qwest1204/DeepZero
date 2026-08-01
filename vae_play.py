"""Play ViZDoom through the VAE bottleneck — original frame, latent maps and reconstruction.

Usage:
    python vae_play.py [--vae-weights path] [--record]

If ``--vae-weights`` is omitted, the script auto-detects the latest epoch checkpoint
in ``weights/doom_*/`` (e.g. ``weights/doom_sq_mid_49/``).
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import pygame
import torch
import vizdoom as vzd
from PIL import Image

from embedder import VAE

IMG_SIZE = 256
LATENT_W = 32       # expected latent map side (32x32 for doom_sq_mid_49)
LATENT_SCALE = 4    # 32x128 -> 128x512
FINAL_SCALE = 2     # final window scale

DEVICE = "mps"


# ---------------------------------------------------------------------------
# helpers — shared with games/common.py
# ---------------------------------------------------------------------------

def _init_pygame(obs_shape, scale, title):
    pygame.init()
    screen_width = obs_shape[1] * scale
    screen_height = obs_shape[0] * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    return screen, clock


def _poll_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _doom_action_index(keys):
    forward = keys[pygame.K_UP]
    left = keys[pygame.K_LEFT]
    right = keys[pygame.K_RIGHT]
    shoot = keys[pygame.K_SPACE]
    if forward and left:
        return 5
    elif forward and right:
        return 6
    elif forward:
        return 1
    elif left:
        return 2
    elif right:
        return 3
    elif shoot:
        return 4
    else:
        return 0


# ---------------------------------------------------------------------------
# VAE helpers
# ---------------------------------------------------------------------------

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


def _frame_to_tensor(frame_rgb, img_size):
    frame_pil = Image.fromarray(frame_rgb)
    if frame_pil.size != (img_size, img_size):
        frame_pil = frame_pil.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(frame_pil, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor


def _recon_to_rgb(recon_tensor, activation="sigmoid"):
    """(B, 3, H, W) -> (H, W, 3) uint8, inverts activation range."""
    arr = recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if activation == "tanh":
        arr = (arr + 1.0) / 2.0
    arr = arr.clip(0, 1)
    return (arr * 255).astype(np.uint8)


def _latent_maps_to_gray(mu):
    """mu (B, C, H, W) -> (H, C*H) uint8 — channels stacked vertically, per-channel norm."""
    maps = mu.squeeze(0).cpu().numpy()  # (C, H, W)
    normed = []
    for ch in range(maps.shape[0]):
        m = maps[ch]
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        normed.append(m)
    stack = np.concatenate(normed, axis=0)  # (H, C*H)
    return (stack * 255).astype(np.uint8)


def _gray_to_rgb(gray):
    return np.repeat(gray[:, :, None], 3, axis=2)


def _render_composite(screen, orig_rgb, recon_rgb, latent_gray):
    """Compose latent (left) + original (top-right) + recon (bottom-right)."""
    lat_h, lat_w = latent_gray.shape
    latent_surf = pygame.surfarray.make_surface(_gray_to_rgb(latent_gray).transpose(1, 0, 2))
    latent_scaled = pygame.transform.scale(latent_surf, (lat_w * LATENT_SCALE, lat_h * LATENT_SCALE))

    orig_surf = pygame.surfarray.make_surface(orig_rgb.transpose(1, 0, 2))
    recon_surf = pygame.surfarray.make_surface(recon_rgb.transpose(1, 0, 2))

    composite = pygame.Surface((lat_w * LATENT_SCALE + IMG_SIZE, 2 * IMG_SIZE), pygame.SRCALPHA)
    composite.blit(latent_scaled, (0, 0))
    composite.blit(orig_surf, (lat_w * LATENT_SCALE, 0))
    composite.blit(recon_surf, (lat_w * LATENT_SCALE, IMG_SIZE))

    screen.blit(pygame.transform.scale(composite, screen.get_size()), (0, 0))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# ViZDoom loop
# ---------------------------------------------------------------------------

def play_doom(vae, do_record):
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

    game.new_episode()
    screen, clock = _init_pygame(
        (LATENT_W * LATENT_SCALE + IMG_SIZE, 2 * IMG_SIZE),
        scale=FINAL_SCALE,
        title="VAE ViZDoom – стрелки + пробел, Esc",
    )
    actions_list = []
    observations_list = []
    running = True
    os.makedirs("try", exist_ok=True)

    while running:
        if _poll_quit():
            running = False
            break

        state = game.get_state()
        frame_raw = state.screen_buffer  # (H, W, 3) np.uint8
        frame_rgb = np.array(
            Image.fromarray(frame_raw).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        )

        tensor = _frame_to_tensor(frame_raw, IMG_SIZE)
        with torch.no_grad():
            recon, mu, _ = vae(tensor)
        recon_rgb = _recon_to_rgb(recon, vae.final_activation)
        latent_gray = _latent_maps_to_gray(mu)
        _render_composite(screen, frame_rgb, recon_rgb, latent_gray)

        if do_record:
            observations_list.append(frame_rgb)

        action = _doom_action_index(pygame.key.get_pressed())
        if do_record:
            actions_list.append(action)
        game.make_action(action_masks[action])

        if game.is_episode_finished():
            if do_record:
                idx = random.randint(0, 1000000)
                np.save(f"try/doom-act{idx}.npy", np.array(actions_list, dtype=np.int32))
                np.save(f"try/doom-obs{idx}.npy", np.array(observations_list))
                print(f"Сохранено doom{idx}: {len(actions_list)} действий, {len(observations_list)} наблюдений")
                actions_list = []
                observations_list = []
            game.new_episode()

        clock.tick(30)

    game.close()
    pygame.quit()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Play ViZDoom through the VAE bottleneck"
    )
    parser.add_argument("--vae-weights", default=None,
                        help="Path to VAE checkpoint (config.json + model.safetensors)")
    parser.add_argument("--record", action="store_true",
                        help="Save observations and actions to try/")
    args = parser.parse_args()

    if args.vae_weights:
        weights_path = args.vae_weights
    else:
        weights_path = _auto_detect_weights()
        print(f"Автоопределённые веса: {weights_path}")

    vae = VAE.from_pretrained(weights_path, map_location="cpu").to(DEVICE)
    vae.eval()
    print(f"VAE загружен: latent_dim={vae.latent_dim}, img_size={vae.img_size}")

    play_doom(vae, do_record=args.record)


if __name__ == "__main__":
    main()
