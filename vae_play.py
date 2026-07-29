"""Play a game through the VAE bottleneck — every frame is encoded then decoded before display.

Usage:
    python vae_play.py [car|doom] [--vae-weights path] [--record]

If ``--vae-weights`` is omitted, the script auto-detects the latest epoch checkpoint
in ``weights/{game}_*/`` (e.g. ``weights/doom_0/``).
"""

import argparse
import glob
import os
import random
import re
import sys

import gymnasium as gym
import numpy as np
import pygame
import torch
import vizdoom as vzd
from PIL import Image

from embedder import VAE


# ---------------------------------------------------------------------------
# helpers — shared with record_human.py
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


def _render_recon(screen, recon_np):
    recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(recon_np.transpose(1, 0, 2))
    scaled = pygame.transform.scale(surf, screen.get_size())
    screen.blit(scaled, (0, 0))
    pygame.display.flip()


def _car_action(keys):
    steer = 0.0
    gas = 0.0
    brake = 0.0
    if keys[pygame.K_LEFT]:
        steer = -1.0
    if keys[pygame.K_RIGHT]:
        steer = 1.0
    if keys[pygame.K_UP]:
        gas = 1.0
    if keys[pygame.K_DOWN]:
        brake = 1.0
        gas = 0.0
    return steer, gas, brake


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

def _auto_detect_weights(game):
    pattern = os.path.join("weights", f"{game}_*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        dirs = sorted(glob.glob(os.path.join("weights", game)))
    if not dirs:
        print(f"Не найдены веса VAE для игры '{game}' в weights/")
        print("Укажите путь через --vae-weights")
        sys.exit(1)
    return dirs[-1]


def _frame_to_tensor(frame_rgb, img_size, vae_activation):
    frame_pil = Image.fromarray(frame_rgb)
    if frame_pil.size != (img_size, img_size):
        frame_pil = frame_pil.resize((img_size, img_size), Image.LANCZOS)
    arr = np.array(frame_pil, dtype=np.float32) / 255.0
    if vae_activation == "tanh":
        arr = arr * 2.0 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _recon_to_display(recon_tensor, orig_size, vae_activation):
    arr = recon_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if vae_activation == "tanh":
        arr = (arr + 1.0) / 2.0
    arr = arr.clip(0, 1)
    if orig_size and arr.shape[:2] != (orig_size[1], orig_size[0]):
        arr = (
            Image.fromarray((arr * 255).astype(np.uint8))
            .resize((orig_size[1], orig_size[0]), Image.LANCZOS)
        )
        arr = np.array(arr, dtype=np.float32) / 255.0
    return arr


# ---------------------------------------------------------------------------
# CarRacing loop
# ---------------------------------------------------------------------------

def play_car_racing(vae, do_record):
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    obs_shape = (96, 96, 3)
    screen, clock = _init_pygame(
        (96, 96), scale=5,
        title="VAE CarRacing – стрелки, Esc для выхода",
    )

    actions_list = []
    observations_list = []
    obs, info = env.reset()
    running = True

    while running:
        if _poll_quit():
            running = False
            break

        tensor = _frame_to_tensor(obs, vae.img_size, vae.final_activation)
        with torch.no_grad():
            recon, _, _ = vae(tensor)
        display = _recon_to_display(recon, obs.shape[:2][::-1], vae.final_activation)
        _render_recon(screen, display)

        if do_record:
            observations_list.append(obs)

        steer, gas, brake = _car_action(pygame.key.get_pressed())
        action = np.array([steer, gas, brake, -0.1], dtype=np.float32)
        if do_record:
            actions_list.append(action)

        obs, reward, terminated, truncated, info = env.step(action[:3])
        if terminated or truncated:
            obs, info = env.reset()
        clock.tick(30)

    if do_record:
        observations_list.append(obs)
        os.makedirs("try", exist_ok=True)
        idx = random.randint(0, 1000000)
        np.save(f"try/actions-car{idx}.npy", np.array(actions_list))
        np.save(f"try/observations-car{idx}.npy", np.array(observations_list))
        print(f"Сохранено car{idx}: {len(actions_list)} действий, {len(observations_list)} наблюдений")

    env.close()
    pygame.quit()


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
    state = game.get_state()
    screen, clock = _init_pygame(
        (vae.img_size, vae.img_size), scale=2,
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

        tensor = _frame_to_tensor(frame_raw, vae.img_size, vae.final_activation)
        with torch.no_grad():
            recon, _, _ = vae(tensor)
        display = _recon_to_display(recon, frame_raw.shape[:2][::-1], vae.final_activation)
        _render_recon(screen, display)

        if do_record:
            observations_list.append(
                np.array(
                    Image.fromarray(frame_raw).resize((vae.img_size, vae.img_size), Image.LANCZOS)
                )
            )

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
        description="Play a game through the VAE bottleneck"
    )
    parser.add_argument("game", nargs="?", default=None,
                        help="car or doom")
    parser.add_argument("--vae-weights", default=None,
                        help="Path to VAE checkpoint (config.json + model.safetensors)")
    parser.add_argument("--record", action="store_true",
                        help="Save observations and actions to try/")
    args = parser.parse_args()

    game = args.game
    if game is None:
        print("Выберите игру:")
        print("  1 — CarRacing")
        print("  2 — ViZDoom")
        choice = input("Введите 1 или 2: ").strip()
        if choice in ("1", "car"):
            game = "car"
        elif choice in ("2", "doom"):
            game = "doom"
        else:
            print("По умолчанию CarRacing")
            game = "car"

    if args.vae_weights:
        weights_path = args.vae_weights
    else:
        weights_path = _auto_detect_weights(game)
        print(f"Автоопределённые веса: {weights_path}")

    vae = VAE.from_pretrained(weights_path, map_location="cpu")
    vae.eval()
    print(f"VAE загружен: latent_dim={vae.latent_dim}, img_size={vae.img_size}")

    if game in ("car", "carracing", "car_racing"):
        play_car_racing(vae, do_record=args.record)
    else:
        play_doom(vae, do_record=args.record)


if __name__ == "__main__":
    main()
