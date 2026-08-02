import os
import random

import numpy as np
import pygame
from PIL import Image

SCREEN_SIZE = 192


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


def _render(screen, obs):
    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    scaled = pygame.transform.scale(surf, screen.get_size())
    screen.blit(scaled, (0, 0))
    pygame.display.flip()


def _new_session_id():
    return random.randint(0, 1000000)


def save_session(save_dir, prefix, actions, observations, rewards=None):
    """Save a recorded session to {prefix}-act{idx}.npy / -obs{idx}.npy / -reward{idx}.npy."""
    os.makedirs(save_dir, exist_ok=True)
    idx = _new_session_id()
    np.save(os.path.join(save_dir, f"{prefix}-act{idx}.npy"), np.array(actions))
    np.save(os.path.join(save_dir, f"{prefix}-obs{idx}.npy"), np.array(observations))
    if rewards is not None:
        np.save(os.path.join(save_dir, f"{prefix}-reward{idx}.npy"), np.array(rewards, dtype=np.float32))
    print(f"Сохранено {len(actions)} действий и {len(observations)} наблюдений ({prefix}{idx}).")
    return idx


# ---------------------------------------------------------------------------
# VAE latent visualisation helpers
# ---------------------------------------------------------------------------

def _norm_map(channel, eps=1e-8):
    """Normalize one latent channel to [0, 255] uint8."""
    m = np.asarray(channel, dtype=np.float32)
    m = (m - m.min()) / (m.max() - m.min() + eps)
    return (m * 255).astype(np.uint8)


def latent_to_rgb(mu):
    """Build display images for latent maps (B, C, H, W) -> list of (H, W, 3) uint8.

    First 3 channels are combined into a single RGB image; extra channels
    (starting with the 4th) are kept as separate grey maps. With 3 or fewer
    channels, every channel is shown as its own grey map.
    """
    maps = mu.squeeze(0).detach().cpu().numpy().astype(np.float32)
    C, H, W = maps.shape
    imgs = []
    if C > 3:
        rgb = np.stack([_norm_map(maps[c]) for c in range(3)], axis=-1)
        imgs.append(rgb)
        for c in range(3, C):
            imgs.append(np.repeat(_norm_map(maps[c])[..., None], 3, axis=-1))
    else:
        for c in range(C):
            imgs.append(np.repeat(_norm_map(maps[c])[..., None], 3, axis=-1))
    return imgs


def build_vae_view(latent_imgs, orig_rgb, recon_rgb, lat_scale=4):
    """Compose latent images (left, scaled) + original (top-right) + recon (bottom-right).

    Returns a single (H, W, 3) uint8 frame.
    """
    H, W = latent_imgs[0].shape[:2]
    rows = []
    for img in latent_imgs:
        scaled = Image.fromarray(img).resize(
            (W * lat_scale, H * lat_scale), Image.NEAREST
        )
        rows.append(np.array(scaled))
    lat = np.concatenate(rows, axis=1)
    lat_h, lat_w = lat.shape[:2]

    right = np.concatenate([np.asarray(orig_rgb), np.asarray(recon_rgb)], axis=0)
    rh, rw = right.shape[:2]

    out_h = max(lat_h, rh)
    out_w = lat_w + rw
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:lat_h, :lat_w] = lat
    canvas[:rh, lat_w:] = right
    return canvas
