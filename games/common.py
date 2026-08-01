import os
import random

import numpy as np
import pygame

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
