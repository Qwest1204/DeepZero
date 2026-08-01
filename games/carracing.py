import gymnasium as gym
import numpy as np
import pygame
from PIL import Image

from games.common import (
    SCREEN_SIZE,
    _init_pygame,
    _poll_quit,
    _render,
    save_session,
)

CAR_DIR = "try/CarRacing"


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


def play_car_racing():
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    screen, clock = _init_pygame(
        (SCREEN_SIZE, SCREEN_SIZE), scale=3,
        title="Car Racing – стрелки: поворот+газ+тормоз, Esc для выхода",
    )

    running = True
    while running:
        actions_list = []
        observations_list = []
        reward_list = []

        obs, info = env.reset()
        obs_resized = np.array(Image.fromarray(obs).resize((SCREEN_SIZE, SCREEN_SIZE)))
        observations_list.append(obs_resized)
        episode_running = True

        while episode_running and running:
            _render(screen, obs_resized)

            if _poll_quit():
                running = False
                break

            steer, gas, brake = _car_action(pygame.key.get_pressed())
            action = np.array([steer, gas, brake], dtype=np.float32)
            actions_list.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            reward_list.append(float(reward))
            obs_resized = np.array(Image.fromarray(obs).resize((SCREEN_SIZE, SCREEN_SIZE)))
            observations_list.append(obs_resized)
            episode_running = not (terminated or truncated)

            clock.tick(30)

        if len(actions_list) > 0:
            save_session(CAR_DIR, "car", actions_list, observations_list, reward_list)

    env.close()
    pygame.quit()
