import os
import sys
import random

import gymnasium as gym
import numpy as np
import pygame
import vizdoom as vzd
from PIL import Image


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


def play_car_racing():
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )

    obs_shape = env.observation_space.shape
    screen, clock = _init_pygame(obs_shape, scale=5, title="Car Racing – стрелки: поворот+газ+тормоз, Esc для выхода")

    actions_list = []
    observations_list = []

    obs, info = env.reset()
    running = True
    reward = -0.1

    while running:
        observations_list.append(obs)
        _render(screen, obs)

        if _poll_quit():
            running = False

        steer, gas, brake = _car_action(pygame.key.get_pressed())
        action = np.array([steer, gas, brake, reward], dtype=np.float32)
        actions_list.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)

        if terminated or truncated:
            obs, info = env.reset()

        clock.tick(30)

    observations_list.append(obs)

    os.makedirs("try", exist_ok=True)
    idx = random.randint(0, 1000000)
    np.save(f"try/actions-car{idx}.npy", np.array(actions_list))
    np.save(f"try/observations-car{idx}.npy", np.array(observations_list))
    print(f"Сохранено {len(actions_list)} действий и {len(observations_list)} наблюдений (car{idx}).")

    env.close()
    pygame.quit()


def play_doom():
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    available_buttons = game.get_available_buttons()
    num_buttons = len(available_buttons)
    print(f"Доступные кнопки: {[b.name for b in available_buttons]}")

    button_idx = {button: i for i, button in enumerate(available_buttons)}

    def mask(*buttons):
        m = [0] * num_buttons
        for b in buttons:
            if b in button_idx:
                m[button_idx[b]] = 1
            else:
                print(f"⚠ Кнопка {b.name} не найдена!")
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
    obs_shape = state.screen_buffer.shape
    screen, clock = _init_pygame(obs_shape, scale=1, title="ViZDoom – стрелки + пробел, Esc")

    actions_list = []
    observations_list = []
    running = True
    os.makedirs("try", exist_ok=True)

    while running:
        state = game.get_state()
        obs_resized = np.array(Image.fromarray(state.screen_buffer).resize((256, 256)))
        observations_list.append(obs_resized)

        _render(screen, obs_resized)

        if _poll_quit():
            running = False

        action = _doom_action_index(pygame.key.get_pressed())
        actions_list.append(action)
        game.make_action(action_masks[action])

        if game.is_episode_finished():
            idx = random.randint(0, 1000000)
            np.save(f"try/doom-act{idx}.npy", np.array(actions_list, dtype=np.int32))
            np.save(f"try/doom-obs{idx}.npy", np.array(observations_list))
            print(f"Сохранено {len(actions_list)} действий и {len(observations_list)} наблюдений (doom{idx}).")
            actions_list = []
            observations_list = []
            game.new_episode()

        clock.tick(30)

    game.close()
    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game = sys.argv[1].lower()
    else:
        print("Выберите игру:")
        print("  1 — CarRacing")
        print("  2 — ViZDoom")
        game = input("Введите 1 или 2: ").strip()

    if game in ("1", "car", "carracing", "car_racing"):
        play_car_racing()
    elif game in ("2", "doom", "vizdoom", "doom-viz"):
        play_doom()
    else:
        print(f"Неизвестный выбор: '{game}'. Запускаю CarRacing по умолчанию.")
        play_car_racing()