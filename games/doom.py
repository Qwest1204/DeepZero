import os

import numpy as np
import pygame
import vizdoom as vzd
from PIL import Image

from games.common import (
    SCREEN_SIZE,
    _init_pygame,
    _poll_quit,
    _render,
    save_session,
)

DOOM_DIR = "try/Doom"


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
    rewards_list = []
    running = True
    os.makedirs(DOOM_DIR, exist_ok=True)

    while running:
        state = game.get_state()
        obs_resized = np.array(Image.fromarray(state.screen_buffer).resize((SCREEN_SIZE, SCREEN_SIZE)))
        observations_list.append(obs_resized)

        _render(screen, obs_resized)

        if _poll_quit():
            running = False

        action = _doom_action_index(pygame.key.get_pressed())
        actions_list.append(action)
        game.make_action(action_masks[action])
        rewards_list.append(float(game.get_last_reward()))

        if game.is_episode_finished():
            save_session(
                DOOM_DIR, "doom",
                np.array(actions_list, dtype=np.int32), observations_list,
                rewards_list,
            )
            actions_list = []
            observations_list = []
            rewards_list = []
            game.new_episode()

        clock.tick(30)

    game.close()
    pygame.quit()
