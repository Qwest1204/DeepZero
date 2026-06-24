import gymnasium as gym
import numpy as np
import pygame
import sys
import os
import vizdoom as vzd
from PIL import Image
from vizdoom import gymnasium_wrapper


# ============================================================
#  Car Racing
# ============================================================
def play_car_racing():
    for i in range(20):
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True
        )

        pygame.init()
        scale = 5
        obs_shape = env.observation_space.shape
        screen_width = obs_shape[1] * scale
        screen_height = obs_shape[0] * scale
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Car Racing – стрелки: поворот+газ+тормоз, Esc для выхода")
        clock = pygame.time.Clock()

        actions_list = []
        observations_list = []

        obs, info = env.reset()
        running = True
        reward = -0.1
        while running:
            observations_list.append(obs)

            # Отрисовка
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            scaled_surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(scaled_surf, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
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

            action = np.array([steer, gas, brake, reward], dtype=np.float32)
            actions_list.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            print(reward)

            if terminated or truncated:
                obs, info = env.reset()

            clock.tick(30)

        observations_list.append(obs)

        os.makedirs("try", exist_ok=True)
        np.save("try/actions-try.npy", np.array(actions_list))
        np.save("try/observations-try.npy", np.array(observations_list))
        print(f"Сохранено {len(actions_list)} действий и {len(observations_list)} наблюдений.")

        env.close()
        pygame.quit()
        sys.exit()
        
# ============================================================
#  ViZDoom
# ============================================================
import vizdoom as vzd  # в начале файла

def play_doom():
    i = 0
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(False)  # только наше окно Pygame
    game.init()

    # --- Динамическое построение масок ---
    available_buttons = game.get_available_buttons()   # список vzd.Button
    num_buttons = len(available_buttons)
    print(f"Доступные кнопки: {[b.name for b in available_buttons]}")

    # Словарь: имя кнопки -> её индекс в общем массиве
    button_idx = {button: i for i, button in enumerate(available_buttons)}

    # Вспомогательная функция для создания маски
    def mask(*buttons):
        m = [0] * num_buttons
        for b in buttons:
            if b in button_idx:
                m[button_idx[b]] = 1
            else:
                print(f"⚠ Кнопка {b.name} не найдена!")
        return m

    # Короткие имена нужных кнопок
    F = vzd.Button.MOVE_FORWARD
    L = vzd.Button.TURN_LEFT
    R = vzd.Button.TURN_RIGHT
    A = vzd.Button.ATTACK

    action_masks = {
        0: mask(),          # ничего
        1: mask(F),         # вперёд
        2: mask(L),         # влево
        3: mask(R),         # вправо
        4: mask(A),         # выстрел
        5: mask(F, L),      # вперёд + влево
        6: mask(F, R),      # вперёд + вправо
    }

    action_names = [
        "ничего", "вперёд", "влево", "вправо",
        "выстрел", "вперёд+влево", "вперёд+вправо"
    ]

    # --- Pygame ---
    pygame.init()
    scale = 1

    game.new_episode()
    state = game.get_state()
    obs_shape = state.screen_buffer.shape
    screen_width = obs_shape[1] * scale
    screen_height = obs_shape[0] * scale
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("ViZDoom – стрелки + пробел, Esc")
    clock = pygame.time.Clock()

    actions_list = []
    observations_list = []
    running = True

    while running:
        i = i +1
        state = game.get_state()
        
        obs_resized = np.array(Image.fromarray(state.screen_buffer).resize((128, 128)))
        observations_list.append(obs_resized)


        # Отрисовка
        surf = pygame.surfarray.make_surface(np.transpose(obs_resized, (1, 0, 2)))
        scaled_surf = pygame.transform.scale(surf, (screen_width, screen_height))
        screen.blit(scaled_surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        forward = keys[pygame.K_UP]
        left = keys[pygame.K_LEFT]
        right = keys[pygame.K_RIGHT]
        shoot = keys[pygame.K_SPACE]

        if forward and left:
            action = 5
        elif forward and right:
            action = 6
        elif forward:
            action = 1
        elif left:
            action = 2
        elif right:
            action = 3
        elif shoot:
            action = 4
        else:
            action = 0

        print(i)
        actions_list.append(action)

        game.make_action(action_masks[action])

        if game.is_episode_finished():
            print(f"Эпизод завершён. Награда: {game.get_total_reward()}")
            game.new_episode()

        clock.tick(30)

    # Сохранение
    os.makedirs("try", exist_ok=True)
    np.save("try/doom-act.npy", np.array(actions_list, dtype=np.int32))
    np.save("try/doom-obs.npy", np.array(observations_list))
    print(f"Сохранено {len(actions_list)} действий и {len(observations_list)} наблюдений.")

    game.close()
    pygame.quit()
    sys.exit()


# ============================================================
#  Точка входа с выбором игры
# ============================================================
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
