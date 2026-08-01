import os
import sys
import random

import gymnasium as gym
import numpy as np
import pygame
import vizdoom as vzd
from PIL import Image

SCREEN_SIZE = 192
DOOM_DIR = "try/Doom"
MW_DIR = "try/MW"
MW_CAM_KEEP = (1, 5, 7)   # user-numbered cameras (1-7) to save
MW_CAM_ROT = (1, 5)       # user-numbered cameras to rotate 180 deg
MW_MAX_STEPS = 2000       # episode length limit (default env limit is 500)

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


def _metaworld_action(keys, gripper_state):
    """Keyboard -> 4-dim continuous action (dx, dy, dz, gripper).

    WASD or arrows for dx/dy, Q/E or Up/Down for dz.
    """
    dx = 0.0
    dy = 0.0
    dz = 0.0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        dx = 0.5
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        dx = -0.5
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        dy = 0.5
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        dy = -0.5
    if keys[pygame.K_e]:
        dz = 0.5
    if keys[pygame.K_q]:
        dz = -0.5
    if keys[pygame.K_SPACE]:
        gripper_state = 1.0 - gripper_state  # toggle open/close
    return np.array([dx, dy, dz, gripper_state], dtype=np.float32), gripper_state


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

        if game.is_episode_finished():
            idx = random.randint(0, 1000000)
            np.save(f"{DOOM_DIR}/doom-act{idx}.npy", np.array(actions_list, dtype=np.int32))
            np.save(f"{DOOM_DIR}/doom-obs{idx}.npy", np.array(observations_list))
            print(f"Сохранено {len(actions_list)} действий и {len(observations_list)} наблюдений (doom{idx}).")
            actions_list = []
            observations_list = []
            game.new_episode()

        clock.tick(30)

    game.close()
    pygame.quit()


def play_metaworld():
    """Record human teleoperation for random MetaWorld V3 tasks (all 50).

    Renders and saves every 2nd frame (frames 0, 2, 4, ...) of 3 selected
    cameras: obs shape (M, C, 192, 192, 3), M = ceil(N/2). Actions, joints,
    rewards and success flags are saved at full rate (one per env step).
    """
    import metaworld
    from mujoco import Renderer

    task_names = sorted(metaworld.ALL_V3_ENVIRONMENTS)
    print(f"MetaWorld: доступно {len(task_names)} задач (V3)")

    screen, clock = _init_pygame(
        (SCREEN_SIZE, SCREEN_SIZE), scale=2,
        title="MetaWorld – WASD/стрелки dx/dy, Q/E dz, пробел = захват, 1-7 = камера, Esc",
    )
    os.makedirs(MW_DIR, exist_ok=True)

    running = True
    while running:
        task = random.choice(task_names)
        ml1 = metaworld.ML1(task)
        env = ml1.train_classes[task](render_mode="rgb_array")
        env.set_task(random.choice(ml1.train_tasks))
        env.max_path_length = MW_MAX_STEPS
        obs, _ = env.reset()
        cam_names = [env.model.cam(i).name for i in range(env.model.ncam)]
        keep_idx = [c - 1 for c in MW_CAM_KEEP if c - 1 < len(cam_names)]
        rot_mask = [c in MW_CAM_ROT for c in MW_CAM_KEEP if c - 1 < len(cam_names)]
        renderer = Renderer(env.model, SCREEN_SIZE, SCREEN_SIZE)
        main_cam = MW_CAM_KEEP[0]  # default user-numbered camera
        print(f"Эпизод: задача '{task}', obs_dim={obs.shape[0]}, "
              f"action_dim={env.action_space.shape[0]}, "
              f"сохраняем камеры {[cam_names[i] for i in keep_idx]}, "
              f"rot180={[cam_names[i] for i, r in zip(keep_idx, rot_mask) if r]}")

        actions_list = []
        observations_list = []
        joints_list = []
        success_list = []
        reward_list = []
        total_reward = 0.0
        gripper_state = 0.0
        episode_running = True
        step_count = 0
        success_logged = False

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

        while episode_running and running:
            # process events (keydown edges for gripper/camera, key state for movement)
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
                        gripper_state = 1.0 - gripper_state  # toggle on edge
                    if pygame.KSCAN_1 <= event.scancode <= pygame.KSCAN_9:
                        cam_num = event.scancode - pygame.KSCAN_1 + 1  # user numbering 1-9
                        if cam_num in MW_CAM_KEEP and cam_num - 1 < len(cam_names):
                            main_cam = cam_num
                            print(f"Камера: {cam_names[cam_num - 1]} ({cam_num})")
                elif event.type == pygame.KEYUP:
                    if event.scancode in key_state:
                        key_state[event.scancode] = False

            if not running:
                break

            frames = []
            for i, cam_id in enumerate(keep_idx):
                renderer.update_scene(env.data, camera=cam_id)
                frame = renderer.render()
                if rot_mask[i]:
                    frame = np.rot90(frame, 2)  # 180 deg
                frames.append(frame)
            frames = np.array(frames)  # (C_keep, 192, 192, 3) uint8
            if step_count % 2 == 0:
                observations_list.append(frames)  # save every 2nd frame
            joints_list.append(obs.astype(np.float32))
            _render(screen, frames[MW_CAM_KEEP.index(main_cam)])

            # build action from key states
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
            actions_list.append(action)

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            success = bool(info.get("success"))
            success_list.append(float(success))
            reward_list.append(float(reward))
            total_reward += reward
            pygame.display.set_caption(
                f"{task} | шаг {step_count} | reward {reward:+.3f} | сумма {total_reward:+.3f}"
            )
            if success and not success_logged:
                success_logged = True
                print(f"[{step_count}] Цель достигнута (success=True)!")
            if success:
                episode_running = False  # end episode on goal reached
            else:
                episode_running = not (terminated or truncated)

            clock.tick(30)

        if len(actions_list) > 0:
            idx = random.randint(0, 1000000)
            np.save(f"{MW_DIR}/metaworld-act{idx}-{task}.npy", np.array(actions_list, dtype=np.float32))
            np.save(f"{MW_DIR}/metaworld-obs{idx}-{task}.npy", np.array(observations_list))
            np.save(f"{MW_DIR}/metaworld-joints{idx}-{task}.npy", np.array(joints_list))
            np.save(f"{MW_DIR}/metaworld-success{idx}-{task}.npy", np.array(success_list, dtype=np.float32))
            np.save(f"{MW_DIR}/metaworld-reward{idx}-{task}.npy", np.array(reward_list, dtype=np.float32))
            print(f"Сохранено {len(actions_list)} действий и {len(observations_list)} наблюдений "
                  f"(mw{idx}-{task}, obs shape={np.array(observations_list).shape}, "
                  f"success={sum(success_list)}, total_reward={total_reward:.3f}).")

        env.close()

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        game = sys.argv[1].lower()
    else:
        print("Выберите игру:")
        print("  1 — CarRacing")
        print("  2 — ViZDoom")
        print("  3 — MetaWorld")
        game = input("Введите 1, 2 или 3: ").strip()

    if game in ("1", "car", "carracing", "car_racing"):
        play_car_racing()
    elif game in ("2", "doom", "vizdoom", "doom-viz"):
        play_doom()
    elif game in ("3", "mw", "metaworld"):
        play_metaworld()
    else:
        print(f"Неизвестный выбор: '{game}'. Запускаю CarRacing по умолчанию.")
        play_car_racing()