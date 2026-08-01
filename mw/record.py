"""Record human teleoperation for MetaWorld V3 tasks.

Renders and saves every 2nd frame (frames 0, 2, 4, ...) of 3 selected
cameras: obs shape (M, C, 192, 192, 3), M = ceil(N/2). Actions, joints,
rewards and success flags are saved at full rate (one per env step).
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from games.common import SCREEN_SIZE, _init_pygame, _render

MW_DIR = "try/MW"
MW_CAM_KEEP = (1, 5, 7)   # user-numbered cameras (1-7) to save
MW_CAM_ROT = (1, 5)       # user-numbered cameras to rotate 180 deg
MW_MAX_STEPS = 2000       # episode length limit (default env limit is 500)


def play_metaworld():
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
    play_metaworld()
