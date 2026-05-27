import gymnasium as gym
import numpy as np
import pygame
import sys

def main():
    for i in range(20):
        # Включаем непрерывное управление, чтобы комбинировать газ/тормоз и поворот
        env = gym.make(
            "CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True               # <-- теперь можно одновременно делать несколько действий
        )

        pygame.init()
        scale = 5
        obs_shape = env.observation_space.shape
        screen_width = obs_shape[1] * scale
        screen_height = obs_shape[0] * scale
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Car Racing – комбинации стрелок (газ+поворот), Esc для выхода")
        clock = pygame.time.Clock()

        actions_list = []
        observations_list = []

        obs, info = env.reset()
        running = True
        reward = -0.1
        while running:
            observations_list.append(obs)

            # Отрисовка текущего кадра
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            scaled_surf = pygame.transform.scale(surf, (screen_width, screen_height))
            screen.blit(scaled_surf, (0, 0))
            pygame.display.flip()

            # Обработка событий выхода
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # Собираем нажатые клавиши
            keys = pygame.key.get_pressed()
            steer = 0.0
            gas = 0.0
            brake = 0.0

            # Поворот (влево/вправо)
            if keys[pygame.K_LEFT]:
                steer = -1.0
            if keys[pygame.K_RIGHT]:
                steer = 1.0
            # Газ и тормоз (можно нажать оба, тогда приоритет у тормоза, как в реальной машине)
            if keys[pygame.K_UP]:
                gas = 1.0
            if keys[pygame.K_DOWN]:
                brake = 1.0
                gas = 0.0   # тормоз перекрывает газ

            # Формируем непрерывное действие: [руль, газ, тормоз]
            action = np.array([steer, gas, brake, reward], dtype=np.float32)

            actions_list.append(action)

            # Шаг среды
            obs, reward, terminated, truncated, info = env.step(action)
            print(reward)

            if terminated or truncated:
                obs, info = env.reset()

            clock.tick(30)  # FPS

        observations_list.append(obs)

        # Сохраняем действия и наблюдения
        np.save(f"try/actions-try.npy", np.array(actions_list))  # action — массив, поэтому object
        np.save(f"try/observations-try.npy.npy", np.array(observations_list))
        print(f"Сохранено {len(actions_list)} действий и {len(observations_list)} наблюдений.")

        env.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()