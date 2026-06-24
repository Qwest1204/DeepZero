import torch
import numpy as np
import pygame
from embedder import CNNVAE
from predictor import PredictorTransformer
from controller import Controller

vae = CNNVAE(in_channels=3, latent_dim=32, img_size=96).to('cpu')
vae.load_state_dict(torch.load('embedder/vaev5.pt', map_location='cpu'))
vae.eval()

predictor = PredictorTransformer(32, 8, 128, 3, 4, 4, 256, 4).to('cpu')
predictor.load_state_dict(torch.load('predictor/predictor_mdn.pt', map_location='cpu'))
predictor.eval()

#actor = Controller(64, 3)

pygame.init()
scale = 5
screen_width = 96 * scale
screen_height = 96 * scale
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Car Racing – комбинации стрелок (газ+поворот), Esc для выхода")
clock = pygame.time.Clock()
    
running = True
actions_list = []
observations_list = []

z_now = torch.randn(1, 32, dtype=torch.float32)

st = 0
while running:

    with torch.no_grad():
        obs = vae.decode(z_now).detach().squeeze(0).numpy()*255
        
    surf = pygame.surfarray.make_surface(np.transpose(obs, (2, 1, 0)))
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
    action = torch.tensor(np.array([steer, gas, brake], dtype=np.float32))
    
    if len(observations_list) >255:
        actions_list = actions_list[1:]
        observations_list = observations_list[1:]
    
    actions_list.append(action)
    observations_list.append(z_now)
    
    with torch.no_grad():
        A = torch.stack(actions_list).unsqueeze(0).to(torch.float32)
        Z = torch.stack(observations_list).permute(1, 0, 2)
        z_now = predictor.sample(*predictor(Z, A))[:, -1, :]

        
    clock.tick(20)