"""Dream rollout through the CarRacing world model (VAE + latent predictor).

The scene starts from a SINGLE random latent ``z ~ N(0, I)`` in the VAE latent
space and unfolds token-by-token: at each step one new latent is sampled from
the most likely MDN component of ``PredictorTransformer``, decoded by the VAE
and displayed, while the causal window grows until it reaches ``max_len`` and
then slides forward. Latent maps are drawn on the left (exactly like
``vae_car``), the dreamed frame on the right at 192x192.

Steer/gas/brake with the arrow keys, Esc to quit.

Usage:
    uv run python games/play_in_dream.py [--vae-weights weights/CR/model_033] \\
        [--checkpoint-weights weights/predictor_car]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pygame
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import VAE
from games.carracing import _car_action
from games.common import _init_pygame, _poll_quit, _render, latent_to_rgb
from predictor import PredictorTransformer

DEVICE = "mps"
LAT_SCALE = 4       # latent map upscale for display
FRAME_SIZE = 192    # decoded frame display width
FINAL_SCALE = 2     # final window scale


def _auto_vae_weights():
    candidates = sorted(glob.glob("weights/CR/*"))
    if not candidates:
        candidates = sorted(glob.glob("weights/car*"))
    if not candidates:
        print("Не найдены веса VAE CarRacing в weights/ (укажите --vae-weights)")
        sys.exit(1)
    return candidates[-1]


def _recon_to_rgb(recon, final_activation):
    arr = recon.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    if final_activation == "tanh":
        arr = (arr + 1.0) / 2.0
    arr = arr.clip(0, 1)
    return (arr * 255).astype(np.uint8)


def _compose_view(lat_imgs, frame_rgb):
    lat_rows = []
    for img in lat_imgs:
        lat_rows.append(
            np.array(
                Image.fromarray(img).resize(
                    (img.shape[1] * LAT_SCALE, img.shape[0] * LAT_SCALE), Image.NEAREST
                )
            )
        )
    lat = np.concatenate(lat_rows, axis=0)
    lat_h, lat_w = lat.shape[:2]

    frame = np.asarray(frame_rgb).astype(np.uint8)
    fh, fw = frame.shape[:2]

    out_h = max(lat_h, fh)
    out_w = lat_w + fw
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:lat_h, :lat_w] = lat
    canvas[:fh, lat_w:] = frame
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Dream through the CarRacing world model")
    parser.add_argument("--vae-weights", default=None)
    parser.add_argument("--checkpoint-weights", default="weights/predictor_car")
    parser.add_argument("--temperature", type=float, default=2.5,
                        help="Sampling noise multiplier (default: 2.5)")
    args = parser.parse_args()

    vae = VAE.from_pretrained(args.vae_weights or _auto_vae_weights(), map_location="cpu").to(DEVICE)
    vae.eval()
    print(f"VAE: latent_dim={vae.latent_dim}, flat_latent={vae.flat_latent}, img_size={vae.img_size}")

    pred = PredictorTransformer.from_pretrained(args.checkpoint_weights, map_location=DEVICE).to(DEVICE)
    pred.eval()
    cfg = pred.config
    print(f"Predictor: d_model={cfg.d_model}, n_layer={cfg.n_layer}, "
          f"n_gaussians={cfg.n_gaussians}, max_len={cfg.max_len}")
    lat_shape = tuple(cfg.latent_shape)
    z_dim = cfg.z_dim
    S = cfg.max_len

    lat_h = lat_shape[-2] * LAT_SCALE
    lat_w = lat_h * lat_shape[0]
    canvas_w, canvas_h = lat_w + FRAME_SIZE, max(lat_h, FRAME_SIZE)
    screen, clock = _init_pygame((canvas_w, canvas_h), scale=FINAL_SCALE,
                 title="Dream CarRacing — стрелки: газ/поворот/тормоз, Esc")

    temp = args.temperature
    if temp != 1.0:
        print(f"temperature = {temp:.1f}")

    LOGVAR_OBS = -2.0   # fixed observation logvar (close to training mean -1.76)
    act_gas = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=DEVICE).view(1, 1, -1)

    def _step(mu_win, lv_win, act_win):
        """One autoregressive forward on a same-length window; return next best pair."""
        pi, mu_n, lv_n, _rew = pred(mu_win, lv_win, act_win, mode="all")
        g = pi[:, -1].argmax(dim=-1)
        idx = g[..., None, None, None].expand(g.shape[0], -1, *lat_shape)
        mu_b = mu_n[:, -1].gather(1, idx).squeeze(1)
        lv_b = lv_n[:, -1].gather(1, idx).squeeze(1)
        return mu_b, lv_b

    def _append(mu_win, lv_win, mu_b):
        tk = mu_b[:, None].to(mu_win.device)
        lv_tk = torch.full((1, 1, *lat_shape), LOGVAR_OBS, device=mu_win.device)
        mu_win = torch.cat([mu_win, tk], dim=1)
        lv_win = torch.cat([lv_win, lv_tk], dim=1)
        return mu_win, lv_win

    # -- Progressive fill: start from ONE random latent, grow the window to max_len,
    #    rendering each generated frame so the dream unfolds before your eyes.
    mu_win = torch.randn(1, 1, z_dim, device=DEVICE).view(1, 1, *lat_shape)
    logvar_win = torch.full((1, 1, *lat_shape), LOGVAR_OBS, device=DEVICE)
    act_win = act_gas.clone()

    for _i in range(S - 1):
        mu_b, lv_b = _step(mu_win, logvar_win, act_win)
        z_next = mu_b + torch.randn_like(mu_b) * torch.exp(0.5 * lv_b) * temp
        recon = vae.decode(z_next)
        mu_win, logvar_win = _append(mu_win, logvar_win, mu_b)
        act_win = torch.cat([act_win, act_gas], dim=1)

        frame_rgb = _recon_to_rgb(recon, vae.final_activation)
        frame_rgb = np.array(
            Image.fromarray(frame_rgb).resize((FRAME_SIZE, FRAME_SIZE), Image.LANCZOS)
        )
        canvas = _compose_view(latent_to_rgb(mu_b), frame_rgb)
        _render(screen, canvas)
        if _poll_quit():
            pygame.quit()
            return
        clock.tick(30)

    print("готово")

    # Show the first decoded state before entering the loop.
    with torch.no_grad():
        z_first = mu_win[:, -1]
        first_recon = vae.decode(z_first)
        first_rgb = _recon_to_rgb(first_recon, vae.final_activation)
        first_rgb = np.array(Image.fromarray(first_rgb).resize((FRAME_SIZE, FRAME_SIZE), Image.LANCZOS))
        canvas = _compose_view(latent_to_rgb(mu_win[:, -1]), first_rgb)
        _render(screen, canvas)

    running = True
    while running:
        with torch.no_grad():
            mu_b, lv_b = _step(mu_win, logvar_win, act_win)
            z_next = mu_b + torch.randn_like(mu_b) * torch.exp(0.5 * lv_b) * temp
            recon = vae.decode(z_next)

            # steady-state: roll the fixed-size window (drop oldest, append new)
            mu_win, logvar_win = _append(mu_win, logvar_win, mu_b)
            mu_win = mu_win[:, -S:]
            logvar_win = logvar_win[:, -S:]

            steer, gas, brake = _car_action(pygame.key.get_pressed())
            act_next = torch.tensor(
                [steer, gas, brake], dtype=torch.float32, device=DEVICE
            ).view(1, 1, -1)
            act_win = torch.cat([act_win, act_next], dim=1)[:, -S:]

        frame_rgb = _recon_to_rgb(recon, vae.final_activation)
        frame_rgb = np.array(
            Image.fromarray(frame_rgb).resize((FRAME_SIZE, FRAME_SIZE), Image.LANCZOS)
        )
        lat_imgs = latent_to_rgb(mu_b)
        canvas = _compose_view(lat_imgs, frame_rgb)

        _render(screen, canvas)

        if _poll_quit():
            break
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()