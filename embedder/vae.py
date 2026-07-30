"""Lightweight ConvVAE — 6-stage Conv2d(stride=2)+ReLU encoder, Upsample+Conv decoder.

Inspired by the World Models VAE (Ha & Schmidhuber, 2018).
Optimised for CPU inference — no GroupNorm, no SelfAttention, no skip connections.
Preserves the same public API: ``encode``, ``decode``, ``forward``, ``loss_vae``,
``save_pretrained``, ``from_pretrained``.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class VAE(nn.Module):
    """ConvVAE with strided convolutions and nearest-neighbour upsampling.

    Args:
        in_channels: Number of input image channels (default 3).
        latent_dim: Dimensionality of the flat latent vector ``z`` (default 768).
        img_size: Spatial size of the squared input image (default 256).
        encoder_channels: Output channels per encoder stage.
            Default ``[32, 64, 128, 256, 256, 256]`` → 6×downsample 256→4.
        final_activation: ``"sigmoid"`` (default) or ``"tanh"``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 768,
        img_size: int = 256,
        encoder_channels: list | None = None,
        final_activation: str = "sigmoid",
        **kwargs,  # ignored — backward compat with old config.json
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.final_activation = final_activation

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 256, 256]
        self.encoder_channels = encoder_channels

        # ---- Encoder ----
        enc_layers = []
        ch = in_channels
        h = w = img_size
        for out_ch in encoder_channels:
            enc_layers.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            enc_layers.append(nn.ReLU())
            ch = out_ch
            h = (h + 2 - 4) // 2 + 1
            w = (w + 2 - 4) // 2 + 1
        self.encoder = nn.Sequential(*enc_layers)
        self._enc_h, self._enc_w = h, w

        flat_dim = ch * h * w
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)

        # ---- Decoder ----
        dec_channels = list(reversed(encoder_channels[1:])) + [in_channels]
        self.decoder_channels = dec_channels
        dec_layers = []
        for i, out_ch in enumerate(dec_channels):
            dec_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            dec_layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
            if i < len(dec_channels) - 1:
                dec_layers.append(nn.ReLU())
            else:
                dec_layers.append(
                    nn.Sigmoid() if final_activation == "sigmoid" else nn.Tanh()
                )
            ch = out_ch
        self.decoder = nn.Sequential(*dec_layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, self.encoder_channels[-1], self._enc_h, self._enc_w)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss_vae(recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _config_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "latent_dim": self.latent_dim,
            "img_size": self.img_size,
            "encoder_channels": self.encoder_channels,
            "final_activation": self.final_activation,
        }

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self._config_dict(), f, indent=2, ensure_ascii=False)
        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, os.path.join(save_dir, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, save_dir: str, map_location: str = "cpu") -> "VAE":
        with open(os.path.join(save_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = load_file(os.path.join(save_dir, "model.safetensors"), device=str(map_location))
        model.load_state_dict(state_dict)
        return model