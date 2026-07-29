"""ResNet-based VAE with GroupNorm residual blocks and Upsample·Conv decoder.

Architecture inspired by SDXL KL-F8 / LDM VAE:
- Encoder: ``ResBlock`` × N → ``Downsample``
- Decoder: ``Upsample`` → ``ResBlock`` × N
- ``GroupNorm(32)`` inside each residual block for training stability
- ``nn.Upsample(scale=2, mode='nearest')`` + ``Conv2d(3,3)`` replaces
  ``ConvTranspose2d`` to avoid checkerboard artifacts and ``output_padding``
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from embedder.attention import SelfAttention2D


# ---------------------------------------------------------------------------
# ResBlock — core building block
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Pre-norm residual block with GroupNorm + ReLU + Conv3x3 × 2.

    ``GroupNorm(32)`` is applied before each convolution. A ``Conv2d(1×1)``
    skip connection is added when ``in_channels != out_channels``.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        norm_groups: Number of groups in GroupNorm (default 32).
    """

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 32):
        super().__init__()
        ng1 = min(norm_groups, in_channels)
        ng2 = min(norm_groups, out_channels)
        self.norm1 = nn.GroupNorm(ng1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(ng2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + shortcut


# ---------------------------------------------------------------------------
# Downsample & Upsample helpers
# ---------------------------------------------------------------------------

class DownsampleBlock(nn.Module):
    """Strided convolution for spatial downsampling.

    Uses ``Conv2d(k=4, s=2, p=1)`` — same receptive field as the original VAE.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Nearest-neighbour upsample → Conv2d(k=3, s=1) to anti-alias.

    Replaces ``ConvTranspose2d`` — no checkerboard artifacts, no output_padding.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# ResVAE — full model
# ---------------------------------------------------------------------------

class ResVAE(nn.Module):
    """ResNet-based VAE with GroupNorm residual blocks and Upsample·Conv decoder.

    Mirrors the ``VAE`` interface (``encode``, ``decode``, ``forward``,
    ``loss_vae``, ``save_pretrained``, ``from_pretrained``) for drop-in
    replacement.

    Args:
        in_channels: Number of input image channels.
        latent_dim: Dimensionality of the latent vector ``z``.
        img_size: Spatial size of the squared input image.
        encoder_channels: List of output channels for each encoder stage.
        decoder_channels: List of output channels for each decoder stage.
        attention_layers: Indices of encoder stages after which to insert
            ``SelfAttention2D``. The decoder mirrors these positions.
        num_attention_heads: Number of heads in each attention block.
        resnet_blocks_per_stage: Number of ``ResBlock2D`` per stage (1 or 2).
        norm_groups: GroupNorm group count inside each ResBlock.
        final_activation: ``"sigmoid"`` or ``"tanh"``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        img_size: int = 96,
        encoder_channels: list | None = None,
        decoder_channels: list | None = None,
        attention_layers: list | None = None,
        num_attention_heads: int = 4,
        resnet_blocks_per_stage: int = 1,
        norm_groups: int = 32,
        final_activation: str = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_attention_heads = num_attention_heads
        self.resnet_blocks_per_stage = resnet_blocks_per_stage
        self.norm_groups = norm_groups
        self.final_activation = final_activation

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 256]
        if decoder_channels is None:
            decoder_channels = list(reversed(encoder_channels[:-1])) + [in_channels]
        if attention_layers is None:
            attention_layers = []

        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.attention_layers = attention_layers

        # ---- Encoder ----
        self.enc_blocks, self._enc_out_size, self.enc_init_conv = self._build_encoder()
        self._enc_h, self._enc_w = self._enc_out_size

        flat_dim = encoder_channels[-1] * self._enc_h * self._enc_w
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)

        # ---- Decoder ----
        self.dec_blocks = self._build_decoder()

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _build_encoder(self):
        blocks = nn.ModuleList()
        first_ch = self.encoder_channels[0]
        init_conv = nn.Conv2d(self.in_channels, first_ch, kernel_size=3, padding=1)
        ch = first_ch
        h = w = self.img_size

        for stage_idx, out_ch in enumerate(self.encoder_channels):
            for _ in range(self.resnet_blocks_per_stage):
                blocks.append(ResBlock2D(ch, out_ch, self.norm_groups))
                ch = out_ch

            blocks.append(DownsampleBlock(ch, out_ch))
            h = (h + 2 - 4) // 2 + 1
            w = (w + 2 - 4) // 2 + 1

            if stage_idx in self.attention_layers:
                blocks.append(SelfAttention2D(out_ch, self.num_attention_heads))

        return blocks, (h, w), init_conv

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _build_decoder(self):
        blocks = nn.ModuleList()
        enc_len = len(self.encoder_channels)
        ch = self.encoder_channels[-1]
        h, w = self._enc_out_size

        for stage_idx, out_ch in enumerate(self.decoder_channels):
            blocks.append(UpsampleBlock(ch, out_ch))
            h *= 2
            w *= 2

            enc_attn_idx = enc_len - 1 - stage_idx
            if enc_attn_idx in self.attention_layers:
                blocks.append(SelfAttention2D(out_ch, self.num_attention_heads))

            for _ in range(self.resnet_blocks_per_stage):
                blocks.append(ResBlock2D(out_ch, out_ch, self.norm_groups))
                ch = out_ch

            if stage_idx == len(self.decoder_channels) - 1:
                act = nn.Tanh() if self.final_activation == "tanh" else nn.Sigmoid()
                blocks.append(act)

        return blocks

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.enc_init_conv(x)
        for block in self.enc_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, self.encoder_channels[-1], self._enc_h, self._enc_w)
        for block in self.dec_blocks:
            x = block(x)
        return x

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
            "decoder_channels": self.decoder_channels,
            "attention_layers": self.attention_layers,
            "num_attention_heads": self.num_attention_heads,
            "resnet_blocks_per_stage": self.resnet_blocks_per_stage,
            "norm_groups": self.norm_groups,
            "final_activation": self.final_activation,
        }

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self._config_dict(), f, indent=2, ensure_ascii=False)
        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, os.path.join(save_dir, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, save_dir: str, map_location: str = "cpu") -> "ResVAE":
        with open(os.path.join(save_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        model = cls(**config)
        state_dict = load_file(os.path.join(save_dir, "model.safetensors"), device=str(map_location))
        model.load_state_dict(state_dict)
        return model
