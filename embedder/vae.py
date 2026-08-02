"""Unified VAE supporting both plain ConvVAE and ResVAE architectures.

Configurable via constructor parameters:
- ``flat_latent=True``  — fc bridge (legacy, backward-compat)
- ``flat_latent=False`` — conv bridge → square latent map (B, C, H, W)
- ``use_resblocks=False`` — plain Conv2d+activation (lightweight, CPU-friendly)
- ``use_resblocks=True``  — ResBlock2D+GroupNorm (stable, detail-preserving)
- ``use_attention=True``  — SelfAttention2D at selected encoder stages
- ``hidden_activation``  — activation in hidden layers (default: ``"relu"``)
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from embedder.attention import SelfAttention2D


# ---------------------------------------------------------------------------
# Hidden activation registry (residual-block fn vs plain-module)
# ---------------------------------------------------------------------------

_HIDDEN_ACTIVATION_FNS = {
    "relu": F.relu,
    "silu": F.silu,
    "gelu": F.gelu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.2),
    "elu": F.elu,
}

_HIDDEN_ACTIVATION_MODULES = {
    "relu": lambda: nn.ReLU(),
    "silu": lambda: nn.SiLU(),
    "gelu": lambda: nn.GELU(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.2),
    "elu": lambda: nn.ELU(),
}

_SUPPORTED_ACTIVATIONS = tuple(_HIDDEN_ACTIVATION_FNS)


# ---------------------------------------------------------------------------
# ResBlock — optional residual building block
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """Pre-norm residual block with GroupNorm + activation + Conv3x3 x 2."""

    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 32,
                 activation: str = "relu"):
        super().__init__()
        ng1 = min(norm_groups, in_channels)
        ng2 = min(norm_groups, out_channels)
        self.norm1 = nn.GroupNorm(ng1, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(ng2, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = _HIDDEN_ACTIVATION_FNS.get(activation, F.relu)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.skip(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + shortcut


class DownsampleBlock(nn.Module):
    """Strided convolution: Conv2d(k=4, s=2, p=1)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """Nearest upsample + Conv2d(k=3) — no checkerboard artifacts."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


# ---------------------------------------------------------------------------
# VAE — unified
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """Unified VAE supporting plain Conv2d or ResBlock encoders and flat or spatial latents.

    Args:
        in_channels: Input image channels.
        latent_dim: Flat latent dimension (``flat_latent=True``) or number of
            latent channels (``flat_latent=False``).
        img_size: Squared input spatial size.
        flat_latent: If ``True`` use fc bridge (legacy), else conv_mu/conv_logvar
            producing a spatial latent map.
        encoder_channels: List of output channels per encoder stage.
        decoder_channels: List of output channels per decoder stage
            (auto-computed if ``None``).
        use_resblocks: If ``True`` use ResBlock2D+GroupNorm, else plain Conv2d+ReLU.
        use_attention: If ``True`` insert SelfAttention2D at selected stages.
        attention_layers: Indices of encoder stages with attention (e.g. ``[3]``).
        num_attention_heads: Heads per SelfAttention2D block.
        res_blocks_per_stage: Number of ResBlock2D per stage (ignored if
            ``use_resblocks=False``).
        norm_groups: GroupNorm groups per ResBlock2D.
        final_activation: ``"sigmoid"`` or ``"tanh"``.
        hidden_activation: Activation in hidden layers (relu/silu/gelu/
            leaky_relu/elu). Default ``"relu"`` keeps old checkpoints intact.
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        img_size: int = 256,
        flat_latent: bool = False,
        encoder_channels: list | None = None,
        decoder_channels: list | None = None,
        use_resblocks: bool = False,
        use_attention: bool = False,
        attention_layers: list | None = None,
        num_attention_heads: int = 4,
        res_blocks_per_stage: int = 1,
        norm_groups: int = 32,
        final_activation: str = "sigmoid",
        hidden_activation: str = "relu",
        **kwargs,  # backward compat
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.flat_latent = flat_latent
        self.use_resblocks = use_resblocks
        self.use_attention = use_attention
        self.attention_layers = attention_layers or []
        self.num_attention_heads = num_attention_heads
        self.res_blocks_per_stage = res_blocks_per_stage
        self.norm_groups = norm_groups
        self.final_activation = final_activation
        if hidden_activation not in _HIDDEN_ACTIVATION_FNS:
            raise ValueError(
                f"Неизвестная hidden_activation: {hidden_activation!r}. "
                f"Доступно: {_SUPPORTED_ACTIVATIONS}"
            )
        self.hidden_activation = hidden_activation

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256, 256, 256]
        self.encoder_channels = encoder_channels

        # ---- Build encoder ----
        if use_resblocks:
            self.enc_init_conv = nn.Conv2d(in_channels, encoder_channels[0], kernel_size=3, padding=1)
            self.enc_blocks = self._build_encoder_res()
        else:
            self.enc_init_conv = nn.Identity()
            self.encoder = self._build_encoder_plain()

        # ---- Latent spatial size after encoder ----
        latent_h, latent_w = self._compute_latent_spatial()
        self.latent_h = latent_h
        self.latent_w = latent_w
        flat_size = encoder_channels[-1] * latent_h * latent_w

        # ---- Bridge: flat fc or spatial conv ----
        if flat_latent:
            self.fc_mu = nn.Linear(flat_size, latent_dim)
            self.fc_logvar = nn.Linear(flat_size, latent_dim)
            self.decoder_fc = nn.Linear(latent_dim, flat_size)
            self.decoder_start_ch = encoder_channels[-1]
        else:
            self.conv_mu = nn.Conv2d(encoder_channels[-1], latent_dim, kernel_size=1)
            self.conv_logvar = nn.Conv2d(encoder_channels[-1], latent_dim, kernel_size=1)
            self.decoder_fc = nn.Identity()
            self.decoder_start_ch = latent_dim

        # ---- Build decoder ----
        if decoder_channels is None:
            decoder_channels = list(reversed(encoder_channels[1:])) + [in_channels]
        self.decoder_channels = decoder_channels

        if use_resblocks:
            self.decoder = self._build_decoder_res()
        else:
            self.decoder = self._build_decoder_plain()

    # ------------------------------------------------------------------
    # Encoder builders
    # ------------------------------------------------------------------

    def _build_encoder_plain(self) -> nn.Sequential:
        layers = []
        ch = self.in_channels
        for out_ch in self.encoder_channels:
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=4, stride=2, padding=1))
            layers.append(_HIDDEN_ACTIVATION_MODULES[self.hidden_activation]())
            ch = out_ch
        return nn.Sequential(*layers)

    def _build_encoder_res(self) -> nn.ModuleList:
        blocks = nn.ModuleList()
        ch = self.encoder_channels[0]
        for stage_idx, out_ch in enumerate(self.encoder_channels):
            for _ in range(self.res_blocks_per_stage):
                blocks.append(ResBlock2D(ch, out_ch, self.norm_groups, self.hidden_activation))
                ch = out_ch
            blocks.append(DownsampleBlock(ch, out_ch))
            if self.use_attention and stage_idx in self.attention_layers:
                blocks.append(SelfAttention2D(out_ch, self.num_attention_heads))
        return blocks

    # ------------------------------------------------------------------
    # Decoder builders
    # ------------------------------------------------------------------

    def _build_decoder_plain(self) -> nn.Sequential:
        layers = []
        ch = self.decoder_start_ch
        for i, out_ch in enumerate(self.decoder_channels):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
            if i < len(self.decoder_channels) - 1:
                layers.append(_HIDDEN_ACTIVATION_MODULES[self.hidden_activation]())
            else:
                layers.append(
                    nn.Sigmoid() if self.final_activation == "sigmoid" else nn.Tanh()
                )
            ch = out_ch
        return nn.Sequential(*layers)

    def _build_decoder_res(self) -> nn.ModuleList:
        blocks = nn.ModuleList()
        ch = self.decoder_start_ch
        enc_len = len(self.encoder_channels)
        for stage_idx, out_ch in enumerate(self.decoder_channels):
            blocks.append(UpsampleBlock(ch, out_ch))
            ch = out_ch
            enc_attn_idx = enc_len - 1 - stage_idx
            if self.use_attention and enc_attn_idx in self.attention_layers:
                blocks.append(SelfAttention2D(out_ch, self.num_attention_heads))
            for _ in range(self.res_blocks_per_stage):
                blocks.append(ResBlock2D(out_ch, out_ch, self.norm_groups, self.hidden_activation))
            if stage_idx == len(self.decoder_channels) - 1:
                act = nn.Tanh() if self.final_activation == "tanh" else nn.Sigmoid()
                blocks.append(act)
        return blocks

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------

    def _compute_latent_spatial(self) -> tuple[int, int]:
        h = w = self.img_size
        for _ in self.encoder_channels:
            h = (h + 2 - 4) // 2 + 1
            w = (w + 2 - 4) // 2 + 1
        return h, w

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_resblocks:
            x = self.enc_init_conv(x)
            for block in self.enc_blocks:
                x = block(x)
        else:
            x = self.encoder(x)
        return x

    def _run_decoder(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_resblocks:
            for block in self.decoder:
                x = block(x)
        else:
            x = self.decoder(x)
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._run_encoder(x)
        if self.flat_latent:
            x = x.view(x.size(0), -1)
            return self.fc_mu(x), self.fc_logvar(x)
        else:
            return self.conv_mu(x), self.conv_logvar(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            if self.flat_latent:
                x = self.decoder_fc(z)
                x = x.view(-1, self.encoder_channels[-1], self.latent_h, self.latent_w)
            else:
                x = z.view(-1, self.latent_dim, self.latent_h, self.latent_w)
        else:
            x = z
        return self._run_decoder(x)

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
            "flat_latent": self.flat_latent,
            "encoder_channels": self.encoder_channels,
            "decoder_channels": self.decoder_channels,
            "use_resblocks": self.use_resblocks,
            "use_attention": self.use_attention,
            "attention_layers": self.attention_layers,
            "num_attention_heads": self.num_attention_heads,
            "res_blocks_per_stage": self.res_blocks_per_stage,
            "norm_groups": self.norm_groups,
            "final_activation": self.final_activation,
            "hidden_activation": self.hidden_activation,
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
        if "flat_latent" not in config:
            config["flat_latent"] = True
        if "hidden_activation" not in config:
            config["hidden_activation"] = "relu"
        model = cls(**config)
        state_dict = load_file(os.path.join(save_dir, "model.safetensors"), device=str(map_location))
        model.load_state_dict(state_dict)
        return model
