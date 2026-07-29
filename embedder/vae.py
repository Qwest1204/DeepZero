"""Variational Autoencoder (VAE) for compressing game frames into latent vectors.

The VAE uses a configurable CNN encoder/decoder with optional self-attention
layers at user-specified depth indices. Weights are saved in safetensors format
alongside a JSON config for architecture reconstruction.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from embedder.attention import SelfAttention2D


class VAE(nn.Module):
    """Configurable convolutional VAE with optional 2D self-attention.

    The encoder downsamples the input image through a stack of Conv2d + ReLU
    layers, optionally followed by SelfAttention2D at specified layer indices.
    The flattened feature map is projected to a Gaussian latent ``(mu, logvar)``.

    The decoder mirrors the encoder layout: ``nn.ConvTranspose2d`` layers with
    zero or more SelfAttention2D blocks inserted at mirrored positions.
    ``output_padding`` is automatically computed to guarantee the output spatial
    size matches the input.

    Args:
        in_channels: Number of input image channels.
        latent_dim: Dimensionality of the latent vector ``z``.
        img_size: Spatial size of the squared input image.
        encoder_channels: List of output channels for each encoder conv layer.
        encoder_kernels: Kernel sizes for each encoder conv (default all 4).
        encoder_strides: Strides for each encoder conv (default all 2).
        decoder_channels: List of output channels for each decoder deconv layer.
            When ``None`` mirrors the encoder layout.
        decoder_kernels / decoder_strides: Same as above for the decoder.
        attention_layers: Indices of encoder layers after which to insert
            ``SelfAttention2D``. The decoder mirrors these positions.
        num_attention_heads: Number of heads in each ``SelfAttention2D`` block.
        final_activation: Output activation — ``"sigmoid"`` for [0, 1] pixel
            range, or ``"tanh"`` for [-1, 1].
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        img_size: int = 96,
        encoder_channels: list | None = None,
        encoder_kernels: list | None = None,
        encoder_strides: list | None = None,
        decoder_channels: list | None = None,
        decoder_kernels: list | None = None,
        decoder_strides: list | None = None,
        attention_layers: list | None = None,
        num_attention_heads: int = 4,
        final_activation: str = "sigmoid",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_attention_heads = num_attention_heads
        assert final_activation in ("sigmoid", "tanh"), f"unsupported final_activation: {final_activation}"
        self.final_activation = final_activation

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        if encoder_kernels is None:
            encoder_kernels = [4] * len(encoder_channels)
        if encoder_strides is None:
            encoder_strides = [2] * len(encoder_channels)
        if attention_layers is None:
            attention_layers = []

        assert len(encoder_channels) == len(encoder_kernels) == len(encoder_strides)

        self.encoder_channels = encoder_channels
        self.encoder_kernels = encoder_kernels
        self.encoder_strides = encoder_strides
        self.attention_layers = attention_layers

        self._encoder_blocks, enc_spatial_sizes, encoder_final_size = self._build_encoder()
        self._enc_h, self._enc_w = enc_spatial_sizes[-1]
        self._enc_spatial_sizes = enc_spatial_sizes

        self.fc_mu = nn.Linear(encoder_final_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_final_size, latent_dim)

        self.decoder_channels, self.decoder_kernels, self.decoder_strides = (
            self._default_decoder_params()
            if decoder_channels is None
            else (decoder_channels, decoder_kernels, decoder_strides)
        )

        assert len(self.decoder_channels) == len(self.decoder_kernels) == len(self.decoder_strides)

        self.decoder_fc = nn.Linear(latent_dim, encoder_final_size)
        self._decoder_blocks = self._build_decoder()

    # ------------------------------------------------------------------
    # Architecture construction helpers
    # ------------------------------------------------------------------

    def _default_decoder_params(self):
        """Return decoder channel / kernel / stride lists that mirror the encoder."""
        channels = list(reversed(self.encoder_channels[:-1])) + [self.in_channels]
        kernels = list(reversed(self.encoder_kernels))
        strides = list(reversed(self.encoder_strides))
        return channels, kernels, strides

    @staticmethod
    def _conv_out_size(in_size: int, kernel: int, stride: int) -> int:
        return (in_size + 2 - kernel) // stride + 1

    def _build_encoder(self):
        """Build encoder module list and record intermediate spatial sizes.

        Returns:
            blocks: ``nn.ModuleList`` containing Conv2d, ReLU and optionally
                SelfAttention2D modules.
            sizes: List of ``(h, w)`` after each conv (including the initial
                input size at index 0).
            final_flat_size: Total elements after flattening.
        """
        blocks = nn.ModuleList()
        cur_channels = self.in_channels
        cur_h = cur_w = self.img_size
        sizes = [(cur_h, cur_w)]

        for i, (out_ch, k, s) in enumerate(
            zip(self.encoder_channels, self.encoder_kernels, self.encoder_strides)
        ):
            blocks.append(nn.Conv2d(cur_channels, out_ch, kernel_size=k, stride=s, padding=1))
            blocks.append(nn.ReLU())
            cur_h = self._conv_out_size(cur_h, k, s)
            cur_w = self._conv_out_size(cur_w, k, s)
            cur_channels = out_ch
            sizes.append((cur_h, cur_w))

            if i in self.attention_layers:
                blocks.append(SelfAttention2D(cur_channels, self.num_attention_heads))

        final_flat_size = cur_channels * cur_h * cur_w
        return blocks, sizes, final_flat_size

    def _build_decoder(self):
        """Build decoder module list with mirrored attention + output_padding.

        ``output_padding`` is computed per layer so that the upsampled spatial
        size exactly matches the corresponding encoder layer's input size.
        """
        blocks = nn.ModuleList()
        enc_len = len(self.encoder_channels)
        dec_len = len(self.decoder_channels)

        # Mirror attention indices: encoder layer ``i`` → decoder layer ``enc_len - 1 - i``
        dec_attention = {
            enc_len - 1 - idx
            for idx in self.attention_layers
            if 0 <= enc_len - 1 - idx < dec_len
        }

        sizes = self._enc_spatial_sizes

        for i, (out_ch, k, s) in enumerate(
            zip(self.decoder_channels, self.decoder_kernels, self.decoder_strides)
        ):
            in_ch = self.encoder_channels[-1] if i == 0 else self.decoder_channels[i - 1]

            target_h, target_w = sizes[enc_len - 1 - i]

            if i in dec_attention:
                blocks.append(SelfAttention2D(in_ch, self.num_attention_heads))

            dec_in_h, dec_in_w = sizes[enc_len - i]
            out_pad_h = target_h - ((dec_in_h - 1) * s - 2 + k)
            out_pad_w = target_w - ((dec_in_w - 1) * s - 2 + k)

            is_last = i == len(self.decoder_channels) - 1
            if is_last:
                act = nn.Tanh() if self.final_activation == "tanh" else nn.Sigmoid()
            else:
                act = nn.ReLU()

            blocks.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=k, stride=s, padding=1,
                    output_padding=(out_pad_h, out_pad_w),
                )
            )
            blocks.append(act)

        return blocks

    # ------------------------------------------------------------------
    # Core VAE methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input into latent Gaussian parameters.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            mu: Mean of the latent posterior ``(B, latent_dim)``.
            logvar: Log-variance of the latent posterior ``(B, latent_dim)``.
        """
        for block in self._encoder_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: ``z = mu + eps * exp(logvar / 2)``."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent vector back into an image.

        Args:
            z: Latent vector of shape ``(B, latent_dim)``.

        Returns:
            Reconstructed image ``(B, C, H, W)``.
        """
        x = self.decoder_fc(z)
        x = x.view(-1, self.encoder_channels[-1], self._enc_h, self._enc_w)
        for block in self._decoder_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass: encode -> reparameterise -> decode.

        Returns:
            recon_x: Reconstructed image ``(B, C, H, W)``.
            mu: Latent mean ``(B, latent_dim)``.
            logvar: Latent log-variance ``(B, latent_dim)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def loss_vae(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE loss: MSE reconstruction + beta-weighted KL divergence.

        Args:
            recon_x: Reconstructed image ``(B, C, H, W)``.
            x: Original input image ``(B, C, H, W)``.
            mu: Latent mean ``(B, latent_dim)``.
            logvar: Latent log-variance ``(B, latent_dim)``.
            beta: Weight for the KL term (beta-VAE).

        Returns:
            total_loss: ``recon_loss + beta * kl_loss``.
            recon_loss: Sum of pixel-wise MSE (unnormalised).
            kl_loss: KL divergence between posterior and standard normal prior.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    # ------------------------------------------------------------------
    # Serialisation: config.json + model.safetensors
    # ------------------------------------------------------------------

    def _config_dict(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "latent_dim": self.latent_dim,
            "img_size": self.img_size,
            "encoder_channels": self.encoder_channels,
            "encoder_kernels": self.encoder_kernels,
            "encoder_strides": self.encoder_strides,
            "decoder_channels": self.decoder_channels,
            "decoder_kernels": self.decoder_kernels,
            "decoder_strides": self.decoder_strides,
            "attention_layers": self.attention_layers,
            "num_attention_heads": self.num_attention_heads,
            "final_activation": self.final_activation,
        }

    def save_pretrained(self, save_dir: str):
        """Save model weights (safetensors) and architecture config (JSON).

        Creates ``save_dir/config.json`` and ``save_dir/model.safetensors``.
        The directory is created if it does not exist.
        """
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._config_dict(), f, indent=2, ensure_ascii=False)

        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, weights_path)

    @classmethod
    def from_pretrained(cls, save_dir: str, map_location: str = "cpu") -> "VAE":
        """Load a model from a previously saved ``save_pretrained`` directory.

        Reads ``config.json`` to reconstruct the architecture, then loads
        ``model.safetensors`` and restores the state dict.

        Args:
            save_dir: Path to the directory containing ``config.json`` and
                ``model.safetensors``.
            map_location: Torch device string to load weights onto.

        Returns:
            A ``VAE`` instance with pretrained weights loaded.
        """
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(**config)
        state_dict = load_file(weights_path, device=str(map_location))
        model.load_state_dict(state_dict)
        return model
