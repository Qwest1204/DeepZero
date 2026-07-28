import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0, "d_model должно делиться на num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        B, S, _ = query.shape

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out_proj(attn_output)


class SelfAttention2D(nn.Module):
    """Применяет MultiHeadAttention к фича-мэпу (B, C, H, W)."""

    def __init__(self, channels, num_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mha = MultiHeadAttention(channels, num_heads, dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_norm = self.norm(x_flat)
        attn_out = self.mha(x_norm, x_norm, x_norm)
        out = x_flat + attn_out
        out = out.transpose(1, 2).view(B, C, H, W)
        return out


class VAE(nn.Module):
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
        self.final_activation = final_activation

        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        if encoder_kernels is None:
            encoder_kernels = [4] * len(encoder_channels)
        if encoder_strides is None:
            encoder_strides = [2] * len(encoder_channels)
        if attention_layers is None:
            attention_layers = []

        assert len(encoder_channels) == len(encoder_kernels) == len(encoder_strides), (
            f"Длины encoder_channels ({len(encoder_channels)}), "
            f"encoder_kernels ({len(encoder_kernels)}), "
            f"encoder_strides ({len(encoder_strides)}) должны совпадать"
        )

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

        assert len(self.decoder_channels) == len(self.decoder_kernels) == len(self.decoder_strides), (
            f"Длины decoder_channels ({len(self.decoder_channels)}), "
            f"decoder_kernels ({len(self.decoder_kernels)}), "
            f"decoder_strides ({len(self.decoder_strides)}) должны совпадать"
        )

        self.decoder_fc = nn.Linear(latent_dim, encoder_final_size)
        self._decoder_blocks = self._build_decoder()

    def _default_decoder_params(self):
        channels = list(reversed(self.encoder_channels[:-1])) + [self.in_channels]
        kernels = list(reversed(self.encoder_kernels))
        strides = list(reversed(self.encoder_strides))
        return channels, kernels, strides

    def _build_encoder(self):
        blocks = nn.ModuleList()
        cur_channels = self.in_channels
        cur_h = cur_w = self.img_size
        sizes = [(cur_h, cur_w)]

        for i, (out_ch, k, s) in enumerate(
            zip(self.encoder_channels, self.encoder_kernels, self.encoder_strides)
        ):
            blocks.append(nn.Conv2d(cur_channels, out_ch, kernel_size=k, stride=s, padding=1))
            blocks.append(nn.ReLU())
            cur_h = (cur_h + 2 - k) // s + 1
            cur_w = (cur_w + 2 - k) // s + 1
            cur_channels = out_ch
            sizes.append((cur_h, cur_w))

            if i in self.attention_layers:
                blocks.append(SelfAttention2D(cur_channels, self.num_attention_heads))

        encoder_final_size = cur_channels * cur_h * cur_w
        return blocks, sizes, encoder_final_size

    def _build_decoder(self):
        blocks = nn.ModuleList()
        enc_len = len(self.encoder_channels)
        dec_len = len(self.decoder_channels)

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

            out_pad_h = target_h - ((sizes[enc_len - i][0] - 1) * s - 2 * 1 + (k - 1) + 1)
            out_pad_w = target_w - ((sizes[enc_len - i][1] - 1) * s - 2 * 1 + (k - 1) + 1)

            is_last = i == len(self.decoder_channels) - 1
            act = nn.Sigmoid() if is_last else nn.ReLU()

            blocks.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=k, stride=s, padding=1,
                    output_padding=(out_pad_h, out_pad_w),
                )
            )
            blocks.append(act)

        return blocks

    def encode(self, x):
        for block in self._encoder_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, self.encoder_channels[-1], self._enc_h, self._enc_w)
        for block in self._decoder_blocks:
            x = block(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    @staticmethod
    def loss_vae(recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def _config_dict(self):
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

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._config_dict(), f, indent=2, ensure_ascii=False)

        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, weights_path)

    @classmethod
    def from_pretrained(cls, save_dir, map_location="cpu"):
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(**config)
        state_dict = load_file(weights_path, device=str(map_location))
        model.load_state_dict(state_dict)
        return model