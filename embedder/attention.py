"""Attention blocks for 2D feature maps used inside the VAE encoder/decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention using PyTorch's scaled_dot_product_attention.

    Operates on sequences (B, S, d_model). Adapted from the DNN Building Blocks project.

    Args:
        d_model: Total feature dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        dropout: Dropout probability applied after attention.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
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
    """Self-attention applied to a 2D feature map (B, C, H, W).

    Reshapes the spatial dimensions into a sequence, applies MultiHeadAttention
    with pre-layer-norm and residual connection, then reshapes back.

    Args:
        channels: Number of input channels (C).
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mha = MultiHeadAttention(channels, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B, H*W, C)
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_norm = self.norm(x_flat)
        attn_out = self.mha(x_norm, x_norm, x_norm)
        out = x_flat + attn_out
        # (B, H*W, C) -> (B, C, H, W)
        out = out.transpose(1, 2).view(B, C, H, W)
        return out
