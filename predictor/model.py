"""Predictor Transformer with MDN head for dynamics prediction in latent space.

Given a sequence of latent vectors ``z`` and corresponding actions, the model
predicts the distribution of the next latent ``z_{t+1}`` as a Mixture of Gaussians.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from embedder.attention import MultiHeadAttention


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (no learnable parameters).

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length to pre-compute.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SimpleFFN(nn.Module):
    """Two-layer feed-forward network with GELU activation.

    Used as the FFN sub-layer inside each Transformer block.

    Args:
        d_model: Input / output dimension.
        d_ff: Hidden dimension (typically 4 * d_model).
        dropout: Dropout probability after activation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block with causal self-attention.

    Architecture: Norm → MHA (causal) → residual → Norm → FFN → residual.

    Args:
        d_model: Model dimension.
        num_heads: Number of attention heads.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SimpleFFN(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        causal_x = self.norm1(x)
        causal_x = self.mha(causal_x, causal_x, causal_x, is_causal=True)
        x = x + self.dropout1(causal_x)

        ff_x = self.norm2(x)
        ff_x = self.ffn(ff_x)
        x = x + self.dropout2(ff_x)
        return x


class PredictorTransformer(nn.Module):
    """Transformer-based dynamics predictor with a Mixture-of-Gaussians head.

    Embeds the concatenation ``[z_t, a_t]`` into ``d_model``, applies
    ``n_layer`` causal Transformer blocks, then projects to MDN parameters
    ``(pi, mu, sigma)`` for the next latent ``z_{t+1}``.

    Args:
        z_dim: Dimensionality of the latent vector.
        act_dim: Dimensionality of the action embedding.
        d_model: Transformer model dimension.
        act_space: Size of the action space (input to action embedder).
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads.
        max_len: Maximum sequence length for positional encoding.
        n_gaussians: Number of mixture components in the MDN head.
    """

    def __init__(
        self,
        z_dim: int,
        act_dim: int,
        d_model: int,
        act_space: int,
        n_layer: int,
        n_head: int,
        max_len: int,
        n_gaussians: int,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.act_space = act_space
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_len = max_len
        self.n_gaussians = n_gaussians

        self.act_embedder = nn.Linear(act_space, act_dim)
        self.in_proj = nn.Linear(z_dim + act_dim, d_model)

        self.pe = SinusoidalPositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, n_head, 4 * d_model, dropout=0.1)
                for _ in range(n_layer)
            ]
        )

        self.pi_head = nn.Linear(d_model, n_gaussians)
        self.mu_head = nn.Linear(d_model, n_gaussians * z_dim)
        self.logsigma_head = nn.Linear(d_model, n_gaussians * z_dim)

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        mode: str = "sample",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        """Predict next latent distribution or sample.

        Args:
            z: Latent sequence ``(B, S, z_dim)``.
            actions: Action sequence ``(B, S, act_space)``.
            mode: One of ``"sample"`` (returns sample from MDN),
                ``"mean"`` (returns mean of the most likely component),
                ``"all"`` (returns ``(pi, mu, sigma)``).

        Returns:
            If mode is ``"sample"`` or ``"mean"``: predicted ``z_next (B, S, z_dim)``.
            If mode is ``"all"``: ``(pi, mu, sigma)`` where
                - pi:    ``(B, S, n_gaussians)``
                - mu:    ``(B, S, n_gaussians, z_dim)``
                - sigma: ``(B, S, n_gaussians, z_dim)``
        """
        act_emb = self.act_embedder(actions)
        in_x = torch.cat([z, act_emb], dim=-1)
        x = self.in_proj(in_x)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x)

        B, S, _ = x.shape

        pi = F.softmax(self.pi_head(x), dim=-1)
        mu = self.mu_head(x).view(B, S, self.n_gaussians, self.z_dim)
        sigma = torch.exp(self.logsigma_head(x).view(B, S, self.n_gaussians, self.z_dim)) + 1e-6

        if mode == "all":
            return pi, mu, sigma

        if mode == "mean":
            best = pi.argmax(dim=-1, keepdim=True)  # (B, S, 1)
            z_next = mu.gather(
                2, best.unsqueeze(-1).expand(-1, -1, -1, self.z_dim)
            ).squeeze(2)
            return z_next

        # mode == "sample"
        return self.sample(pi, mu, sigma)

    def sample(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample ``z_{t+1}`` from the mixture of Gaussians.

        Args:
            pi: Mixture weights ``(B, S, n_gaussians)``.
            mu: Component means ``(B, S, n_gaussians, z_dim)``.
            sigma: Component stds ``(B, S, n_gaussians, z_dim)``.
            temperature: Softmax temperature for ``pi`` (1 = unchanged).

        Returns:
            Sampled latent ``(B, S, z_dim)``.
        """
        B, S, n_g, z_dim = mu.shape

        if temperature != 1.0:
            pi = (pi + 1e-10).log() / temperature
            pi = F.softmax(pi, dim=-1)

        component = torch.multinomial(
            pi.view(B * S, n_g), num_samples=1
        ).view(B, S, 1, 1)

        mu_selected = mu.gather(dim=2, index=component.expand(-1, -1, -1, z_dim))
        sigma_selected = sigma.gather(dim=2, index=component.expand(-1, -1, -1, z_dim))

        eps = torch.randn_like(mu_selected)
        return (mu_selected + sigma_selected * eps).squeeze(2)

    @staticmethod
    def mdn_loss(
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Mixture-of-Gaussians negative log-likelihood loss.

        Args:
            pi: Mixture weights ``(B, S, n_gaussians)``.
            mu: Component means ``(B, S, n_gaussians, z_dim)``.
            sigma: Component stds ``(B, S, n_gaussians, z_dim)``.
            target: Ground-truth next latent ``(B, S, z_dim)``.

        Returns:
            Scalar NLL averaged over all elements.
        """
        z_dim = target.size(-1)
        target = target.unsqueeze(-2)

        const = -0.5 * z_dim * math.log(2 * math.pi)
        log_prob = (
            const
            - sigma.log().sum(dim=-1)
            - 0.5 * ((target - mu) / sigma).pow(2).sum(dim=-1)
        )

        log_weighted = log_prob + (pi + 1e-10).log()
        nll = -torch.logsumexp(log_weighted, dim=-1)
        return nll.mean()

    # ------------------------------------------------------------------
    # Serialisation: config.json + model.safetensors
    # ------------------------------------------------------------------

    def _config_dict(self) -> dict:
        return {
            "z_dim": self.z_dim,
            "act_dim": self.act_dim,
            "d_model": self.d_model,
            "act_space": self.act_space,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "max_len": self.max_len,
            "n_gaussians": self.n_gaussians,
        }

    def save_pretrained(self, save_dir: str):
        """Save model weights (safetensors) and architecture config (JSON).

        Creates ``save_dir/config.json`` and ``save_dir/model.safetensors``.
        """
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._config_dict(), f, indent=2, ensure_ascii=False)

        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, weights_path)

    @classmethod
    def from_pretrained(cls, save_dir: str, map_location: str = "cpu") -> "PredictorTransformer":
        """Load a model from a previously saved ``save_pretrained`` directory.

        Reads ``config.json`` to reconstruct the architecture, then loads
        ``model.safetensors`` and restores the state dict.
        """
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model = cls(**config)
        state_dict = load_file(weights_path, device=str(map_location))
        model.load_state_dict(state_dict)
        return model
