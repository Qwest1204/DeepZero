"""Predictor Transformer with MDN head for dynamics over the VAE latent pair.

The model consumes the VAE posterior pair ``(mu, logvar)`` of the current latent
sequence and predicts the mixture-of-Gaussians parameters of the next latent
``(pi, mu_next, logvar_next)`` plus an optional ``normal`` reward head
``(reward_mean, reward_logvar)``. The latent is either spatial
``(B, S, C, H, W)`` or flat ``(B, S, z_dim)``; both are flattened into a single
token per timestep and restored to ``latent_shape`` on the output.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from functools import reduce
from operator import mul
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from embedder.attention import MultiHeadAttention


def _prod(shape) -> int:
    return reduce(mul, shape, 1)


@dataclass
class PredictorConfig:
    """Hyper-parameters of the predictor transformer.

    ``latent_shape`` is either a spatial layout ``(C, H, W)`` or a flat size
    ``(z_dim,)``; ``z_dim = prod(latent_shape)`` and must not be set manually.

    Args:
        latent_shape: Spatial or flat latent shape of the VAE.
        act_space: Dimensionality of the action input (one-hot for discrete
            Doom actions, raw vector for continuous Car/MW actions).
        d_model: Transformer model dimension.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads.
        n_gaussians: Number of mixture components in the MDN head.
        max_len: Sequence length (windows are truncated to this).
        act_dim: Dimension of the action embedding.
        dropout: Dropout probability.
        predict_reward: Add a ``normal`` reward head.
        reward_mode: Distribution of the reward head (``"normal"`` only).
        reward_bins: Reserved for a future categorical reward head.
        rotary_theta: Reserved for a future RoPE position embedding.
    """

    latent_shape: tuple
    act_space: int = 7
    d_model: int = 1024
    n_layer: int = 6
    n_head: int = 8
    n_gaussians: int = 4
    max_len: int = 32
    act_dim: int = 256
    dropout: float = 0.1
    predict_reward: bool = True
    reward_mode: str = "normal"
    reward_bins: int = 256
    rotary_theta: float = 10000.0

    def __post_init__(self):
        if not isinstance(self.latent_shape, (tuple, list)) or len(self.latent_shape) not in (1, 3):
            raise ValueError(
                f"latent_shape должен быть (z_dim,) или (C, H, W), got {self.latent_shape!r}"
            )
        self.latent_shape = tuple(int(v) for v in self.latent_shape)
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) должен делиться на n_head ({self.n_head})")
        for name in ("act_space", "d_model", "n_layer", "n_head", "n_gaussians",
                     "max_len", "act_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть положительным, got {getattr(self, name)}")
        if self.reward_mode not in ("normal", "categorical"):
            raise ValueError(
                f"reward_mode должен быть 'normal' или 'categorical', got {self.reward_mode!r}"
            )

    @property
    def z_dim(self) -> int:
        """Flattened latent dimension, derived from ``latent_shape``."""
        return _prod(self.latent_shape)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["latent_shape"] = list(self.latent_shape)
        return d

    @classmethod
    def from_dict(cls, cfg: dict) -> "PredictorConfig":
        known = {f.name for f in fields(cls) if f.init}
        filtered = {k: v for k, v in cfg.items() if k in known}
        return cls(**filtered)


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
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with causal self-attention.

    Norm → MHA (causal) → residual → Norm → FFN → residual.
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
        xn = self.norm1(x)
        x = x + self.dropout1(self.mha(xn, xn, xn, is_causal=True))

        ff = self.norm2(x)
        x = x + self.dropout2(self.ffn(ff))
        return x


class PredictorTransformer(nn.Module):
    """Transformer dynamics predictor over the VAE latent pair.

    Concatenates the flattened ``mean`` and ``logvar`` of each latent token with
    the action embedding, runs ``n_layer`` causal blocks, then issues:
      * an MDN head ``(pi, mu, logvar)`` for the next latent, and
      * a ``normal`` head ``(mean, logvar)`` for the next reward.

    Args:
        config: A :class:`PredictorConfig` instance.
        device: Optional device to place parameters on.
    """

    def __init__(self, config: PredictorConfig, device: Union[str, torch.device, None] = None):
        super().__init__()
        self.config = config
        c = config
        z_dim = c.z_dim

        self.act_embedder = nn.Linear(c.act_space, c.act_dim)
        self.in_proj = nn.Linear(2 * z_dim + c.act_dim, c.d_model)

        self.pe = SinusoidalPositionalEncoding(c.d_model, c.max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(c.d_model, c.n_head, 4 * c.d_model, dropout=c.dropout)
                for _ in range(c.n_layer)
            ]
        )

        self.pi_head = nn.Linear(c.d_model, c.n_gaussians)
        self.mu_head = nn.Linear(c.d_model, c.n_gaussians * z_dim)
        self.logvar_head = nn.Linear(c.d_model, c.n_gaussians * z_dim)

        self.reward_head = nn.Linear(c.d_model, 2)  # (mean, log_var)

        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        actions: torch.Tensor,
        mode: str = "all",
    ) -> Union[torch.Tensor, tuple]:
        """Predict the next latent distribution (and optionally reward).

        Args:
            mu: VAE mean of the latent sequence ``(B, S, C, H, W)`` or ``(B, S, z_dim)``.
            logvar: VAE log-var of the latent sequence, same shape as ``mu``.
            actions: Action sequence ``(B, S, act_space)`` (one-hot or raw).
            mode: ``"all"``, ``"mean"`` or ``"sample"``.

        Returns (mode):
            - ``"all"``: ``(pi, mu_next, logvar_next, reward)`` with
              ``pi (B, S, G)``,
              ``mu_next/logvar_next (B, S, G, *latent_shape)`` and
              ``reward (B, S, 2)`` (or ``None`` if ``predict_reward=False``).
            - ``"mean"``: ``(mu_best, logvar_best)`` ``(B, S, *latent_shape)``.
            - ``"sample"``: a sampled ``z_next (B, S, *latent_shape)``.
        """
        cfg = self.config
        mu_f, logvar_f = self._flatten(mu), self._flatten(logvar)
        B, S = mu_f.shape[0], mu_f.shape[1]

        act_emb = self.act_embedder(actions)
        in_x = torch.cat([mu_f, logvar_f, act_emb], dim=-1)
        x = self.in_proj(in_x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)

        pi = F.softmax(self.pi_head(x), dim=-1)  # (B, S, G)
        mu_n = self.mu_head(x).view(B, S, cfg.n_gaussians, cfg.z_dim)
        logvar_n = self.logvar_head(x).view(B, S, cfg.n_gaussians, cfg.z_dim)

        if mode == "all":
            shape = (B, S, cfg.n_gaussians, *cfg.latent_shape)
            out_mu = mu_n.reshape(shape)
            out_logvar = logvar_n.reshape(shape)
            reward = None
            if cfg.predict_reward:
                reward = self.reward_head(x)  # (B, S, 2)
            return pi, out_mu, out_logvar, reward

        best = pi.argmax(dim=-1, keepdim=True)  # (B, S, 1)
        idx = best.unsqueeze(-1).expand(-1, -1, -1, cfg.z_dim)
        mu_best = mu_n.gather(2, idx).squeeze(2)
        logvar_best = logvar_n.gather(2, idx).squeeze(2)

        out_shape = (B, S, *cfg.latent_shape)
        if mode == "mean":
            return mu_best.reshape(out_shape), logvar_best.reshape(out_shape)

        if mode == "sample":
            z = mu_best + torch.randn_like(mu_best) * torch.exp(0.5 * logvar_best)
            return z.reshape(out_shape)

        raise ValueError(f"Неизвестный режим {mode!r}")

    def _flatten(self, q: torch.Tensor) -> torch.Tensor:
        """Flatten a latent sequence to ``(B, S, z_dim)``."""
        if q.ndim == 3:
            return q
        if q.ndim == 5:
            return q.reshape(q.shape[0], q.shape[1], self.config.z_dim)
        raise ValueError(f"Ожидался (B,S,C,H,W) или (B,S,z_dim), got {tuple(q.shape)}")

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    @staticmethod
    def mdn_loss(
        pi: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target_mu: torch.Tensor,
        target_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Negative expected log-likelihood of an MDN vs the VAE posterior pair.

        The target is a noisy observation of the latent state:
        ``p(z_tgt) = N(mu_tgt, var_tgt)``.  We maximise
        ``E_{z_tgt}[log Σ_g π_g N(z_tgt | mu_g, var_g)]``,
        which evaluates to:

        ``log_prob_g = -½ Σ_d [ log var_{g,d} + (var_{tgt,d} + Δ²_{g,d}) / var_{g,d} ]``.

        This formulation forces the predictor to produce well-calibrated
        variances (the optimal ``var_g`` equals ``var_tgt + (mu_g - mu_tgt)²``),
        preventing the variance collapse that the simpler convolution loss allows.
        """
        B, S = pi.shape[0], pi.shape[1]

        def flat_comp(t):
            return t.reshape(t.shape[0], t.shape[1], t.shape[2], -1) if t.ndim >= 4 else t

        def flat_target(t):
            return t.reshape(t.shape[0], t.shape[1], -1) if t.ndim >= 4 else t

        mu = flat_comp(mu)
        logvar = flat_comp(logvar)
        target_mu = flat_target(target_mu)
        target_logvar = flat_target(target_logvar)

        z_dim = target_mu.size(-1)

        var_pred = torch.exp(logvar).clamp(min=1e-8)
        var_tgt = torch.exp(target_logvar.unsqueeze(-2))

        diff2 = (target_mu.unsqueeze(-2) - mu).pow(2)

        log_prob = -0.5 * z_dim * math.log(2 * math.pi)
        log_prob = log_prob - 0.5 * (logvar + (var_tgt + diff2) / var_pred).sum(-1)

        log_weighted = log_prob + (pi + 1e-10).log()
        nll = -torch.logsumexp(log_weighted, dim=-1)
        return nll.mean()

    @staticmethod
    def reward_loss(
        reward_mean: torch.Tensor,
        reward_logvar: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> torch.Tensor:
        """Mean Gaussian NLL for the reward head.

        Args:
            reward_mean: ``(B, S)`` or ``(B, S, 1)``.
            reward_logvar: ``(B, S)`` or ``(B, S, 1)``.
            target_reward: ``(B, S)`` reward of the next step.
        """
        mean = reward_mean.squeeze(-1)
        logvar = reward_logvar.squeeze(-1)
        tgt = target_reward
        var = torch.exp(logvar) + 1e-6
        return 0.5 * (logvar + (tgt - mean).pow(2) / var).mean()

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------
    # Serialisation: config.json + model.safetensors (same as the VAE)
    # ------------------------------------------------------------------
    def save_pretrained(self, save_dir: str):
        """Save weights (safetensors) and architecture config (JSON).

        Creates ``save_dir/config.json`` and ``save_dir/model.safetensors``.
        """
        os.makedirs(save_dir, exist_ok=True)
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)

        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, weights_path)

    @classmethod
    def from_pretrained(cls, save_dir: str, map_location: str = "cpu") -> "PredictorTransformer":
        """Load a model from a previously saved ``save_pretrained`` directory."""
        config_path = os.path.join(save_dir, "config.json")
        weights_path = os.path.join(save_dir, "model.safetensors")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)

        config = PredictorConfig.from_dict(cfg_dict)
        model = cls(config)
        state_dict = load_file(weights_path, device=str(map_location))
        model.load_state_dict(state_dict)
        return model