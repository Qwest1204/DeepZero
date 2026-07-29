"""Perceptual loss (LPIPS) and PatchGAN discriminator for high-quality VAE training.

These losses help preserve fine details (NPCs, monsters, items) that standard
MSE + KL tends to smooth out.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


# ---------------------------------------------------------------------------
# LPIPS — Learned Perceptual Image Patch Similarity (simplified)
# ---------------------------------------------------------------------------

class LPIPS(nn.Module):
    """Perceptual loss using VGG16 feature maps (no learned linear layers).

    Computes L1 distance in normalized feature space across 4 VGG layers.
    This is equivalent to LPIPS with unit weights (the "lin" variant).

    Args:
        layernames: List of VGG layer names to extract features from.
    """

    def __init__(self, layernames: list[str] | None = None):
        super().__init__()
        if layernames is None:
            layernames = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False

        features = vgg.features
        self.layernames = layernames

        name_map = {
            "relu1_2": 3,
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 22,
            "relu5_3": 29,
        }
        self.slice_ids = [name_map[n] for n in layernames]

        self.slices = nn.ModuleList()
        start = 0
        for idx in self.slice_ids:
            self.slices.append(nn.Sequential(*list(features.children())[start: idx + 1]))
            start = idx + 1

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Perceptual distance ``(B,)`` averaged across layers."""
        input_norm = (input - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        dist = 0.0
        x, y = input_norm, target_norm
        for block in self.slices:
            x = block(x)
            y = block(y)
            # Normalise per-channel spatial mean
            x_n = x / (x.norm(p=2, dim=[2, 3], keepdim=True) + 1e-10)
            y_n = y / (y.norm(p=2, dim=[2, 3], keepdim=True) + 1e-10)
            dist = dist + (x_n - y_n).norm(p=1, dim=[1, 2, 3])

        return dist / len(self.slices)


# ---------------------------------------------------------------------------
# PatchGAN Discriminator (70×70 receptive field)
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator from the Pix2Pix paper.

    Classifies overlapping image patches as real or fake.

    Args:
        in_channels: Number of input image channels.
        ndf: Base feature dimension (doubled each layer).
    """

    def __init__(self, in_channels: int = 3, ndf: int = 64):
        super().__init__()
        layers = OrderedDict()
        layers["conv1"] = nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
        layers["lrelu1"] = nn.LeakyReLU(0.2, inplace=True)

        mult = 1
        for i in range(1, 3):
            out_mul = min(2**i, 8)
            layers[f"conv{i+1}"] = nn.Conv2d(
                ndf * mult, ndf * out_mul, kernel_size=4, stride=2, padding=1
            )
            layers[f"inorm{i+1}"] = nn.InstanceNorm2d(ndf * out_mul)
            layers[f"lrelu{i+1}"] = nn.LeakyReLU(0.2, inplace=True)
            mult = out_mul

        layers["conv4"] = nn.Conv2d(
            ndf * mult, ndf * mult, kernel_size=4, stride=1, padding=1
        )
        layers["inorm4"] = nn.InstanceNorm2d(ndf * mult)
        layers["lrelu4"] = nn.LeakyReLU(0.2, inplace=True)

        layers["conv_out"] = nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=1)

        self.net = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patch predictions ``(B, 1, H_p, W_p)``."""
        return self.net(x)


def discriminator_loss(
    pred_real: torch.Tensor,
    pred_fake: torch.Tensor,
    loss_type: str = "lsgan",
) -> tuple[torch.Tensor, torch.Tensor]:
    """LSGAN or hinge losses for the discriminator and generator.

    Args:
        pred_real: Discriminator output for real images ``(B, 1, H, W)``.
        pred_fake: Discriminator output for fake (reconstructed) images ``(B, 1, H, W)``.
        loss_type: One of ``"lsgan"`` (least squares) or ``"hinge"``.

    Returns:
        loss_d: Discriminator loss (real=1, fake=0).
        loss_g: Generator (adversarial) loss (fool D into predicting 1).
    """
    if loss_type == "lsgan":
        loss_fn = F.mse_loss
        real_target = torch.ones_like(pred_real)
        fake_target = torch.zeros_like(pred_fake)
    elif loss_type == "hinge":
        loss_d_real = F.relu(1.0 - pred_real).mean()
        loss_d_fake = F.relu(1.0 + pred_fake).mean()
        loss_d = loss_d_real + loss_d_fake
        loss_g = -pred_fake.mean()
        return loss_d, loss_g
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    loss_d_real = loss_fn(pred_real, real_target)
    loss_d_fake = loss_fn(pred_fake, fake_target)
    loss_d = 0.5 * (loss_d_real + loss_d_fake)
    loss_g = loss_fn(pred_fake, real_target)
    return loss_d, loss_g


# ---------------------------------------------------------------------------
# Combined VAE loss wrapper
# ---------------------------------------------------------------------------

class VAECombinedLoss(nn.Module):
    """Combines MSE, KL, LPIPS, and adversarial losses for VAE training.

    The discriminator must be trained separately — this module only computes
    losses for the VAE (generator) side.

    Args:
        lpips_weight: Weight for the perceptual loss term.
        adv_weight: Weight for the adversarial (fool discriminator) term.
        beta: KL-divergence weight (β-VAE).
    """

    def __init__(self, lpips_weight: float = 0.1, adv_weight: float = 0.01, beta: float = 0.1):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.adv_weight = adv_weight
        self.beta = beta
        self.lpips = LPIPS()
        self.lpips.eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        d_fake: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all VAE losses.

        Args:
            recon_x: Reconstructed image ``(B, C, H, W)``.
            x: Original image ``(B, C, H, W)``.
            mu: Latent mean ``(B, latent_dim)``.
            logvar: Latent log-variance ``(B, latent_dim)``.
            d_fake: Discriminator output for ``recon_x`` (if using adversarial loss).

        Returns:
            dict with keys: ``loss`` (total), ``mse``, ``kl``, ``lpips``, ``adv``.
        """
        mse = F.mse_loss(recon_x, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + self.beta * kl

        losses = {"mse": mse.detach(), "kl": kl.detach(), "lpips": torch.tensor(0.0), "adv": torch.tensor(0.0)}

        if self.lpips_weight > 0:
            lpips_val = self.lpips(recon_x, x).sum()
            losses["lpips"] = lpips_val.detach()
            loss = loss + self.lpips_weight * lpips_val

        if d_fake is not None and self.adv_weight > 0:
            adv_loss = (
                F.mse_loss(d_fake, torch.ones_like(d_fake)) * recon_x.size(0)
            )
            losses["adv"] = adv_loss.detach()
            loss = loss + self.adv_weight * adv_loss

        losses["loss"] = loss
        return losses
