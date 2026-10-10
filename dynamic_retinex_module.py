"""
Clean-room dynamic Retinex front-end for dynamic multi-exposure fusion (MEF).

Goal:
    Each exposure frame is decomposed independently with SHARED parameters:
        I_i -> R_i, L_i, G_i
    where
        R_i: exposure-robust reflectance-like representation for motion correspondence
        L_i: illumination map
        G_i: glare / over-exposure residual

This file is intentionally written from scratch instead of modifying the Retinex-MEF
repository source. It is designed to be easy to connect to a later motion module:
    (R_i, R_ref) -> (V_i, C_motion_i)

PyTorch only. No dependency on the Retinex-MEF repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, act: bool = True):
        super().__init__()
        p = k // 2
        layers = [
            nn.ReflectionPad2d(p),
            nn.Conv2d(in_ch, out_ch, k, padding=0),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvAct(ch, ch, 3, act=True)
        self.conv2 = ConvAct(ch, ch, 3, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class IlluminationEstimator(nn.Module):
    """Predict one-channel illumination L in [0, 1]."""

    def __init__(self, width: int = 48, depth: int = 4):
        super().__init__()
        layers: List[nn.Module] = [ConvAct(3, width)]
        layers += [ResidualBlock(width) for _ in range(depth)]
        layers += [ConvAct(width, 1, act=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(image))


class ReflectanceEstimator(nn.Module):
    """
    Per-frame reflectance estimator.

    Input is a SINGLE frame plus its illumination, rather than concatenating two
    exposure frames. This is the key structural change for dynamic scenes.
    """

    def __init__(self, width: int = 64, depth: int = 6):
        super().__init__()
        # RGB + illumination = 4 channels.
        self.in_proj = ConvAct(4, width)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.out_proj = ConvAct(width, 3, act=False)

    def forward(self, image: torch.Tensor, illum: torch.Tensor) -> torch.Tensor:
        feat = self.body(self.in_proj(torch.cat([image, illum], dim=1)))
        return torch.sigmoid(self.out_proj(feat))


class GlareEstimator(nn.Module):
    """
    Predict non-negative glare residual G from clean reflectance + illumination.

    We gate G with a soft over-exposure prior derived from illumination so the
    network is discouraged from explaining normal/dark pixels using glare.
    """

    def __init__(self, width: int = 32, depth: int = 3):
        super().__init__()
        self.in_proj = ConvAct(4, width)
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.out_proj = ConvAct(width, 3, act=False)

    def forward(self, reflectance: torch.Tensor, illum: torch.Tensor) -> torch.Tensor:
        feat = self.body(self.in_proj(torch.cat([reflectance, illum], dim=1)))
        raw = torch.sigmoid(self.out_proj(feat))
        # Soft gate; not a hard threshold, so gradients remain smooth.
        gate = torch.sigmoid((illum - 0.75) * 12.0)
        return raw * gate


@dataclass
class RetinexOutput:
    R: torch.Tensor
    L: torch.Tensor
    G: torch.Tensor
    reconstruction: torch.Tensor


class DynamicRetinexDecomposer(nn.Module):
    """
    Shared-parameter, per-frame dynamic Retinex decomposition.

    Formula used here:
        I_hat = clamp(L * (R + G), 0, 1)

    Important distinction:
        - network weights are shared across all exposures;
        - R outputs are NOT forced to be the same pixel-by-pixel for dynamic frames.
    """

    def __init__(
        self,
        illum_width: int = 48,
        refl_width: int = 64,
        glare_width: int = 32,
    ):
        super().__init__()
        self.illum_net = IlluminationEstimator(illum_width)
        self.refl_net = ReflectanceEstimator(refl_width)
        self.glare_net = GlareEstimator(glare_width)

    def forward(self, image: torch.Tensor) -> RetinexOutput:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected BCHW RGB tensor, got {tuple(image.shape)}")
        L = self.illum_net(image)
        R = self.refl_net(image, L)
        G = self.glare_net(R, L)
        reconstruction = torch.clamp(L * (R + G), 0.0, 1.0)
        return RetinexOutput(R=R, L=L, G=G, reconstruction=reconstruction)

    def forward_sequence(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, T, 3, H, W]
        Returns:
            dict with R/L/G/reconstruction stacked on T dimension.
        """
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected [B,T,3,H,W], got {tuple(images.shape)}")
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        out = self.forward(flat)
        return {
            "R": out.R.reshape(b, t, 3, h, w),
            "L": out.L.reshape(b, t, 1, h, w),
            "G": out.G.reshape(b, t, 3, h, w),
            "reconstruction": out.reconstruction.reshape(b, t, 3, h, w),
        }


def image_gradients(x: torch.Tensor):
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


def illumination_smoothness_loss(L: torch.Tensor, image: torch.Tensor, edge_scale: float = 10.0) -> torch.Tensor:
    """Edge-aware illumination smoothness: smooth L except around strong image edges."""
    gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
    Lx, Ly = image_gradients(L)
    Ix, Iy = image_gradients(gray)
    wx = torch.exp(-edge_scale * Ix.abs())
    wy = torch.exp(-edge_scale * Iy.abs())
    return (Lx.abs() * wx).mean() + (Ly.abs() * wy).mean()


def pairwise_reflectance_consistency(Rs: torch.Tensor) -> torch.Tensor:
    """
    Exposure-invariance loss for ALIGNED STATIC exposure sequences only.

    Rs: [B, T, 3, H, W]

    Do NOT use this loss directly on dynamic MEF frames before motion alignment.
    """
    if Rs.ndim != 5:
        raise ValueError("Rs must have shape [B,T,C,H,W]")
    t = Rs.shape[1]
    if t < 2:
        return Rs.new_zeros(())
    loss = Rs.new_zeros(())
    count = 0
    for i in range(t):
        for j in range(i + 1, t):
            loss = loss + F.smooth_l1_loss(Rs[:, i], Rs[:, j])
            count += 1
    return loss / count


def reflectance_gradient_consistency(Rs: torch.Tensor) -> torch.Tensor:
    """Encourage exposure-invariant structures without over-constraining absolute RGB values."""
    t = Rs.shape[1]
    if t < 2:
        return Rs.new_zeros(())
    loss = Rs.new_zeros(())
    count = 0
    for i in range(t):
        gxi, gyi = image_gradients(Rs[:, i])
        for j in range(i + 1, t):
            gxj, gyj = image_gradients(Rs[:, j])
            loss = loss + F.smooth_l1_loss(gxi, gxj) + F.smooth_l1_loss(gyi, gyj)
            count += 1
    return loss / count


def glare_regularization(G: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """
    Keep glare concentrated around bright/saturated regions.

    G, image: [N,3,H,W]
    """
    brightness = image.amax(dim=1, keepdim=True)
    bright_prior = torch.sigmoid((brightness - 0.85) * 16.0)
    outside_penalty = (G * (1.0 - bright_prior)).mean()
    sparsity = G.mean()
    return outside_penalty + 0.05 * sparsity


@dataclass
class LossWeights:
    reconstruction: float = 1.0
    illumination_init: float = 0.1
    illumination_smooth: float = 0.5
    reflectance_consistency: float = 0.5
    reflectance_grad_consistency: float = 0.2
    glare_regularization: float = 0.1


def static_pretrain_loss(
    model: DynamicRetinexDecomposer,
    images: torch.Tensor,
    weights: Optional[LossWeights] = None,
) -> Dict[str, torch.Tensor]:
    """
    Loss for STATIC, spatially aligned multi-exposure sequences (e.g. SICE patches).

    This is the recommended stage-1 training for the Retinex front-end. It learns an
    exposure-robust R without incorrectly forcing moving objects to align.

    images: [B,T,3,H,W] in [0,1]
    """
    weights = weights or LossWeights()
    out = model.forward_sequence(images)

    b, t, _, h, w = images.shape
    flat_img = images.reshape(b * t, 3, h, w)
    flat_rec = out["reconstruction"].reshape(b * t, 3, h, w)
    flat_L = out["L"].reshape(b * t, 1, h, w)
    flat_G = out["G"].reshape(b * t, 3, h, w)

    loss_recon = F.l1_loss(flat_rec, flat_img)
    target_L = flat_img.amax(dim=1, keepdim=True)
    loss_L_init = F.l1_loss(flat_L, target_L)
    loss_L_smooth = illumination_smoothness_loss(flat_L, flat_img)
    loss_R_cons = pairwise_reflectance_consistency(out["R"])
    loss_R_grad = reflectance_gradient_consistency(out["R"])
    loss_G = glare_regularization(flat_G, flat_img)

    total = (
        weights.reconstruction * loss_recon
        + weights.illumination_init * loss_L_init
        + weights.illumination_smooth * loss_L_smooth
        + weights.reflectance_consistency * loss_R_cons
        + weights.reflectance_grad_consistency * loss_R_grad
        + weights.glare_regularization * loss_G
    )

    return {
        "total": total,
        "reconstruction": loss_recon,
        "illumination_init": loss_L_init,
        "illumination_smooth": loss_L_smooth,
        "reflectance_consistency": loss_R_cons,
        "reflectance_grad_consistency": loss_R_grad,
        "glare_regularization": loss_G,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    net = DynamicRetinexDecomposer()
    x = torch.rand(2, 3, 3, 96, 96)
    out = net.forward_sequence(x)
    losses = static_pretrain_loss(net, x)
    print({k: tuple(v.shape) for k, v in out.items()})
    print({k: float(v.detach()) for k, v in losses.items()})
