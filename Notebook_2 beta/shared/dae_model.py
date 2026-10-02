# Reused from scripts/denoise/dae_model.py; kept here for a standalone Colab folder.
"""Denoising Autoencoder (DAE) Architecture.

Reference: Thamilarasi et al. (2025)
"Enhanced ensemble segmentation of lung chest X-ray images by denoising autoencoder and CLAHE."

Architecture:
  - Convolutional Encoder with multi-scale feature extraction (Stage 1 -> Stage 2 -> Stage 3).
  - Bottleneck representation.
  - Convolutional Decoder with U-Net style skip connections to preserve fine anatomical
    structures (ribs, vascular markings, subtle diffuse infiltrations).
  - Residual head with zero initialization: enables exact identity passthrough prior to
    training, and accelerates residual noise prediction during training.
"""
from typing import Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Two successive Conv2d-BatchNorm-LeakyReLU layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DAE(nn.Module):
    """Convolutional Denoising Autoencoder with skip connections and residual learning."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, residual: bool = True):
        super().__init__()
        self.residual = residual
        c = base_channels

        # Encoder stages
        self.enc1 = ConvBlock(in_channels, c)       # -> (B, c, H, W)
        self.enc2 = ConvBlock(c, c * 2)             # -> (B, 2c, H/2, W/2)
        self.enc3 = ConvBlock(c * 2, c * 4)         # -> (B, 4c, H/4, W/4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 4, c * 8)   # -> (B, 8c, H/8, W/8)

        # Decoder stages (with skip connection concatenation)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(c * 2 + c, c)

        # Final projection
        self.head = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

        # Initialize weights: zero-initialize residual head so untrained model acts as identity
        if self.residual:
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W) in range [0, 1].

        Returns
        -------
        torch.Tensor
            Denoised tensor of shape (B, 1, H, W) in range [0, 1].
        """
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder
        d3 = self.up3(b)
        # Pad if dimension mismatch due to odd dimensions
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Head
        delta = self.head(d1)

        if self.residual:
            out = torch.clamp(x + delta, 0.0, 1.0)
        else:
            out = torch.sigmoid(delta)

        return out

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "residual": self.residual,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, device: str = "cpu") -> int:
        """Load model checkpoint and return epoch."""
        ckpt = torch.load(path, map_location=device)
        if "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])
            return ckpt.get("epoch", 0)
        else:
            self.load_state_dict(ckpt)
            return 0


def denoise_image(
    model: DAE,
    img: np.ndarray,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Run single 2D grayscale uint8 image through DAE.

    Parameters
    ----------
    model : DAE
        Instantiated DAE model.
    img : np.ndarray
        Grayscale image (H, W), dtype uint8.
    device : torch.device, optional
        Torch device (defaults to CPU or GPU if available).

    Returns
    -------
    np.ndarray
        Denoised uint8 grayscale image (H, W).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Convert uint8 [0, 255] -> float32 [0.0, 1.0] tensor
    h, w = img.shape[:2]
    # Pad to multiple of 8 for symmetric 3-level pooling
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect") if (pad_h > 0 or pad_w > 0) else img
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = model(tensor)

    out_np = out_tensor.squeeze().cpu().numpy()
    if pad_h > 0 or pad_w > 0:
        out_np = out_np[:h, :w]

    out_uint8 = np.clip(np.round(out_np * 255.0), 0, 255).astype(np.uint8)
    return out_uint8
