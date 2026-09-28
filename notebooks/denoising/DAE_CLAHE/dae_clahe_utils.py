"""DAE + CLAHE Utilities Module for Chest X-Ray Denoising.

Reference:
    Thamilarasi, Asaithambi & Roselin (2025).
    "Enhanced ensemble segmentation of lung chest X-ray images by denoising autoencoder and CLAHE."

Pipeline:
    Input CXR -> Denoising Autoencoder (DAE) -> CLAHE (Contrast Enhancement)

Strict Contract:
    All image outputs are 2D numpy arrays with dtype np.uint8 in [0, 255].
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


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

        # Decoder stages (with skip connections)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(c * 2 + c, c)

        # Final reconstruction head
        self.head = nn.Conv2d(c, in_channels, kernel_size=3, padding=1)

        if self.residual:
            nn.init.zeros_(self.head.weight)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        b = self.bottleneck(p3)

        d3 = self.up3(b)
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

        delta = self.head(d1)
        if self.residual:
            out = torch.clamp(x + delta, 0.0, 1.0)
        else:
            out = torch.sigmoid(delta)
        return out

    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0) -> None:
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
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])
            return ckpt.get("epoch", 0)
        else:
            self.load_state_dict(ckpt)
            return 0


# Predefined strength levels for CLAHE post-processing
LEVELS: Dict[int, Dict[str, Any]] = {
    1: {"clip_limit": 2.0, "name": "Level 1 (Mild)"},
    2: {"clip_limit": 4.0, "name": "Level 2 (Moderate - Recommended)"},
    3: {"clip_limit": 8.0, "name": "Level 3 (Aggressive)"},
}


def enforce_uint8(arr: np.ndarray) -> np.ndarray:
    """Ensure image is strictly uint8 2D array in [0, 255]."""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(2)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def add_synthetic_noise(
    img: np.ndarray,
    noise_type: str = "mixed",
    gaussian_sigma: float = 0.04,
    poisson_scale: float = 35.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """Inject realistic synthetic noise (Poisson quantum noise + Gaussian electronic noise)."""
    rng = np.random.default_rng(seed)
    norm = img.astype(np.float32) / 255.0

    if noise_type in ("gaussian", "mixed"):
        gauss = rng.normal(0.0, gaussian_sigma, norm.shape).astype(np.float32)
        norm = norm + gauss

    if noise_type in ("poisson", "mixed"):
        scaled = np.clip(norm, 0.0, 1.0) * poisson_scale
        noisy_p = rng.poisson(scaled).astype(np.float32) / poisson_scale
        norm = 0.5 * norm + 0.5 * noisy_p

    noisy_u8 = np.clip(np.round(norm * 255.0), 0, 255).astype(np.uint8)
    return noisy_u8


def denoise_dae_only(
    model: DAE,
    img: np.ndarray,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Run single 2D grayscale uint8 image through DAE neural network."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    img_u8 = enforce_uint8(img)
    h, w = img_u8.shape[:2]

    # Pad to multiple of 8 for 3-level pooling
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(img_u8, ((0, pad_h), (0, pad_w)), mode="reflect") if (pad_h > 0 or pad_w > 0) else img_u8
    tensor = torch.from_numpy(padded.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        out_tensor = model(tensor)

    out_np = out_tensor.squeeze().cpu().numpy()
    if pad_h > 0 or pad_w > 0:
        out_np = out_np[:h, :w]

    return enforce_uint8(out_np * 255.0)


def apply_dae_clahe(
    img: np.ndarray,
    model: DAE,
    level: int = 2,
    clip_limit: Optional[float] = None,
    tile_grid_size: Tuple[int, int] = (8, 8),
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Full Pipeline: DAE Denoising -> CLAHE Contrast Enhancement."""
    img_u8 = enforce_uint8(img)

    # Step 1: DAE Denoising
    dae_out = denoise_dae_only(model, img_u8, device=device)

    # Step 2: CLAHE Enhancement
    eff_clip = clip_limit if clip_limit is not None else LEVELS.get(level, {}).get("clip_limit", 4.0)
    clahe = cv2.createCLAHE(clipLimit=eff_clip, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(dae_out)

    return enforce_uint8(enhanced)


# ==========================================
# Image Quality Evaluation Metrics
# ==========================================

def compute_psnr(clean: np.ndarray, test_img: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
    c = clean.astype(np.float64)
    t = test_img.astype(np.float64)
    mse = np.mean((c - t) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two uint8 images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(x ** 2, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(y ** 2, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return float(np.mean(ssim_map))


def compute_cir(original: np.ndarray, enhanced: np.ndarray) -> float:
    """Contrast Improvement Ratio (CIR) = std(enhanced) / std(original)."""
    std_orig = float(np.std(original.astype(np.float64)))
    std_enh = float(np.std(enhanced.astype(np.float64)))
    return (std_enh / std_orig) if std_orig > 1e-4 else 1.0


def compute_edge_preservation_index(orig: np.ndarray, denoised: np.ndarray) -> float:
    """Edge Preservation Index (EPI): Sobel edge correlation."""
    sobel_orig_x = cv2.Sobel(orig, cv2.CV_64F, 1, 0, ksize=3)
    sobel_orig_y = cv2.Sobel(orig, cv2.CV_64F, 0, 1, ksize=3)
    grad_orig = np.sqrt(sobel_orig_x**2 + sobel_orig_y**2)

    sobel_den_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_den_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    grad_den = np.sqrt(sobel_den_x**2 + sobel_den_y**2)

    num = np.sum((grad_orig - np.mean(grad_orig)) * (grad_den - np.mean(grad_den)))
    denom = np.sqrt(np.sum((grad_orig - np.mean(grad_orig))**2) * np.sum((grad_den - np.mean(grad_den))**2))
    return float(num / denom) if denom > 1e-6 else 1.0
