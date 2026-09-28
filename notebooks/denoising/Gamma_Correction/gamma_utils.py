"""Gamma Correction Enhancement Utilities for Chest X-Ray Infiltration.

Reference:
- Rahman et al. (2021). "Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images."
  (Reported 96.29% accuracy with Gamma Correction).

Formula:
  I_out = 255 * (I_in / 255.0) ** gamma

Values:
  - gamma < 1.0 (e.g. 0.5, 0.8): Non-linear expansion of low-intensity dark regions (brightens lung parenchyma, highlights faint alveolar infiltrates).
  - gamma = 1.0: Baseline raw identity pass.
  - gamma > 1.0 (e.g. 1.2, 1.5): Compresses low intensities, darkens midtones (sharpens dense consolidations, increases peripheral contrast).
"""

import cv2
import numpy as np
from typing import Dict, Any


GAMMA_PRESETS: Dict[str, float] = {
    "Bright_High": 0.5,
    "Bright_Mild": 0.8,
    "Baseline": 1.0,
    "Dark_Mild": 1.2,
    "Dark_High": 1.5,
}


def apply_gamma_correction(img: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Apply fast gamma correction using a 256-element precomputed Look-Up Table (LUT).
    Guarantees strict 2D uint8 output in [0, 255].
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)

    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    # Build 256 LUT table
    inv_gamma = float(gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)

    return cv2.LUT(img, table)


def compute_contrast_metrics(img: np.ndarray) -> Dict[str, float]:
    """Compute standard deviation (contrast), Shannon entropy, and dynamic range."""
    std = float(np.std(img.astype(np.float64)))
    min_v, max_v = int(np.min(img)), int(np.max(img))
    
    # Shannon Entropy
    hist, _ = np.histogram(img, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -float(np.sum(hist * np.log2(hist)))

    return {
        "Mean_Intensity": round(float(np.mean(img)), 2),
        "Contrast_Std": round(std, 2),
        "Entropy": round(entropy, 3),
        "Dynamic_Range": int(max_v - min_v)
    }
