"""Lung Field Segmentation and Masking Utilities for Chest X-Rays.

Reference:
- Rahman et al. (2021). "Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images."
- Thesis Proposal (Section 2.2.4: Lung Boundary Isolation).

Purpose:
  Isolate the left and right lung lobes to eliminate external false-positive activations
  (collarbones, ribs, humeral heads, abdominal gas, and ECG wires) in Explainable AI.
"""

import cv2
import numpy as np
from typing import Tuple


def extract_lung_mask(img: np.ndarray, blur_ksize: int = 7) -> np.ndarray:
    """Extract anatomical binary lung mask (2D uint8, values 0 or 255) using adaptive thresholding
    and morphological connected component analysis.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    h, w = img.shape[:2]

    # Step 1: Contrast stretching & Gaussian Blur
    p2, p98 = np.percentile(img, (2, 98))
    stretched = np.clip((img - p2) / max(p98 - p2, 1e-5) * 255.0, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(stretched, (blur_ksize, blur_ksize), 0)

    # Step 2: Thresholding for dark lung parenchyma
    # Lung fields are radiolucent (darker than bones and soft tissues)
    thresh_val = int(np.mean(blurred) * 0.95)
    _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Remove outer image border artifacts
    border_margin = int(w * 0.05)
    binary[:border_margin, :] = 0
    binary[-border_margin:, :] = 0
    binary[:, :border_margin] = 0
    binary[:, -border_margin:] = 0

    # Step 3: Morphological closing to bridge blood vessels and small infiltrates
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Step 4: Extract two largest connected components (Left and Right lungs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    lung_mask = np.zeros((h, w), dtype=np.uint8)
    if num_labels > 1:
        # Exclude background (index 0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1] + 1  # 1-indexed

        # Take up to 2 largest components that resemble lung lobes
        selected = []
        for idx in sorted_indices[:4]:
            area = stats[idx, cv2.CC_STAT_AREA]
            cx, cy = centroids[idx]
            # Must occupy significant area (between 3% and 40% of the image)
            if (0.03 * h * w) <= area <= (0.45 * h * w):
                # Must be located in central/thoracic region
                if (0.15 * w) <= cx <= (0.85 * w) and (0.15 * h) <= cy <= (0.85 * h):
                    selected.append(idx)
                    if len(selected) == 2:
                        break

        for s_idx in selected:
            lung_mask[labels == s_idx] = 255

    # Step 5: Fill internal holes using contours and dilate slightly to avoid clipping margins
    contours, _ = cv2.findContours(lung_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) > (0.02 * h * w):
            # Convex hull for smooth anatomical lung boundary
            hull = cv2.convexHull(cnt)
            cv2.drawContours(final_mask, [hull], -1, 255, thickness=-1)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=1)

    # If heuristic failed, fallback to central thoracic ellipse
    if np.sum(final_mask) < (0.05 * h * w * 255):
        cv2.ellipse(final_mask, (int(w * 0.35), int(h * 0.5)), (int(w * 0.18), int(h * 0.32)), 0, 0, 360, 255, -1)
        cv2.ellipse(final_mask, (int(w * 0.65), int(h * 0.5)), (int(w * 0.18), int(h * 0.32)), 0, 0, 360, 255, -1)

    return final_mask


def apply_lung_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary lung mask to image: regions outside lung are set to 0 (black)."""
    mask_bin = (mask > 0).astype(img.dtype)
    return img * mask_bin
