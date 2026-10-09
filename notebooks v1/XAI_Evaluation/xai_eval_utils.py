"""XAI Quantitative Evaluation Utilities for Chest X-Ray Infiltration Detection.

Implements the three core objective metrics:
1. Pointing Game (Hit Rate %): Checks if argmax(Heatmap) is inside the ground truth BBox.
2. Energy Inside BBox (%): Measures the proportion of CAM saliency energy inside the BBox.
3. Intersection over Union (IoU) & Dice Coefficient: Measures spatial overlap at threshold tau.

Reference:
- Zhang et al. (2018). "Top-down Neural Attention by Excitation Backprop" (Pointing Game).
- Thesis Chapter 3 (Research Gap 6 & Objective XAI Verification).
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Dict, Any, List, Optional


def create_bbox_mask(df_bbox: pd.DataFrame, image_name: str, target_size: Tuple[int, int] = (224, 224), orig_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Create a binary mask (H, W) where 1 indicates radiologist ground-truth infiltration bounding box.
    Handles multiple bounding boxes per image.
    """
    mask = np.zeros(target_size, dtype=np.uint8)
    sub = df_bbox[df_bbox['Image Index'] == image_name]
    if sub.empty:
        return mask

    scale_x = target_size[1] / float(orig_size[1])
    scale_y = target_size[0] / float(orig_size[0])

    for _, row in sub.iterrows():
        bx = int(np.clip(row['Bbox [x'] * scale_x, 0, target_size[1] - 1))
        by = int(np.clip(row['y'] * scale_y, 0, target_size[0] - 1))
        bw = int(np.clip(row['w'] * scale_x, 1, target_size[1] - bx))
        bh = int(np.clip(row['h]'] * scale_y, 1, target_size[0] - by))
        mask[by:by + bh, bx:bx + bw] = 1

    return mask


def compute_pointing_game(cam: np.ndarray, mask: np.ndarray, tolerance: int = 5) -> Tuple[bool, Tuple[int, int], float]:
    """Compute Pointing Game Hit/Miss.
    
    Parameters:
    - cam: 2D numpy array [0, 1]
    - mask: 2D binary numpy array [0, 1]
    - tolerance: pixel radius margin for hits near boundaries
    
    Returns:
    - is_hit: bool
    - peak_coord: (y, x)
    - peak_val: float
    """
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    peak_val = float(cam[peak_y, peak_x])

    if mask[peak_y, peak_x] == 1:
        return True, (int(peak_y), int(peak_x)), peak_val

    # Check with tolerance radius if requested
    if tolerance > 0:
        y_min = max(0, peak_y - tolerance)
        y_max = min(cam.shape[0], peak_y + tolerance + 1)
        x_min = max(0, peak_x - tolerance)
        x_max = min(cam.shape[1], peak_x + tolerance + 1)
        if np.sum(mask[y_min:y_max, x_min:x_max]) > 0:
            return True, (int(peak_y), int(peak_x)), peak_val

    return False, (int(peak_y), int(peak_x)), peak_val


def compute_energy_inside_bbox(cam: np.ndarray, mask: np.ndarray) -> float:
    """Compute percentage of heatmap energy inside the Ground Truth BBox.
    Energy Inside (%) = sum(CAM * Mask) / sum(CAM) * 100
    """
    total_energy = float(np.sum(cam))
    if total_energy <= 1e-7:
        return 0.0

    inside_energy = float(np.sum(cam * mask))
    return (inside_energy / total_energy) * 100.0


def compute_iou_and_dice(cam: np.ndarray, mask: np.ndarray, threshold: float = 0.3) -> Tuple[float, float]:
    """Compute Intersection over Union (IoU) and Dice Coefficient at a specified saliency threshold."""
    bin_cam = (cam >= threshold).astype(np.uint8)
    bin_mask = (mask > 0).astype(np.uint8)

    intersection = float(np.sum(bin_cam & bin_mask))
    union = float(np.sum(bin_cam | bin_mask))

    iou = (intersection / union) if union > 0 else 0.0
    dice = (2.0 * intersection) / (float(np.sum(bin_cam)) + float(np.sum(bin_mask))) if (np.sum(bin_cam) + np.sum(bin_mask)) > 0 else 0.0

    return iou, dice


class GradCAMGenerator:
    """Fast ResNet50 Grad-CAM extractor."""
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        self.features = []
        self.gradients = []

        def forward_hook(m, inp, out):
            self.features.append(out)

        def backward_hook(m, grad_in, grad_out):
            self.gradients.append(grad_out[0])

        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate(self, u8_gray: np.ndarray) -> np.ndarray:
        self.features.clear()
        self.gradients.clear()

        rgb = cv2.cvtColor(u8_gray, cv2.COLOR_GRAY2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        out = self.model(tensor)
        target_class = torch.argmax(out[0]).item()
        loss = out[0, target_class]
        self.model.zero_grad()
        loss.backward()

        feat = self.features[0][0].detach().cpu().numpy()
        grad = self.gradients[0][0].detach().cpu().numpy()
        weights = np.mean(grad, axis=(1, 2))

        cam = np.zeros(feat.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feat[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val
        return cam
