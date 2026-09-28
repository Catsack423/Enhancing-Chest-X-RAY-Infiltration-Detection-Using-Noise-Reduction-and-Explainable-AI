"""Score-CAM (Score-Weighted Visual Explanations) Implementation.

Reference:
- Wang et al. (2020). "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks."
- Rahman et al. (2021). "Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images."

Gradient-Free visual explanation:
Overcomes gradient saturation and noisy gradients in standard Grad-CAM by directly probing
forward classification score increases when input regions are masked by activation maps.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Optional


class ScoreCAMGenerator:
    """Gradient-Free Score-CAM Extractor for CNN models."""
    def __init__(self, model: torch.nn.Module, device: torch.device, top_k: int = 32):
        self.model = model
        self.device = device
        self.top_k = top_k
        self.model.eval()

        self.features = []
        def forward_hook(m, inp, out):
            self.features.append(out)

        # Hook last convolutional layer (layer4 in ResNet50)
        target_layer = self.model.layer4[-1]
        target_layer.register_forward_hook(forward_hook)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate(self, u8_gray: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
        self.features.clear()
        rgb = cv2.cvtColor(u8_gray, cv2.COLOR_GRAY2RGB)
        input_tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(input_tensor)
            if target_class is None:
                target_class = torch.argmax(out[0]).item()

        # Feature maps shape: (1, C, H, W), e.g. (1, 2048, 7, 7)
        feat = self.features[0]
        c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]

        # Select top K channels by spatial variance/energy to ensure real-time CPU speed
        variances = torch.var(feat.squeeze(0), dim=(1, 2))
        top_k = min(self.top_k, c)
        top_indices = torch.topk(variances, k=top_k).indices

        scores = []
        masks = []

        with torch.no_grad():
            for idx in top_indices:
                act = feat[0, idx:idx+1, :, :]  # (1, 1, 7, 7)
                # Upsample to input resolution (224, 224)
                upsampled = F.interpolate(act.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
                
                # Normalize activation mask to [0, 1]
                min_v = upsampled.min()
                max_v = upsampled.max()
                if max_v > min_v:
                    norm_mask = (upsampled - min_v) / (max_v - min_v)
                else:
                    norm_mask = torch.zeros_like(upsampled)

                masks.append(norm_mask)

                # Mask original input tensor
                masked_input = input_tensor * norm_mask
                masked_out = self.model(masked_input)
                score = masked_out[0, target_class].item()
                scores.append(score)

        # Softmax weighting across the K forward scores
        weights = F.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).numpy()

        # Weighted combination
        cam = np.zeros((224, 224), dtype=np.float32)
        for i, w_val in enumerate(weights):
            mask_np = masks[i].squeeze().cpu().numpy()
            cam += w_val * mask_np

        # ReLU and normalization
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = cam / max_val

        return cam
