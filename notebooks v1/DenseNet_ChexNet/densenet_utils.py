"""DenseNet121 (ChexNet) vs ResNet50 Evaluation Utilities for Chest X-Ray Infiltration.

Reference:
- Rajpurkar et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning."
- Thesis Proposal (Section 2.2.6: Deep Convolutional Architectures for Chest Radiographs).

Comparison Rationale:
- ResNet50 uses additive residual bypass: x_l = H(x_l-1) + x_l-1
- DenseNet121 uses iterative channel concatenation: x_l = H([x_0, x_1, ..., x_l-1])
  This multi-scale feature reuse enables DenseNet121 to preserve fine diffuse alveolar textures
  (Infiltration) across deep layers much better than ResNet50.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Optional


class DenseNetGradCAMGenerator:
    """Grad-CAM generator hooked to the final dense block of DenseNet121."""
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

        # Hook the last conv layer of the 4th dense block
        target_layer = self.model.features.denseblock4.denselayer16.conv2
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate(self, u8_gray: np.ndarray, target_class: Optional[int] = None) -> np.ndarray:
        self.features.clear()
        self.gradients.clear()

        rgb = cv2.cvtColor(u8_gray, cv2.COLOR_GRAY2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)

        out = self.model(tensor)
        if target_class is None:
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
