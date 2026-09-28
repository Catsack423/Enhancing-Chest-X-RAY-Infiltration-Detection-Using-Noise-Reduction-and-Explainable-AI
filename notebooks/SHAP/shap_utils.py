"""SHAP Utilities for Chest X-ray Infiltration Detection.
Implements GradientExplainer (Aumann-Shapley / Expected Gradients) on ResNet50.
Computes positive attributions (evidence for Infiltration - Red)
and negative attributions (evidence against Infiltration / for Normal - Blue).
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import shap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResNet50BinaryWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        
    def forward(self, x):
        # We output 2 logits: class 0 (Normal) and class 1 (Infiltration proxy)
        logits = self.base(x)
        # Combine into 2 class outputs
        score_norm = logits[:, :500].mean(dim=1, keepdim=True)
        score_infil = logits[:, 500:].mean(dim=1, keepdim=True)
        return torch.cat([score_norm, score_infil], dim=1)

def get_shap_explainer(model):
    wrapped = ResNet50BinaryWrapper(model).to(device)
    wrapped.eval()
    
    # 2 baseline reference images (black and mean background)
    bg = torch.zeros(2, 3, 224, 224).to(device)
    explainer = shap.GradientExplainer(wrapped, bg)
    return wrapped, explainer

def compute_shap_map(img_input, explainer, target_class=1):
    """Computes pixel-level Shapley values for target_class (1 = Infiltration).
    Returns:
        pos_map: Positive Shapley attribution (evidence FOR Infiltration) [0, 1]
        neg_map: Negative Shapley attribution (evidence AGAINST Infiltration) [0, 1]
        overlay: Blended RGB image with red (pos) and blue (neg) attributions
    """
    if isinstance(img_input, np.ndarray):
        img_pil = Image.fromarray(img_input).convert('RGB')
    else:
        img_pil = img_input.convert('RGB')
        
    inp_t = preprocess(img_pil).unsqueeze(0).to(device)
    
    # Compute Shapley values
    shap_vals, indexes = explainer.shap_values(inp_t, ranked_outputs=2)
    # Target class 1 attribution
    idx_target = 0 if indexes[0][0] == target_class else 1
    sv = shap_vals[idx_target][0]  # shape: (3, 224, 224)
    
    # Aggregate across color channels
    sv_sum = np.sum(sv, axis=0)  # shape: (224, 224)
    w, h = img_pil.size
    sv_resized = cv2.resize(sv_sum, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Separate Positive and Negative Shapley values
    pos_map = np.maximum(sv_resized, 0)
    neg_map = np.maximum(-sv_resized, 0)
    
    if pos_map.max() > 0:
        pos_map /= pos_map.max()
    if neg_map.max() > 0:
        neg_map /= neg_map.max()
        
    # Create Red-Blue overlay
    cxr_gray = np.array(img_pil.convert('L'))
    overlay = cv2.cvtColor(cxr_gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    
    # Red for Positive (Infiltration support)
    overlay[:, :, 0] += pos_map * 140.0
    # Blue for Negative (Normal support)
    overlay[:, :, 2] += neg_map * 140.0
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return pos_map, neg_map, overlay
