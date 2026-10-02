import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Dataset root: {DATASET_DIR}")
print(f"Output folder: {OUTPUT_DIR}")

# Load manifest and BBox data
manifest_path = BASE_DIR / 'sample_manifest_200.csv'
df_manifest = pd.read_csv(manifest_path)

bbox_path = DATASET_DIR / 'BBox_List_2017.csv'
df_bbox = pd.read_csv(bbox_path).iloc[:, :6]
df_bbox.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']

# Model Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.eval()

# Grad-CAM Hooks
gradients = []
activations = []
def hook_fwd(m, inp, out):
    activations.append(out)
def hook_bwd(m, g_inp, g_out):
    gradients.append(g_out[0])

target_layer = model.layer4[-1]
target_layer.register_forward_hook(hook_fwd)
target_layer.register_full_backward_hook(hook_bwd)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def compute_gradcam(img_pil):
    gradients.clear()
    activations.clear()
    input_tensor = preprocess(img_pil.convert('RGB')).unsqueeze(0).to(device)
    output = model(input_tensor)
    pred_idx = output.argmax(dim=1).item()
    score = output[0, pred_idx]
    model.zero_grad()
    score.backward()
    
    grad = gradients[0][0].detach()
    act = activations[0][0].detach()
    weights = torch.mean(grad, dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * act, dim=0).clamp(min=0).cpu().numpy()
    
    w, h = img_pil.size
    cam_resized = cv2.resize(cam, (w, h))
    if cam_resized.max() > cam_resized.min():
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
    else:
        cam_norm = np.zeros_like(cam_resized)
        
    heatmap = np.uint8(255 * cam_norm)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    cxr_rgb = np.array(img_pil.convert('RGB'))
    overlay = np.uint8(0.45 * heatmap_colored + 0.55 * cxr_rgb)
    return cam_norm, overlay

# 1. Generate Single Pair Comparison
normal_row = df_manifest[df_manifest['Class'] == 'Normal'].iloc[0]
infil_row = df_manifest[df_manifest['Class'] == 'Infiltration'].iloc[0]

norm_img = Image.open(DATASET_DIR / normal_row['Relative_Path'])
norm_cam, norm_overlay = compute_gradcam(norm_img)

infil_img = Image.open(DATASET_DIR / infil_row['Relative_Path'])
infil_cam, infil_overlay = compute_gradcam(infil_img)
infil_boxes = df_bbox[df_bbox['Image Index'] == infil_row['Image Index']]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Normal
axes[0, 0].imshow(norm_img, cmap='gray')
axes[0, 0].set_title(f"[NORMAL] Raw CXR\n{normal_row['Image Index']}", fontsize=11, weight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(norm_cam, cmap='jet')
axes[0, 1].set_title("[NORMAL] Grad-CAM Heatmap", fontsize=11, weight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(norm_overlay)
axes[0, 2].set_title("[NORMAL] Overlay (CXR + Grad-CAM)", fontsize=11, weight='bold')
axes[0, 2].axis('off')

# Row 2: Infiltration
axes[1, 0].imshow(infil_img, cmap='gray')
axes[1, 0].set_title(f"[INFILTRATION] Raw CXR\n{infil_row['Image Index']}", fontsize=11, weight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(infil_cam, cmap='jet')
axes[1, 1].set_title("[INFILTRATION] Grad-CAM Heatmap", fontsize=11, weight='bold')
axes[1, 1].axis('off')

axes[1, 2].imshow(infil_overlay)
for _, b in infil_boxes.iterrows():
    bx, by, bw, bh = float(b['x']), float(b['y']), float(b['w']), float(b['h'])
    rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor='#00FF66', facecolor='none', linestyle='--')
    axes[1, 2].add_patch(rect)
    axes[1, 2].text(
        bx, max(0, by - 10), "BBox: Infiltrate",
        color='black', fontsize=9, weight='bold',
        bbox=dict(boxstyle='square,pad=0.2', facecolor='#00FF66', edgecolor='none', alpha=0.9)
    )
axes[1, 2].set_title("[INFILTRATION] Overlay + Doctor's BBox (Green)", fontsize=11, weight='bold')
axes[1, 2].axis('off')

plt.tight_layout()
pair_out = OUTPUT_DIR / "comparison_single_pair.png"
plt.savefig(pair_out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {pair_out.name}")

# 2. Generate 4 Normal vs 4 Infiltration Multi-grid
fig, axes = plt.subplots(4, 4, figsize=(18, 17))
norm_samples = df_manifest[df_manifest['Class'] == 'Normal'].iloc[:4]
infil_samples = df_manifest[df_manifest['Class'] == 'Infiltration'].iloc[:4]

for i in range(4):
    # Normal Raw CXR
    n_row = norm_samples.iloc[i]
    n_img = Image.open(DATASET_DIR / n_row['Relative_Path'])
    n_cam, n_over = compute_gradcam(n_img)
    axes[i, 0].imshow(n_img, cmap='gray')
    axes[i, 0].set_title(f"Normal #{i+1}: {n_row['Image Index']}", fontsize=10)
    axes[i, 0].axis('off')
    
    # Normal Overlay
    axes[i, 1].imshow(n_over)
    axes[i, 1].set_title(f"Normal #{i+1} Grad-CAM Overlay", fontsize=10)
    axes[i, 1].axis('off')
    
    # Infiltration Raw CXR
    inf_row = infil_samples.iloc[i]
    inf_img = Image.open(DATASET_DIR / inf_row['Relative_Path'])
    inf_cam, inf_over = compute_gradcam(inf_img)
    axes[i, 2].imshow(inf_img, cmap='gray')
    axes[i, 2].set_title(f"Infiltration #{i+1}: {inf_row['Image Index']}", fontsize=10)
    axes[i, 2].axis('off')
    
    # Infiltration Overlay + BBox
    axes[i, 3].imshow(inf_over)
    boxes = df_bbox[df_bbox['Image Index'] == inf_row['Image Index']]
    for _, b in boxes.iterrows():
        bx, by, bw, bh = float(b['x']), float(b['y']), float(b['w']), float(b['h'])
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor='#00FF66', facecolor='none')
        axes[i, 3].add_patch(rect)
    axes[i, 3].set_title(f"Infiltration #{i+1} + BBox", fontsize=10)
    axes[i, 3].axis('off')

plt.tight_layout()
batch_out = OUTPUT_DIR / "batch_comparison_4x4.png"
plt.savefig(batch_out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {batch_out.name}")

print("Preview generation finished!")
