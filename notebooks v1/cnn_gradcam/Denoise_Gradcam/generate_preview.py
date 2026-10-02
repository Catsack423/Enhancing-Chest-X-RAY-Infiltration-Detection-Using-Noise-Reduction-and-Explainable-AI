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
import torchvision.models as models
import torchvision.transforms as transforms

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.append(str(BASE_DIR))
import denoise_methods as dm

# Load Manifest & BBox
manifest_path = DATASET_DIR / 'notebooks' / 'Gradcam' / 'sample_manifest_200.csv'
df_manifest = pd.read_csv(manifest_path)

bbox_path = DATASET_DIR / 'BBox_List_2017.csv'
df_bbox = pd.read_csv(bbox_path).iloc[:, :6]
df_bbox.columns = ['Image Index', 'Finding Label', 'x', 'y', 'w', 'h']

# Model Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model.eval()

gradients, activations = [], []
def hook_fwd(m, inp, out): activations.append(out)
def hook_bwd(m, g_inp, g_out): gradients.append(g_out[0])

model.layer4[-1].register_forward_hook(hook_fwd)
model.layer4[-1].register_full_backward_hook(hook_bwd)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def compute_gradcam(img_u8):
    gradients.clear()
    activations.clear()
    img_pil = Image.fromarray(img_u8).convert('RGB')
    t_in = preprocess(img_pil).unsqueeze(0).to(device)
    out = model(t_in)
    pred_idx = out.argmax(dim=1).item()
    score = out[0, pred_idx]
    model.zero_grad()
    score.backward(retain_graph=True)
    
    grad = gradients[0][0].detach()
    act = activations[0][0].detach()
    weights = torch.mean(grad, dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * act, dim=0).clamp(min=0).cpu().numpy()
    
    w, h = img_u8.shape[1], img_u8.shape[0]
    cam_resized = cv2.resize(cam, (w, h))
    if cam_resized.max() > cam_resized.min():
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min())
    else:
        cam_norm = np.zeros_like(cam_resized)
        
    heatmap = np.uint8(255 * cam_norm)
    heatmap_colored = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    cxr_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    overlay = np.uint8(0.45 * heatmap_colored + 0.55 * cxr_rgb)
    return cam_norm, overlay

# Pick sample Infiltration
sample_row = df_manifest[df_manifest['Class'] == 'Infiltration'].iloc[0]
img_name = sample_row['Image Index']
img_path = DATASET_DIR / sample_row['Relative_Path']
raw_u8 = np.array(Image.open(img_path).convert('L'))
boxes = df_bbox[df_bbox['Image Index'] == img_name]

print(f"Processing sample image: {img_name}")

# 1. Plot Denoised Images
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for idx, (name, fn) in enumerate(dm.ALL_METHODS.items()):
    processed = fn(raw_u8)
    axes[idx].imshow(processed, cmap='gray')
    axes[idx].set_title(name, fontsize=11, weight='bold', pad=8)
    axes[idx].axis('off')
axes[7].axis('off')
plt.suptitle(f"7 Denoising Methods Comparison | {img_name}", fontsize=14, weight='bold')
plt.tight_layout()
p1 = OUTPUT_DIR / "denoise_methods_visual_comparison.png"
plt.savefig(p1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p1.name}")

# 2. Plot Denoise + Grad-CAM + Doctor's BBox
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
for idx, (name, fn) in enumerate(dm.ALL_METHODS.items()):
    processed = fn(raw_u8)
    cam, overlay = compute_gradcam(processed)
    axes[idx].imshow(overlay)
    for _, b in boxes.iterrows():
        bx, by, bw, bh = float(b['x']), float(b['y']), float(b['w']), float(b['h'])
        rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.5, edgecolor='#00FF66', facecolor='none', linestyle='--')
        axes[idx].add_patch(rect)
        axes[idx].text(
            bx, max(0, by - 8), "BBox: Infiltrate",
            color='black', fontsize=9, weight='bold',
            bbox=dict(boxstyle='square,pad=0.2', facecolor='#00FF66', edgecolor='none', alpha=0.9)
        )
    axes[idx].set_title(name, fontsize=11, weight='bold', pad=8)
    axes[idx].axis('off')
axes[7].axis('off')
plt.suptitle(f"Denoise + Grad-CAM Comparison | {img_name}\n(Green Dashed Box = Doctor's Ground Truth BBox)", fontsize=13, weight='bold')
plt.tight_layout()
p2 = OUTPUT_DIR / "denoise_gradcam_7methods_comparison.png"
plt.savefig(p2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {p2.name}")

print("All previews generated successfully!")
