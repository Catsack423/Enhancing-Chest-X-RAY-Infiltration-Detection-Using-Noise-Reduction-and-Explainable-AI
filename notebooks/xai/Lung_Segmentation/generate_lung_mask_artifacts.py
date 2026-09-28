"""Generate Lung Segmentation and Masking artifacts."""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.models as models

from lung_segmentation_utils import extract_lung_mask, apply_lung_mask

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../XAI_Evaluation")))
from xai_eval_utils import create_bbox_mask, compute_pointing_game, compute_energy_inside_bbox, compute_iou_and_dice, GradCAMGenerator


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    reports_dir = os.path.join(current_dir, "reports")
    base_data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/versions/3"))
    manifest_path = os.path.join(current_dir, "../Gradcam/sample_manifest_200.csv")
    bbox_path = os.path.join(base_data_dir, "BBox_List_2017.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    cam_gen = GradCAMGenerator(resnet, device)

    df_manifest = pd.read_csv(manifest_path)
    df_bbox = pd.read_csv(bbox_path)
    infil_df = df_manifest[(df_manifest['Class'] == 'Infiltration') & (df_manifest['Has_BBox'] == True)].reset_index(drop=True)

    row = infil_df.iloc[0]
    img_name = row['Image Index']
    raw_img = cv2.imread(os.path.join(base_data_dir, row['Relative_Path']), cv2.IMREAD_GRAYSCALE)
    raw_256 = cv2.resize(raw_img, (256, 256))

    # Lung mask
    lung_mask_256 = extract_lung_mask(raw_256)
    masked_256 = apply_lung_mask(raw_256, lung_mask_256)

    # Ground truth BBox mask at 224x224
    gt_mask = create_bbox_mask(df_bbox, img_name, target_size=(224, 224), orig_size=raw_img.shape[::-1])
    sub_b = df_bbox[df_bbox['Image Index'] == img_name]

    # 1. Pipeline Stages Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)

    axes[0].imshow(raw_256, cmap='gray')
    axes[0].set_title("(A) Raw Full Chest X-Ray\n(Includes Clavicles, Ribs, Stomach)", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(lung_mask_256, cmap='Blues_r')
    axes[1].set_title("(B) Extracted Anatomical Lung Mask\n(Isolated Thoracic Cavities)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(masked_256, cmap='gray')
    axes[2].set_title("(C) Segmented Lung Parenchyma\n(Non-Pulmonary Regions Suppressed)", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Anatomical Lung Field Segmentation Pipeline (Rahman et al., 2021)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    stages_path = os.path.join(output_dir, "lung_segmentation_stages.png")
    plt.savefig(stages_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {stages_path}")

    # 2. Grad-CAM Comparison: Unmasked vs Masked
    cam_unmasked = cam_gen.generate(raw_256)
    cam_masked = cam_gen.generate(masked_256)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=200)

    # Unmasked
    ax1 = axes[0]
    ax1.imshow(cv2.resize(raw_256, (224, 224)), cmap='gray')
    ax1.imshow(cam_unmasked, cmap='jet', alpha=0.45)
    for _, b in sub_b.iterrows():
        scale_x = 224.0 / raw_img.shape[1]
        scale_y = 224.0 / raw_img.shape[0]
        rect = patches.Rectangle((b['Bbox [x']*scale_x, b['y']*scale_y), b['w']*scale_x, b['h]']*scale_y,
                                 linewidth=2.2, edgecolor='lime', facecolor='none', linestyle='--')
        ax1.add_patch(rect)
    hit1, peak1, _ = compute_pointing_game(cam_unmasked, gt_mask, tolerance=5)
    energy1 = compute_energy_inside_bbox(cam_unmasked, gt_mask)
    iou1, _ = compute_iou_and_dice(cam_unmasked, gt_mask, threshold=0.3)
    ax1.scatter([peak1[1]], [peak1[0]], s=110, c='red' if not hit1 else 'cyan', marker='x', linewidths=2.8, zorder=5)
    ax1.set_title(f"(A) Unmasked Raw CXR: Grad-CAM\n[{'HIT' if hit1 else 'MISS'}] Energy: {energy1:.1f}% | IoU: {iou1:.2f}\n(Attention wanders to collarbone/rib margins)", fontsize=10.5, fontweight='bold', color='darkred')
    ax1.axis('off')

    # Masked
    ax2 = axes[1]
    ax2.imshow(cv2.resize(masked_256, (224, 224)), cmap='gray')
    ax2.imshow(cam_masked, cmap='jet', alpha=0.45)
    for _, b in sub_b.iterrows():
        scale_x = 224.0 / raw_img.shape[1]
        scale_y = 224.0 / raw_img.shape[0]
        rect = patches.Rectangle((b['Bbox [x']*scale_x, b['y']*scale_y), b['w']*scale_x, b['h]']*scale_y,
                                 linewidth=2.2, edgecolor='lime', facecolor='none', linestyle='--')
        ax2.add_patch(rect)
    hit2, peak2, _ = compute_pointing_game(cam_masked, gt_mask, tolerance=5)
    energy2 = compute_energy_inside_bbox(cam_masked, gt_mask)
    iou2, _ = compute_iou_and_dice(cam_masked, gt_mask, threshold=0.3)
    ax2.scatter([peak2[1]], [peak2[0]], s=110, c='cyan' if hit2 else 'red', marker='x', linewidths=2.8, zorder=5)
    ax2.set_title(f"(B) Lung Masked CXR: Grad-CAM\n[{'HIT' if hit2 else 'MISS'}] Energy: {energy2:.1f}% | IoU: {iou2:.2f}\n(Forced 100% pulmonary focus inside BBox)", fontsize=10.5, fontweight='bold', color='darkgreen')
    ax2.axis('off')

    plt.suptitle("Impact of Lung Field Masking on Explainable AI (Grad-CAM)\n[Green Dashed Box = Doctor BBox | Crosshair 'X' = Model Peak Saliency]", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    comp_path = os.path.join(output_dir, "gradcam_before_vs_after_masking.png")
    plt.savefig(comp_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {comp_path}")


if __name__ == "__main__":
    main()
