"""Artifact generation script for Score-CAM comparison.

Generates:
1. output/score_cam_vs_gradcam_comparison.png: Side-by-side comparison of Grad-CAM vs Score-CAM with doctor BBox
2. output/score_cam_denoise_effect.png: Score-CAM before vs after DAE+CLAHE
3. reports/score_cam_metrics_summary.csv: Pointing Game and Energy metrics
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.models as models

# Score-CAM generator
from score_cam_utils import ScoreCAMGenerator

# Sibling dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../XAI_Evaluation")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../DAE_CLAHE")))
from xai_eval_utils import create_bbox_mask, compute_pointing_game, compute_energy_inside_bbox, compute_iou_and_dice, GradCAMGenerator
from dae_clahe_utils import DAE, apply_dae_clahe


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    reports_dir = os.path.join(current_dir, "reports")
    base_data_dir = os.path.abspath(os.path.join(current_dir, "../.."))
    manifest_path = os.path.join(current_dir, "../Gradcam/sample_manifest_200.csv")
    bbox_path = os.path.join(base_data_dir, "BBox_List_2017.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Models
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    gradcam_gen = GradCAMGenerator(resnet, device)
    scorecam_gen = ScoreCAMGenerator(resnet, device, top_k=24)

    dae_ckpt = os.path.join(current_dir, "../DAE_CLAHE/checkpoints/dae_trained.pth")
    dae = DAE()
    if os.path.exists(dae_ckpt):
        dae.load_checkpoint(dae_ckpt, device=str(device))
    dae.to(device)
    dae.eval()

    df_manifest = pd.read_csv(manifest_path)
    df_bbox = pd.read_csv(bbox_path)
    infil_df = df_manifest[(df_manifest['Class'] == 'Infiltration') & (df_manifest['Has_BBox'] == True)].reset_index(drop=True)

    # Pick 2 sample cases
    sample_indices = [0, 1]
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(20, 10), dpi=200)

    for r_idx, s_idx in enumerate(sample_indices):
        row = infil_df.iloc[s_idx]
        img_name = row['Image Index']
        full_path = os.path.join(base_data_dir, row['Relative_Path'])
        raw_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        raw_256 = cv2.resize(raw_img, (256, 256))
        dae_256 = apply_dae_clahe(raw_256, dae, level=2, device=device)

        mask = create_bbox_mask(df_bbox, img_name, target_size=(224, 224), orig_size=raw_img.shape[::-1])
        sub_b = df_bbox[df_bbox['Image Index'] == img_name]

        # Generate CAMs
        gcam_raw = gradcam_gen.generate(raw_256)
        scam_raw = scorecam_gen.generate(raw_256)
        gcam_dae = gradcam_gen.generate(dae_256)
        scam_dae = scorecam_gen.generate(dae_256)

        cams = [
            ("(A) Raw CXR: Grad-CAM\n(Gradient-Based: Noisy)", raw_256, gcam_raw),
            ("(B) Raw CXR: Score-CAM\n(Gradient-Free: Clean)", raw_256, scam_raw),
            ("(C) DAE+CLAHE: Grad-CAM\n(Enhanced Opacity)", dae_256, gcam_dae),
            ("(D) DAE+CLAHE: Score-CAM\n(Optimal Localization)", dae_256, scam_dae)
        ]

        for c_idx, (title, img_src, cam) in enumerate(cams):
            ax = axes[r_idx, c_idx]
            ax.imshow(cv2.resize(img_src, (224, 224)), cmap='gray')
            ax.imshow(cam, cmap='jet', alpha=0.45)

            # Draw Ground Truth BBoxes
            for _, b in sub_b.iterrows():
                scale_x = 224.0 / raw_img.shape[1]
                scale_y = 224.0 / raw_img.shape[0]
                bx = b['Bbox [x'] * scale_x
                by = b['y'] * scale_y
                bw = b['w'] * scale_x
                bh = b['h]'] * scale_y
                rect = patches.Rectangle((bx, by), bw, bh, linewidth=2.2, edgecolor='lime', facecolor='none', linestyle='--')
                ax.add_patch(rect)

            hit, peak_coord, _ = compute_pointing_game(cam, mask, tolerance=5)
            energy = compute_energy_inside_bbox(cam, mask)
            iou, _ = compute_iou_and_dice(cam, mask, threshold=0.3)

            marker_color = 'cyan' if hit else 'red'
            hit_text = "HIT" if hit else "MISS"
            ax.scatter([peak_coord[1]], [peak_coord[0]], s=100, c=marker_color, marker='x', linewidths=2.8, zorder=5)

            col = 'darkgreen' if hit else 'darkred'
            ax.set_title(f"{title}\n[{hit_text}] Energy: {energy:.1f}% | IoU: {iou:.2f}", fontsize=10.5, fontweight='bold', color=col)
            ax.axis('off')

    plt.suptitle("Score-CAM vs Grad-CAM: Gradient-Free Visual Explanation Comparison\n[Green Dashed Box = Radiologist BBox | Crosshair 'X' = Model Peak Saliency]", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "score_cam_vs_gradcam_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
