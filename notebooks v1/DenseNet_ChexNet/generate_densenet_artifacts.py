"""Generate DenseNet121 vs ResNet50 benchmark artifacts."""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.models as models

from densenet_utils import DenseNetGradCAMGenerator

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

    # Load ResNet50 & DenseNet121
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT).to(device)

    resnet_cam = GradCAMGenerator(resnet, device)
    densenet_cam = DenseNetGradCAMGenerator(densenet, device)

    dae_ckpt = os.path.join(current_dir, "../DAE_CLAHE/checkpoints/dae_trained.pth")
    dae = DAE()
    if os.path.exists(dae_ckpt):
        dae.load_checkpoint(dae_ckpt, device=str(device))
    dae.to(device)
    dae.eval()

    df_manifest = pd.read_csv(manifest_path)
    df_bbox = pd.read_csv(bbox_path)
    infil_df = df_manifest[(df_manifest['Class'] == 'Infiltration') & (df_manifest['Has_BBox'] == True)].reset_index(drop=True)

    sample_indices = [0, 1]
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(20, 10), dpi=200)

    records = []
    for r_idx, s_idx in enumerate(sample_indices):
        row = infil_df.iloc[s_idx]
        img_name = row['Image Index']
        full_path = os.path.join(base_data_dir, row['Relative_Path'])
        raw_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        raw_256 = cv2.resize(raw_img, (256, 256))
        dae_256 = apply_dae_clahe(raw_256, dae, level=2, device=device)

        mask = create_bbox_mask(df_bbox, img_name, target_size=(224, 224), orig_size=raw_img.shape[::-1])
        sub_b = df_bbox[df_bbox['Image Index'] == img_name]

        # 4 Combinations
        r_raw_cam = resnet_cam.generate(raw_256)
        d_raw_cam = densenet_cam.generate(raw_256)
        r_dae_cam = resnet_cam.generate(dae_256)
        d_dae_cam = densenet_cam.generate(dae_256)

        cams = [
            ("(A) ResNet50 (Raw CXR)\n(Baseline Architecture)", raw_256, r_raw_cam),
            ("(B) DenseNet121 (Raw CXR)\n(ChexNet SOTA Architecture)", raw_256, d_raw_cam),
            ("(C) ResNet50 (DAE+CLAHE)\n(Denoised Opacity)", dae_256, r_dae_cam),
            ("(D) DenseNet121 (DAE+CLAHE)\n(Dense Multi-Scale Feature Reuse)", dae_256, d_dae_cam)
        ]

        for c_idx, (title, img_src, cam) in enumerate(cams):
            ax = axes[r_idx, c_idx]
            ax.imshow(cv2.resize(img_src, (224, 224)), cmap='gray')
            ax.imshow(cam, cmap='jet', alpha=0.45)

            for _, b in sub_b.iterrows():
                scale_x = 224.0 / raw_img.shape[1]
                scale_y = 224.0 / raw_img.shape[0]
                rect = patches.Rectangle((b['Bbox [x']*scale_x, b['y']*scale_y), b['w']*scale_x, b['h]']*scale_y,
                                         linewidth=2.2, edgecolor='lime', facecolor='none', linestyle='--')
                ax.add_patch(rect)

            hit, peak_coord, _ = compute_pointing_game(cam, mask, tolerance=5)
            energy = compute_energy_inside_bbox(cam, mask)
            iou, _ = compute_iou_and_dice(cam, mask, threshold=0.3)

            marker_color = 'cyan' if hit else 'red'
            hit_text = "HIT" if hit else "MISS"
            ax.scatter([peak_coord[1]], [peak_coord[0]], s=110, c=marker_color, marker='x', linewidths=2.8, zorder=5)

            status_col = 'darkgreen' if hit else 'darkred'
            ax.set_title(f"{title}\n[{hit_text}] Energy: {energy:.1f}% | IoU: {iou:.2f}", fontsize=10.5, fontweight='bold', color=status_col)
            ax.axis('off')

    plt.suptitle("Architecture Benchmark: DenseNet121 (ChexNet) vs ResNet50 on Infiltration XAI\n[Green Dashed Box = Doctor BBox | Crosshair 'X' = Model Peak Saliency]", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "densenet_vs_resnet_gradcam_comparison.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
