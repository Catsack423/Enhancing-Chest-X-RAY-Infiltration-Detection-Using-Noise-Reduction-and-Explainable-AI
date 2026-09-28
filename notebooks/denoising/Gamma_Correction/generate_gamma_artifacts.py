"""Generate Gamma Correction artifacts and visualizations."""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.models as models

from gamma_utils import apply_gamma_correction, compute_contrast_metrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../xai/XAI_Evaluation")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../DAE_CLAHE")))
from xai_eval_utils import create_bbox_mask, compute_pointing_game, compute_energy_inside_bbox, compute_iou_and_dice, GradCAMGenerator
from dae_clahe_utils import DAE, apply_dae_clahe


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    reports_dir = os.path.join(current_dir, "reports")
    base_data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/versions/3"))
    manifest_path = os.path.join(current_dir, "../../xai/Gradcam/sample_manifest_200.csv")
    bbox_path = os.path.join(base_data_dir, "BBox_List_2017.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    cam_gen = GradCAMGenerator(resnet, device)

    dae_ckpt = os.path.join(current_dir, "../DAE_CLAHE/checkpoints/dae_trained.pth")
    dae = DAE()
    if os.path.exists(dae_ckpt):
        dae.load_checkpoint(dae_ckpt, device=str(device))
    dae.to(device)
    dae.eval()

    df_manifest = pd.read_csv(manifest_path)
    df_bbox = pd.read_csv(bbox_path)
    infil_df = df_manifest[(df_manifest['Class'] == 'Infiltration') & (df_manifest['Has_BBox'] == True)].reset_index(drop=True)

    row = infil_df.iloc[0]
    img_name = row['Image Index']
    raw_img = cv2.imread(os.path.join(base_data_dir, row['Relative_Path']), cv2.IMREAD_GRAYSCALE)
    raw_256 = cv2.resize(raw_img, (256, 256))

    mask = create_bbox_mask(df_bbox, img_name, target_size=(224, 224), orig_size=raw_img.shape[::-1])
    sub_b = df_bbox[df_bbox['Image Index'] == img_name]

    # 1. Plot Gamma Presets Grid
    gammas = [0.5, 0.8, 1.0, 1.2, 1.5]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=200)

    summary_records = []
    for idx, g in enumerate(gammas):
        g_img = apply_gamma_correction(raw_256, gamma=g)
        m = compute_contrast_metrics(g_img)
        m['Gamma'] = g
        summary_records.append(m)

        ax = axes[idx]
        ax.imshow(g_img, cmap='gray')
        note = " (Raw)" if g == 1.0 else (" (Brighten)" if g < 1.0 else " (Darken)")
        ax.set_title(f"gamma = {g}{note}\nContrast (Std): {m['Contrast_Std']} | Entropy: {m['Entropy']}", fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle("Gamma Correction Non-Linear Transformations (Rahman et al., 2021)", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    grid_path = os.path.join(output_dir, "gamma_levels_comparison.png")
    plt.savefig(grid_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {grid_path}")

    # 2. Benchmark: Gamma vs CLAHE vs DAE+CLAHE on Grad-CAM
    enhanced_dict = [
        ("Raw CXR (gamma=1.0)", raw_256),
        ("Gamma = 0.8 (Optimal)", apply_gamma_correction(raw_256, gamma=0.8)),
        ("CLAHE (clip=4.0)", cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(raw_256)),
        ("DAE + CLAHE (Level 2)", apply_dae_clahe(raw_256, dae, level=2, device=device))
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=200)
    for idx, (title, proc) in enumerate(enhanced_dict):
        ax = axes[idx]
        cam = cam_gen.generate(proc)
        ax.imshow(cv2.resize(proc, (224, 224)), cmap='gray')
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

        marker_col = 'cyan' if hit else 'red'
        hit_txt = "HIT" if hit else "MISS"
        ax.scatter([peak_coord[1]], [peak_coord[0]], s=110, c=marker_col, marker='x', linewidths=2.8, zorder=5)

        stat_col = 'darkgreen' if hit else 'darkred'
        ax.set_title(f"{title}\n[{hit_txt}] Energy: {energy:.1f}% | IoU: {iou:.2f}", fontsize=10.5, fontweight='bold', color=stat_col)
        ax.axis('off')

    plt.suptitle("Enhancement Benchmark: Gamma vs CLAHE vs DAE+CLAHE\n[Green Dashed Box = Doctor BBox | Crosshair 'X' = Peak Activation Point]", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    bench_path = os.path.join(output_dir, "gamma_vs_clahe_benchmark.png")
    plt.savefig(bench_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {bench_path}")

    # Save metrics table
    df_metrics = pd.DataFrame(summary_records)
    csv_path = os.path.join(reports_dir, "gamma_metrics_summary.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
