"""Execution Script for Quantitative XAI Evaluation on 100 Infiltration Chest X-Rays.

Evaluates 4 Preprocessing Conditions:
1. Baseline (Raw CXR)
2. Median Filter (Level 2: 5x5)
3. CLAHE + DWT (Level 2: db1 wavelet)
4. DAE + CLAHE (Level 2: Thamilarasi et al., 2025)

Computes:
- Pointing Game Accuracy (Strict & Margin)
- Heatmap Energy inside Radiologist Ground-Truth Bounding Box (%)
- Intersection over Union (IoU) @ tau=0.3 and tau=0.5
- Dice Coefficient @ tau=0.3

Outputs:
- reports/xai_quantitative_metrics_detailed.csv
- reports/xai_quantitative_summary.csv
- output/xai_metrics_comparison_barchart.png
- output/xai_energy_boxplot.png
- output/xai_visual_pointing_game_examples.png
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.models as models

# Local imports
from xai_eval_utils import (
    create_bbox_mask, compute_pointing_game, compute_energy_inside_bbox,
    compute_iou_and_dice, GradCAMGenerator
)

# Sibling notebook imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../denoising/Median_CLAHE_DWT")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../denoising/DAE_CLAHE")))
from denoise_methods import apply_median, apply_clahe_dwt
from dae_clahe_utils import DAE, apply_dae_clahe


def run_benchmark():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/versions/3"))
    manifest_path = os.path.join(current_dir, "../Gradcam/sample_manifest_200.csv")
    bbox_path = os.path.join(base_data_dir, "BBox_List_2017.csv")

    output_dir = os.path.join(current_dir, "output")
    reports_dir = os.path.join(current_dir, "reports")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=== Starting Quantitative XAI Benchmark on Infiltration CXRs ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load ResNet50 Grad-CAM
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
    cam_gen = GradCAMGenerator(resnet, device)

    # Load DAE
    dae_ckpt = os.path.join(current_dir, "../../denoising/DAE_CLAHE/checkpoints/dae_trained.pth")
    dae = DAE()
    if os.path.exists(dae_ckpt):
        dae.load_checkpoint(dae_ckpt, device=str(device))
        print(f"Loaded trained DAE checkpoint from {dae_ckpt}")
    else:
        print("Warning: Trained DAE checkpoint not found, running with default DAE initialization.")
    dae.to(device)
    dae.eval()

    # Load Manifest & BBoxes
    df_manifest = pd.read_csv(manifest_path)
    df_bbox = pd.read_csv(bbox_path)
    infil_df = df_manifest[(df_manifest['Class'] == 'Infiltration') & (df_manifest['Has_BBox'] == True)].reset_index(drop=True)
    print(f"Total verified Infiltration test images: {len(infil_df)}")

    methods = ["Baseline_Raw", "Median_L2", "CLAHE_DWT_L2", "DAE_CLAHE_L2"]
    detailed_records = []

    t_start = time.time()
    for idx, row in infil_df.iterrows():
        img_name = row['Image Index']
        rel_path = row['Relative_Path']
        full_path = os.path.join(base_data_dir, rel_path)

        if not os.path.exists(full_path):
            continue

        raw_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            continue

        h_orig, w_orig = raw_img.shape
        raw_256 = cv2.resize(raw_img, (256, 256))

        # Ground truth mask at 224x224
        mask = create_bbox_mask(df_bbox, img_name, target_size=(224, 224), orig_size=(w_orig, h_orig))

        # Preprocess with 4 methods
        proc_images = {
            "Baseline_Raw": raw_256,
            "Median_L2": apply_median(raw_256, level=2),
            "CLAHE_DWT_L2": apply_clahe_dwt(raw_256, level=2),
            "DAE_CLAHE_L2": apply_dae_clahe(raw_256, dae, level=2, device=device)
        }

        row_record = {
            "Image_Index": img_name,
            "Patient_ID": row['Patient ID'],
            "BBox_Area_Pixels": int(np.sum(mask)),
            "BBox_Area_Percent": float(np.sum(mask)) / (224.0 * 224.0) * 100.0
        }

        for m_name, im in proc_images.items():
            cam = cam_gen.generate(im)

            # Metrics
            hit_strict, peak_coord, peak_val = compute_pointing_game(cam, mask, tolerance=0)
            hit_margin, _, _ = compute_pointing_game(cam, mask, tolerance=5)
            energy_in = compute_energy_inside_bbox(cam, mask)
            iou_30, dice_30 = compute_iou_and_dice(cam, mask, threshold=0.3)
            iou_50, dice_50 = compute_iou_and_dice(cam, mask, threshold=0.5)

            row_record[f"{m_name}_Hit_Strict"] = int(hit_strict)
            row_record[f"{m_name}_Hit_Margin"] = int(hit_margin)
            row_record[f"{m_name}_Energy_Inside_Pct"] = round(energy_in, 2)
            row_record[f"{m_name}_IoU_03"] = round(iou_30, 4)
            row_record[f"{m_name}_IoU_05"] = round(iou_50, 4)
            row_record[f"{m_name}_Dice_03"] = round(dice_30, 4)
            row_record[f"{m_name}_Peak_Coord_Y"] = peak_coord[0]
            row_record[f"{m_name}_Peak_Coord_X"] = peak_coord[1]

        detailed_records.append(row_record)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(infil_df):
            elapsed = time.time() - t_start
            print(f"Processed [{idx + 1}/{len(infil_df)}] images ({elapsed:.1f}s)")

    df_detailed = pd.DataFrame(detailed_records)
    detailed_csv = os.path.join(reports_dir, "xai_quantitative_metrics_detailed.csv")
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"Saved detailed metrics: {detailed_csv}")

    # Compute Summary Table
    summary_data = []
    for m in methods:
        hit_rate_strict = df_detailed[f"{m}_Hit_Strict"].mean() * 100.0
        hit_rate_margin = df_detailed[f"{m}_Hit_Margin"].mean() * 100.0
        mean_energy = df_detailed[f"{m}_Energy_Inside_Pct"].mean()
        std_energy = df_detailed[f"{m}_Energy_Inside_Pct"].std()
        mean_iou_30 = df_detailed[f"{m}_IoU_03"].mean()
        std_iou_30 = df_detailed[f"{m}_IoU_03"].std()
        mean_iou_50 = df_detailed[f"{m}_IoU_05"].mean()
        mean_dice_30 = df_detailed[f"{m}_Dice_03"].mean()

        summary_data.append({
            "Method": m.replace("_", " "),
            "Pointing_Game_HitRate_Strict_%": round(hit_rate_strict, 1),
            "Pointing_Game_HitRate_Margin_%": round(hit_rate_margin, 1),
            "Energy_Inside_BBox_Mean_%": round(mean_energy, 2),
            "Energy_Inside_BBox_Std_%": round(std_energy, 2),
            "IoU_at_0.3_Mean": round(mean_iou_30, 4),
            "IoU_at_0.3_Std": round(std_iou_30, 4),
            "IoU_at_0.5_Mean": round(mean_iou_50, 4),
            "Dice_at_0.3_Mean": round(mean_dice_30, 4),
        })

    df_summary = pd.DataFrame(summary_data)
    summary_csv = os.path.join(reports_dir, "xai_quantitative_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary metrics: {summary_csv}")

    print("\n" + "="*70)
    print("=== FINAL QUANTITATIVE XAI BENCHMARK RESULTS (N=100) ===")
    print("="*70)
    print(df_summary.to_string(index=False))
    print("="*70 + "\n")

    # ========================================================
    # Plot 1: Comparison Bar Chart (Hit Rate, Energy, IoU)
    # ========================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=200)
    colors = ['#5c6bc0', '#ef5350', '#ffa726', '#26a69a']
    labels = ['Baseline (Raw)', 'Median Filter', 'CLAHE + DWT', 'DAE + CLAHE (Ours)']

    # 1. Pointing Game Hit Rate
    hit_vals = df_summary['Pointing_Game_HitRate_Margin_%'].tolist()
    bars1 = axes[0].bar(labels, hit_vals, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    axes[0].set_title("(A) Pointing Game Hit Rate (%)\n[Peak Activation Inside Doctor BBox]", fontsize=11, fontweight='bold', pad=10)
    axes[0].set_ylabel("Hit Rate (%)", fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars1:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2., h + 1.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    # 2. Energy Inside BBox
    energy_vals = df_summary['Energy_Inside_BBox_Mean_%'].tolist()
    energy_err = df_summary['Energy_Inside_BBox_Std_%'].tolist()
    bars2 = axes[1].bar(labels, energy_vals, yerr=energy_err, capsize=5, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    axes[1].set_title("(B) Saliency Energy Inside BBox (%)\n[Proportion of Attention on Pathology]", fontsize=11, fontweight='bold', pad=10)
    axes[1].set_ylabel("Mean Energy Inside (%)", fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, max(energy_vals) + max(energy_err) + 10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars2:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2., h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    # 3. IoU @ 0.3
    iou_vals = df_summary['IoU_at_0.3_Mean'].tolist()
    bars3 = axes[2].bar(labels, iou_vals, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    axes[2].set_title("(C) Spatial Overlap (IoU @ tau=0.3)\n[Intersection over Union with Radiologist BBox]", fontsize=11, fontweight='bold', pad=10)
    axes[2].set_ylabel("Mean IoU", fontsize=10, fontweight='bold')
    axes[2].set_ylim(0, max(iou_vals) * 1.35)
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars3:
        h = b.get_height()
        axes[2].text(b.get_x() + b.get_width()/2., h + 0.01, f"{h:.3f}", ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    for ax in axes:
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9.5, fontweight='bold')

    plt.suptitle("Quantitative Explainable AI (XAI) Evaluation Across Denoising Conditions (N=100 Infiltration CXRs)", fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    barchart_path = os.path.join(output_dir, "xai_metrics_comparison_barchart.png")
    plt.savefig(barchart_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {barchart_path}")

    # ========================================================
    # Plot 2: Boxplot of Energy Distribution
    # ========================================================
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    box_data = [
        df_detailed["Baseline_Raw_Energy_Inside_Pct"],
        df_detailed["Median_L2_Energy_Inside_Pct"],
        df_detailed["CLAHE_DWT_L2_Energy_Inside_Pct"],
        df_detailed["DAE_CLAHE_L2_Energy_Inside_Pct"]
    ]
    bp = ax.boxplot(box_data, patch_artist=True, labels=labels, notch=True,
                    medianprops=dict(color='black', linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title("Distribution of Heatmap Energy Inside Ground-Truth Bounding Box (N=100)", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Energy Inside BBox (%)", fontsize=11, fontweight='bold')
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, "xai_energy_boxplot.png")
    plt.savefig(boxplot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {boxplot_path}")

    # ========================================================
    # Plot 3: Visual Demonstration Cases with Pointing Game Crosshair
    # ========================================================
    sample_indices = [0, 1]
    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(18, 9), dpi=200)

    for r_idx, s_idx in enumerate(sample_indices):
        s_row = infil_df.iloc[s_idx]
        img_name = s_row['Image Index']
        full_path = os.path.join(base_data_dir, s_row['Relative_Path'])
        raw_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        raw_256 = cv2.resize(raw_img, (256, 256))
        mask = create_bbox_mask(df_bbox, img_name, target_size=(224, 224), orig_size=raw_img.shape[::-1])

        # Preprocess
        proc_imgs = [
            ("Raw CXR", raw_256, "Baseline_Raw"),
            ("Median L2", apply_median(raw_256, level=2), "Median_L2"),
            ("CLAHE + DWT L2", apply_clahe_dwt(raw_256, level=2), "CLAHE_DWT_L2"),
            ("DAE + CLAHE L2", apply_dae_clahe(raw_256, dae, level=2, device=device), "DAE_CLAHE_L2")
        ]

        sub_b = df_bbox[df_bbox['Image Index'] == img_name]

        for c_idx, (m_title, p_img, m_col) in enumerate(proc_imgs):
            ax = axes[r_idx, c_idx]
            cam = cam_gen.generate(p_img)
            ax.imshow(cv2.resize(p_img, (224, 224)), cmap='gray')
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

            # Peak point crosshair
            py = df_detailed.loc[df_detailed['Image_Index'] == img_name, f"{m_col}_Peak_Coord_Y"].values[0]
            px = df_detailed.loc[df_detailed['Image_Index'] == img_name, f"{m_col}_Peak_Coord_X"].values[0]
            hit = df_detailed.loc[df_detailed['Image_Index'] == img_name, f"{m_col}_Hit_Margin"].values[0]
            energy = df_detailed.loc[df_detailed['Image_Index'] == img_name, f"{m_col}_Energy_Inside_Pct"].values[0]
            iou = df_detailed.loc[df_detailed['Image_Index'] == img_name, f"{m_col}_IoU_03"].values[0]

            marker_color = 'cyan' if hit == 1 else 'red'
            hit_text = "HIT" if hit == 1 else "MISS"
            ax.scatter([px], [py], s=90, c=marker_color, marker='x', linewidths=2.5, zorder=5)

            status_color = 'navy' if hit == 1 else 'darkred'
            ax.set_title(f"{m_title}\n[{hit_text}] Energy: {energy:.1f}% | IoU: {iou:.2f}", fontsize=10, fontweight='bold', color=status_color)
            ax.axis('off')

    plt.suptitle("Pointing Game and Energy Verification: Visual Case Demonstrations\n[Green Dashed Box = Doctor Annotation | Crosshair 'X' = Peak Activation Point]", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    visual_path = os.path.join(output_dir, "xai_visual_pointing_game_examples.png")
    plt.savefig(visual_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {visual_path}")
    print("\nBenchmark successfully completed!")


if __name__ == "__main__":
    run_benchmark()
