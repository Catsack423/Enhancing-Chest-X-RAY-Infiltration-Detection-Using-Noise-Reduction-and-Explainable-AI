"""Sanity Check Script for Phase 2 Denoising Pipeline.

Tests all 4 denoising groups across strength levels on 4 sample images from the
training set (2 Infiltration-only, 2 No-Finding-only).

Key Validations:
  1. Strict uint8 [0, 255] contract verification for every output.
  2. Quantitative sanity metrics (PSNR, SSIM, mean pixel delta) across levels.
  3. Before/After visual comparison grids (full image + lung crop detail).
  4. Explicit visual tagging '[DAE (untrained) + CLAHE]' and '_untrained' filename suffix.
"""
import os
import sys
from typing import Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure package root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scripts.denoise as sd
from scripts.denoise.dae_clahe import is_dae_untrained

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/phase2_sanity_check"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two uint8 images."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute structural similarity index (SSIM) between two uint8 images (Wang et al. 2004)."""
    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    ksize = (11, 11)
    sigma = 1.5

    mu1 = cv2.GaussianBlur(i1, ksize, sigma)
    mu2 = cv2.GaussianBlur(i2, ksize, sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(i1 * i1, ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(i2 * i2, ksize, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(i1 * i2, ksize, sigma) - mu1_mu2

    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


def select_sample_images() -> List[Dict[str, str]]:
    """Select 2 Infiltration-only and 2 No-Finding-only samples from manifests."""
    train_inf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/manifests/train_infiltration.csv"))
    train_nrm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/manifests/train_normal.csv"))

    samples = []
    if os.path.isfile(train_inf_path):
        df_inf = pd.read_csv(train_inf_path)
        for _, row in df_inf.head(2).iterrows():
            samples.append({
                "image_index": row["Image Index"],
                "path": row["path"],
                "label": "Infiltration-only",
            })

    if os.path.isfile(train_nrm_path):
        df_nrm = pd.read_csv(train_nrm_path)
        for _, row in df_nrm.head(2).iterrows():
            samples.append({
                "image_index": row["Image Index"],
                "path": row["path"],
                "label": "No-Finding-only",
            })

    return samples


def crop_center_lung(img: np.ndarray, crop_size: int = 300) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract a representative lung region crop for texture and edge inspection."""
    h, w = img.shape[:2]
    # Center-left lung field
    y1 = int(h * 0.35)
    y2 = min(h, y1 + crop_size)
    x1 = int(w * 0.20)
    x2 = min(w, x1 + crop_size)
    return img[y1:y2, x1:x2], (y1, y2, x1, x2)


def run_sanity_checks() -> None:
    samples = select_sample_images()
    print("=" * 80)
    print("PHASE 2: SANITY CHECK & DENOISING PIPELINE VERIFICATION")
    print("=" * 80)
    print(f"Selected {len(samples)} training samples for verification:")
    for i, s in enumerate(samples, 1):
        print(f"  [{i}] {s['image_index']} | {s['label']} | {s['path']}")

    methods = ["baseline", "median", "clahe_dwt", "dae_clahe"]
    levels = [1, 2, 3]

    dae_untrained = is_dae_untrained()
    dae_display_name = "DAE (untrained) + CLAHE" if dae_untrained else "DAE (trained) + CLAHE"
    print(f"\nDAE Status: {'UNTRAINED (zero-residual identity coupling with CLAHE)' if dae_untrained else 'TRAINED'}")

    all_metrics = []

    for s_idx, sample in enumerate(samples, 1):
        img_path = sample["path"]
        img_id = sample["image_index"]
        label = sample["label"]

        raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if raw is None:
            print(f"[ERROR] Could not read {img_path}")
            continue

        print(f"\nProcessing Sample {s_idx}/{len(samples)}: {img_id} ({label}) [size: {raw.shape}]")

        # Container for processed images: {method: {level: img}}
        results = {"raw": raw}

        for method in methods:
            results[method] = {}
            for lvl in (levels if method != "baseline" else [1]):
                denoised = sd.apply_denoise(raw, method=method, level=lvl)

                # Strict Contract Assertions
                assert isinstance(denoised, np.ndarray), f"Contract FAIL: {method} L{lvl} not np.ndarray"
                assert denoised.dtype == np.uint8, f"Contract FAIL: {method} L{lvl} dtype is {denoised.dtype} (expected uint8)"
                assert denoised.shape == raw.shape, f"Contract FAIL: {method} L{lvl} shape {denoised.shape} != {raw.shape}"
                assert denoised.min() >= 0 and denoised.max() <= 255, f"Contract FAIL: values out of [0, 255]"

                results[method][lvl] = denoised

                # Compute metrics vs raw
                diff = np.mean(np.abs(denoised.astype(float) - raw.astype(float)))
                psnr = compute_psnr(raw, denoised)
                ssim = compute_ssim(raw, denoised)

                all_metrics.append({
                    "sample": img_id,
                    "label": label,
                    "method": method,
                    "level": lvl,
                    "mean_delta": round(diff, 3),
                    "psnr_db": round(psnr, 2) if psnr != float("inf") else "inf (identical)",
                    "ssim": round(ssim, 4),
                    "contract_ok": True,
                })

        # Generate Visual Comparisons
        # 1. Full Image Multi-Method Comparison Grid
        fig, axes = plt.subplots(3, 4, figsize=(18, 14))
        plt.subplots_adjust(wspace=0.15, hspace=0.25)

        # Row 0: Original, Baseline, Median L1, Median L2
        axes[0, 0].imshow(raw, cmap="gray")
        axes[0, 0].set_title(f"Original ({label})\n{img_id}", fontsize=11, fontweight="bold")

        axes[0, 1].imshow(results["baseline"][1], cmap="gray")
        axes[0, 1].set_title("Baseline (No-op)\nPSNR: inf | SSIM: 1.000", fontsize=10)

        axes[0, 2].imshow(results["median"][1], cmap="gray")
        axes[0, 2].set_title("Median L1 (k=3)\nWeak smoothing", fontsize=10)

        axes[0, 3].imshow(results["median"][2], cmap="gray")
        axes[0, 3].set_title("Median L2 (k=5)\nMedium smoothing", fontsize=10)

        # Row 1: Median L3, CLAHE+DWT L1, CLAHE+DWT L2, CLAHE+DWT L3
        axes[1, 0].imshow(results["median"][3], cmap="gray")
        axes[1, 0].set_title("Median L3 (k=7)\nStrong smoothing", fontsize=10)

        axes[1, 1].imshow(results["clahe_dwt"][1], cmap="gray")
        axes[1, 1].set_title("CLAHE+DWT L1\n(clip=2.0, dwt=1, s=0.5)", fontsize=10)

        axes[1, 2].imshow(results["clahe_dwt"][2], cmap="gray")
        axes[1, 2].set_title("CLAHE+DWT L2\n(clip=4.0, dwt=2, s=1.0)", fontsize=10)

        axes[1, 3].imshow(results["clahe_dwt"][3], cmap="gray")
        axes[1, 3].set_title("CLAHE+DWT L3\n(clip=8.0, dwt=3, s=2.0)", fontsize=10)

        # Row 2: DAE+CLAHE across levels (with explicit untrained badge)
        for i, lvl in enumerate(levels):
            axes[2, i].imshow(results["dae_clahe"][lvl], cmap="gray")
            axes[2, i].set_title(
                f"[{dae_display_name}]\nLevel {lvl} (clip={lvl * 2 if lvl <= 2 else 8}.0)",
                fontsize=10,
                color="navy" if dae_untrained else "black"
            )

        # Extra summary quadrant
        axes[2, 3].axis("off")
        info_text = (
            f"Image: {img_id}\n"
            f"Category: {label}\n"
            f"Original Size: {raw.shape}\n\n"
            f"Contract Check: PASS (uint8 [0, 255])\n"
            f"DAE Mode: {'UNTRAINED identity' if dae_untrained else 'TRAINED'}\n\n"
            f"Notice: DAE operates via zero-residual\n"
            f"identity pass prior to Phase 3 training."
        )
        axes[2, 3].text(0.1, 0.5, info_text, fontsize=10, verticalalignment="center",
                        bbox=dict(boxstyle="round,pad=0.8", facecolor="aliceblue", edgecolor="royalblue"))

        for ax in axes.flat:
            ax.axis("off")

        full_fig_path = os.path.join(RESULTS_DIR, f"sample_{s_idx}_{img_id.replace('.png', '')}_comparison_dae_untrained.png")
        fig.suptitle(f"Phase 2 Denoising Comparison: {img_id} ({label})", fontsize=15, fontweight="bold", y=0.98)
        fig.savefig(full_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Saved full comparison: {full_fig_path}")

        # 2. Zoomed-In Lung Crop Detail Comparison
        crop_raw, (y1, y2, x1, x2) = crop_center_lung(raw)
        crop_fig, c_axes = plt.subplots(2, 5, figsize=(18, 8))
        plt.subplots_adjust(wspace=0.15, hspace=0.25)

        c_axes[0, 0].imshow(crop_raw, cmap="gray")
        c_axes[0, 0].set_title("Original Crop\n(Lung Parenchyma)", fontsize=10, fontweight="bold")

        c_axes[0, 1].imshow(results["median"][1][y1:y2, x1:x2], cmap="gray")
        c_axes[0, 1].set_title("Median L1 (k=3)", fontsize=10)

        c_axes[0, 2].imshow(results["median"][2][y1:y2, x1:x2], cmap="gray")
        c_axes[0, 2].set_title("Median L2 (k=5)", fontsize=10)

        c_axes[0, 3].imshow(results["median"][3][y1:y2, x1:x2], cmap="gray")
        c_axes[0, 3].set_title("Median L3 (k=7)\n[Check Smoothing]", fontsize=10, color="darkred")

        c_axes[0, 4].imshow(results["baseline"][1][y1:y2, x1:x2], cmap="gray")
        c_axes[0, 4].set_title("Baseline (Raw)", fontsize=10)

        c_axes[1, 0].imshow(results["clahe_dwt"][1][y1:y2, x1:x2], cmap="gray")
        c_axes[1, 0].set_title("CLAHE+DWT L1", fontsize=10)

        c_axes[1, 1].imshow(results["clahe_dwt"][2][y1:y2, x1:x2], cmap="gray")
        c_axes[1, 1].set_title("CLAHE+DWT L2", fontsize=10)

        c_axes[1, 2].imshow(results["clahe_dwt"][3][y1:y2, x1:x2], cmap="gray")
        c_axes[1, 2].set_title("CLAHE+DWT L3\n[Enhanced Contrast]", fontsize=10, color="darkblue")

        c_axes[1, 3].imshow(results["dae_clahe"][1][y1:y2, x1:x2], cmap="gray")
        c_axes[1, 3].set_title(f"[{dae_display_name}]\nL1 (clip=2.0)", fontsize=9, color="navy")

        c_axes[1, 4].imshow(results["dae_clahe"][3][y1:y2, x1:x2], cmap="gray")
        c_axes[1, 4].set_title(f"[{dae_display_name}]\nL3 (clip=8.0)", fontsize=9, color="navy")

        for ax in c_axes.flat:
            ax.axis("off")

        crop_fig_path = os.path.join(RESULTS_DIR, f"sample_{s_idx}_{img_id.replace('.png', '')}_crop_dae_untrained.png")
        crop_fig.suptitle(f"Lung Parenchyma Crop Detail: {img_id} ({label})", fontsize=14, fontweight="bold", y=0.98)
        crop_fig.savefig(crop_fig_path, dpi=150, bbox_inches="tight")
        plt.close(crop_fig)
        print(f"  -> Saved crop detail: {crop_fig_path}")

    # Print summary metrics table
    df_metrics = pd.DataFrame(all_metrics)
    csv_metrics_path = os.path.join(RESULTS_DIR, "phase2_sanity_metrics.csv")
    df_metrics.to_csv(csv_metrics_path, index=False)
    print(f"\nSaved metrics CSV to: {csv_metrics_path}")

    print("\n" + "=" * 80)
    print("SANITY CHECK METRICS SUMMARY")
    print("=" * 80)
    print(df_metrics[["sample", "method", "level", "mean_delta", "psnr_db", "ssim", "contract_ok"]].to_string(index=False))
    print("\nALL CONTRACT CHECKS PASSED: Every output strictly complies with uint8 [0, 255].")


if __name__ == "__main__":
    run_sanity_checks()
