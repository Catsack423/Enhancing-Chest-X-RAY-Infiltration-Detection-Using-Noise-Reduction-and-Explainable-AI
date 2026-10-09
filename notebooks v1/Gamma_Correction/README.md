# Gamma Correction Suite for Chest X-Ray Enhancement

This module implements **Gamma Correction**, a non-linear contrast adjustment technique highlighted in *Rahman et al. (2021)* which reported superior classification accuracy (96.29%) on chest radiographs.

---

## 1. Key Concept

$$I_{\text{out}} = 255 \times \left(\frac{I_{\text{in}}}{255}\right)^\gamma$$

- $\gamma < 1.0$ (e.g. 0.5, 0.8): Brightens dark lung parenchyma, improving detection of faint diffuse infiltration.
- $\gamma = 1.0$: Baseline raw image.
- $\gamma > 1.0$ (e.g. 1.2, 1.5): Compresses dark areas, sharpens dense consolidations and high-contrast borders.

---

## 2. Directory Contents

- `gamma_correction_pipeline.ipynb`: Interactive Jupyter Notebook testing different gamma values and benchmarking against CLAHE.
- `gamma_utils.py`: High-speed Look-Up Table (LUT) gamma functions and contrast metrics.
- `generate_gamma_artifacts.py`: Automated generation of comparison charts.
- `generate_gamma_confusion_matrices.py`: Benchmark script generating 4-condition Confusion Matrices and metrics.
- `output/`:
  - `gamma_levels_comparison.png`: Visual evaluation of $\gamma = 0.5, 0.8, 1.0, 1.2, 1.5$.
  - `gamma_vs_clahe_benchmark.png`: Direct face-off with CLAHE and DAE+CLAHE.
  - `confusion_matrix_gamma_classification.png`: 4-panel Binary Classification Confusion Matrix.
  - `confusion_matrix_gamma_localization.png`: 4-panel XAI Localization (Pointing Game) Confusion Matrix.
- `reports/`:
  - `gamma_confusion_matrix_summary.csv`: Quantitative summary metrics table across all 4 conditions.
  - `gamma_evaluation_records.csv`: Per-case evaluation results.
  - `summary_report.md`: Comprehensive academic summary report for chapter 4.

---

## 3. Benchmark Summary (Gamma Correction vs CLAHE)

| Enhancement Technique | Accuracy | Recall (Sensitivity) | F1-Score | Pointing Game Hit Rate | Mean Energy Inside BBox |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline Raw (gamma=1.0)** | 86.67% | 90.00% | 0.8710 | 23.33% | 12.08% |
| **Gamma = 0.8 (Optimal Bright)** | **91.67%** | **93.33%** | **0.9180** | **23.33%** | 11.11% |
| **Gamma = 1.2 (Contrast Dark)** | 88.33% | 90.00% | 0.8852 | 16.67% | **13.55%** |
| **CLAHE Benchmark (clip=4.0)** | 90.00% | 93.33% | 0.9032 | 23.33% | 12.10% |

