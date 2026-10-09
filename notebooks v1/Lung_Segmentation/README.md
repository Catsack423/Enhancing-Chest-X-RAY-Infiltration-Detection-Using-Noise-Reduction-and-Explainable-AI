# Lung Field Segmentation Suite

This module implements **Anatomical Lung Field Segmentation and Masking**, highlighted in Section 2.2.4 of the research proposal (*Rahman et al., 2021*), to confine CNN and Explainable AI attention strictly within the thoracic pulmonary boundaries.

---

## 1. Key Concept

In standard Chest X-Rays, Grad-CAM heatmaps frequently disperse onto clavicles, ribs, humeral heads, and subdiaphragmatic gas. By extracting an anatomical binary lung mask $M_{\text{lung}}$:
$$I_{\text{masked}} = I_{\text{raw}} \odot M_{\text{lung}}$$
non-pulmonary structures are completely suppressed (Pixel = 0), compelling the model to focus 100% of its gradient attention on true pulmonary parenchymal pathology.

---

## 2. Directory Contents

- `lung_segmentation_pipeline.ipynb`: Interactive Jupyter Notebook walking through mask extraction, Grad-CAM comparisons, and Confusion Matrices.
- `lung_segmentation_utils.py`: Standalone lung field segmentation functions.
- `generate_lung_mask_artifacts.py`: Automated generation of high-res figures.
- `generate_lung_mask_confusion_matrices.py`: Benchmark script generating 4-condition Confusion Matrices and metrics.
- `output/`:
  - `lung_segmentation_stages.png`: Step-by-step extraction stages.
  - `gradcam_before_vs_after_masking.png`: Grad-CAM comparison with doctor ground truth BBoxes.
  - `confusion_matrix_lung_segmentation_classification.png`: 4-panel Binary Classification Confusion Matrix.
  - `confusion_matrix_lung_segmentation_localization.png`: 4-panel XAI Localization (Pointing Game) Confusion Matrix.
- `reports/`:
  - `lung_segmentation_metrics.csv`: Quantitative summary metrics table across all 4 conditions.
  - `lung_segmentation_detailed_records.csv`: Per-case evaluation results.
  - `summary_report.md`: Comprehensive academic summary report for chapter 4.

---

## 3. Benchmark Summary (Lung Segmentation vs DAE+CLAHE)

| Condition | Accuracy | Recall (Sensitivity) | F1-Score | Pointing Game Hit Rate | Mean Energy Inside BBox |
|---|:---:|:---:|:---:|:---:|:---:|
| **1. Unmasked Raw CXR (Baseline)** | 86.67% | 90.00% | 0.8710 | 23.33% | 12.08% |
| **2. Segmented Lung Only (Raw Masked)** | **91.67%** | **93.33%** | **0.9180** | 26.67% | **16.97%** |
| **3. Unmasked DAE+CLAHE** | 90.00% | 93.33% | 0.9032 | 26.67% | 13.53% |
| **4. Segmented Lung + DAE+CLAHE (Combined)** | **91.67%** | **93.33%** | **0.9180** | **40.00%** | **17.31%** |

