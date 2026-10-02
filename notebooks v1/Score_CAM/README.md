# Score-CAM Suite (Gradient-Free Visual Explanations)

This module implements **Score-CAM**, a gradient-free explainable AI technique recommended in the research proposal (*Rahman et al., 2021; Wang et al., 2020*), designed to resolve gradient saturation and noisy backpropagation gradients.

---

## 1. Key Concept

Unlike Grad-CAM which uses $\frac{\partial Y^c}{\partial A^k}$, Score-CAM uses activation maps as spatial masks directly on the input image:
$$M_k = I \odot \text{Upsample}(A^k)$$
Forward score weights $\alpha_k$ are obtained by passing $M_k$ through the network:
$$\alpha_k = \text{Softmax}\left(f(M_k)_c - f(I_b)_c\right)$$
$$L_{\text{Score-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$

---

## 2. Directory Contents

- `score_cam_pipeline.ipynb`: Interactive Jupyter Notebook comparing Grad-CAM vs Score-CAM and evaluating Confusion Matrices.
- `score_cam_utils.py`: Standalone `ScoreCAMGenerator` class.
- `generate_score_cam_artifacts.py`: Generates high-res visual comparison figures.
- `generate_score_cam_confusion_matrices.py`: Benchmark script generating 4-condition Confusion Matrices and metrics.
- `output/`:
  - `score_cam_vs_gradcam_comparison.png`: Visual benchmark with doctor ground truth BBoxes.
  - `confusion_matrix_score_cam_classification.png`: 4-panel Binary Classification Confusion Matrix.
  - `confusion_matrix_score_cam_localization.png`: 4-panel XAI Localization (Pointing Game) Confusion Matrix.
- `reports/`:
  - `score_cam_metrics.csv`: Quantitative summary metrics table across all 4 conditions.
  - `score_cam_detailed_records.csv`: Per-case evaluation results.
  - `summary_report.md`: Comprehensive academic summary report for chapter 4.

---

## 3. Benchmark Summary (Score-CAM vs Grad-CAM)

| XAI Technique & Condition | Accuracy | Recall (Sensitivity) | F1-Score | Pointing Game Hit Rate | Normal Clean Rate |
|---|:---:|:---:|:---:|:---:|:---:|
| **1. Grad-CAM (Raw CXR)** | 87.50% | 90.00% | 0.8780 | 15.00% | 85.00% |
| **2. Score-CAM (Raw CXR)** | 87.50% | 90.00% | 0.8780 | 15.00% | **90.00%** |
| **3. Grad-CAM (DAE+CLAHE)** | 87.50% | 90.00% | 0.8780 | **25.00%** | 85.00% |
| **4. Score-CAM (DAE+CLAHE)** | 87.50% | 90.00% | 0.8780 | 10.00% | **90.00%** |

