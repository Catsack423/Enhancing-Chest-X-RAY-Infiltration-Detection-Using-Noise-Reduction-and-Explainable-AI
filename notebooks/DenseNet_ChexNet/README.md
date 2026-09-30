# DenseNet121 (ChexNet) Architecture Suite

This module benchmarks **DenseNet121** against **ResNet50**, referencing *Rajpurkar et al. (2017) (CheXNet)* and Section 2.2.6 of the senior seminar proposal, to investigate how dense concatenative feature reuse captures diffuse, ill-defined infiltration opacities compared to residual additive architectures.

---

## 1. Key Concept

- **ResNet50:** $\mathbf{x}_l = H_l(\mathbf{x}_{l-1}) + \mathbf{x}_{l-1}$ (Additive shortcut).
- **DenseNet121:** $\mathbf{x}_l = H_l([\mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{l-1}])$ (Iterative feature concatenation).

Dense connectivity retains low-level edge and texture representations into deeper classification layers, making it the proven gold standard for chest radiographs.

---

## 2. Directory Contents

- `densenet_vs_resnet_pipeline.ipynb`: Interactive Jupyter Notebook comparing ResNet50 vs DenseNet121 across Raw and DAE+CLAHE conditions.
- `densenet_utils.py`: DenseNet121 Grad-CAM extractor (`features.denseblock4.denselayer16.conv2`).
- `generate_densenet_artifacts.py`: Automated generation of comparison figures.
- `generate_densenet_confusion_matrices.py`: Benchmark script generating 4-condition Confusion Matrices and metrics.
- `output/`:
  - `densenet_vs_resnet_gradcam_comparison.png`: Visual evaluation with doctor ground truth BBoxes.
  - `confusion_matrix_densenet_vs_resnet_classification.png`: 4-panel Binary Classification Confusion Matrix.
  - `confusion_matrix_densenet_vs_resnet_localization.png`: 4-panel XAI Localization (Pointing Game) Confusion Matrix.
- `reports/`:
  - `densenet_vs_resnet_metrics.csv`: Quantitative summary metrics table across all 4 conditions.
  - `densenet_vs_resnet_detailed_records.csv`: Per-case evaluation results.
  - `summary_report.md`: Comprehensive academic summary report for chapter 4.

---

## 3. Benchmark Summary (ResNet50 vs DenseNet121)

| Model & Condition | Accuracy | Recall (Sensitivity) | F1-Score | Pointing Game Hit Rate | Mean Energy Inside BBox |
|---|:---:|:---:|:---:|:---:|:---:|
| **ResNet50 (Raw CXR)** | 86.67% | 90.00% | 0.8710 | 23.33% | 12.08% |
| **DenseNet121 (Raw CXR)** | 91.67% | 93.33% | 0.9180 | 13.33% | 8.62% |
| **ResNet50 (DAE+CLAHE)** | 90.00% | 93.33% | 0.9032 | 26.67% | 13.53% |
| **DenseNet121 (DAE+CLAHE)** | **95.00%** | **96.67%** | **0.9508** | 13.33% | 10.99% |

