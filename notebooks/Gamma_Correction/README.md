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
- `output/gamma_levels_comparison.png`: Visual evaluation of $\gamma = 0.5, 0.8, 1.0, 1.2, 1.5$.
- `output/gamma_vs_clahe_benchmark.png`: Direct face-off with CLAHE and DAE+CLAHE.
