# DAE + CLAHE Evaluation Suite (Senior Seminar Project)

This folder contains the complete implementation, training pipeline, and evaluation artifacts for **DAE + CLAHE (Denoising Autoencoder + Contrast Limited Adaptive Histogram Equalization)**, directly addressing **Research Gap 6** from Chapter 3 of the thesis.

---

## 1. Directory Structure

```text
DAE_CLAHE/
│
├── dae_clahe_pipeline.ipynb                # Interactive Jupyter Notebook for DAE+CLAHE experiments
├── dae_clahe_utils.py                      # Core architecture, loss functions, metrics & strict uint8 contract
├── train_dae_model.py                      # Standalone training script for DAE with Poisson+Gaussian noise
├── generate_dae_clahe_artifacts.py         # Automated generation of visual figures and confusion matrices
├── build_dae_notebook.py                   # Script to rebuild the notebook programmatically
│
├── checkpoints/
│   └── dae_trained.pth                     # Pre-trained DAE weights with verified loss convergence
│
├── output/
│   ├── dae_clahe_pipeline_stages.png       # 4-stage pipeline: Raw -> Noisy -> DAE -> CLAHE Level 2
│   ├── dae_levels_comparison.png           # Level 1 (2.0), Level 2 (4.0), Level 3 (8.0) comparison
│   ├── four_denoise_methods_faceoff.png    # Direct comparison: Raw vs Median vs CLAHE+DWT vs DAE+CLAHE
│   ├── dae_gradcam_localization_comparison.png # ResNet50 Grad-CAM localization on Doctor Bounding Box
│   ├── confusion_matrix_dae_classification.png # Classification performance on Infiltration vs Normal
│   └── confusion_matrix_dae_localization.png   # Localization overlap performance on radiologist BBox
│
└── reports/
    ├── summary_report.md                   # Comprehensive academic thesis report
    └── dae_vs_traditional_metrics.csv      # CSV data table of PSNR, SSIM, EPI, and CIR metrics
```

---

## 2. Key Highlights Addressing Research Gap 6

1. **Over-smoothing Prevention:**
   - Traditional **Median Filter** aggressively blurs delicate boundaries of hazy, diffuse infiltrative opacities.
   - **DAE with U-Net Skip Connections** preserves fine lung markings and edge continuity while removing Poisson quantum noise and Gaussian sensor noise.

2. **Contrast Amplification:**
   - Followed by **CLAHE (Level 2: clip_limit=4.0, tile_grid=(8,8))**, the subtle opacity contrast between healthy parenchyma and infiltration is boosted by **1.2x - 1.7x**.

3. **Explainable AI (Grad-CAM) Impact:**
   - On raw noisy images, Grad-CAM gradients often disperse toward the collarbone, rib edges, or diaphragm.
   - With DAE + CLAHE preprocessing, the activation gradient focuses tightly within the ground-truth **Radiologist Bounding Box**.

---

## 3. How to Run

### Interactive Notebook:
Open `dae_clahe_pipeline.ipynb` in VS Code or Jupyter Notebook. Ensure the kernel is set to Python 3.11 or Python 3.14.

### Retrain DAE:
```bash
python train_dae_model.py
```

### Regenerate All Visual Artifacts & Matrices:
```bash
python generate_dae_clahe_artifacts.py
```
