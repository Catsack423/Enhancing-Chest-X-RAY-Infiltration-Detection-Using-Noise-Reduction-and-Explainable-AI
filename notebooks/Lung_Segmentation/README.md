# Lung Field Segmentation Suite

This module implements **Anatomical Lung Field Segmentation and Masking**, highlighted in Section 2.2.4 of the research proposal (*Rahman et al., 2021*), to confine CNN and Explainable AI attention strictly within the thoracic pulmonary boundaries.

---

## 1. Key Concept

In standard Chest X-Rays, Grad-CAM heatmaps frequently disperse onto clavicles, ribs, humeral heads, and subdiaphragmatic gas. By extracting an anatomical binary lung mask $M_{\text{lung}}$:
$$I_{\text{masked}} = I_{\text{raw}} \odot M_{\text{lung}}$$
non-pulmonary structures are completely suppressed (Pixel = 0), compelling the model to focus 100% of its gradient attention on true pulmonary parenchymal pathology.

---

## 2. Directory Contents

- `lung_segmentation_pipeline.ipynb`: Interactive Jupyter Notebook walking through mask extraction and before-and-after Grad-CAM comparisons.
- `lung_segmentation_utils.py`: Standalone lung field segmentation functions.
- `generate_lung_mask_artifacts.py`: Automated generation of high-res figures.
- `output/lung_segmentation_stages.png`: Step-by-step extraction stages.
- `output/gradcam_before_vs_after_masking.png`: Grad-CAM comparison with doctor ground truth BBoxes.
