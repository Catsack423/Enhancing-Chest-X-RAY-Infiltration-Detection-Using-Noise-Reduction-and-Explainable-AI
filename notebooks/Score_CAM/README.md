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

- `score_cam_pipeline.ipynb`: Interactive Jupyter Notebook comparing Grad-CAM vs Score-CAM.
- `score_cam_utils.py`: Standalone `ScoreCAMGenerator` class.
- `generate_score_cam_artifacts.py`: Generates high-res visual comparison figures.
- `output/score_cam_vs_gradcam_comparison.png`: Visual benchmark with doctor ground truth BBoxes.
