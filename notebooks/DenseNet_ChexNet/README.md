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
- `output/densenet_vs_resnet_gradcam_comparison.png`: Visual evaluation with doctor ground truth BBoxes.
