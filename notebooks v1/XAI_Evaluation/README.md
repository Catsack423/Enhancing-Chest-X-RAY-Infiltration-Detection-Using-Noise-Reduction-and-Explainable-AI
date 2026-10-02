# Quantitative XAI Evaluation Suite (Pointing Game, Energy & IoU)

This folder contains the complete pipeline for **Objective, Statistical Evaluation of Explainable AI (Grad-CAM)** on 100 Infiltration Chest X-Rays with Radiologist Ground-Truth Bounding Boxes from the NIH ChestX-ray14 dataset.

---

## 1. Overview & Research Objectives

Evaluating XAI purely through qualitative heatmap visualizations is susceptible to cherry-picking bias. To provide rigorous empirical evidence for the Senior Seminar Thesis (Chapter 4 & 5), this suite computes 3 internationally recognized objective metrics:

1. **Pointing Game (Hit Rate %):**
   - Assesses whether the peak saliency coordinate $(x^*, y^*) = \arg\max H(x, y)$ falls strictly inside the doctor's bounding box.
   - Measures raw localization precision.

2. **Saliency Energy Inside BBox (%):**
   - $\text{Energy Inside} = \frac{\sum_{(x,y) \in \text{BBox}} H(x,y)}{\sum_{(x,y)} H(x,y)} \times 100\%$
   - Quantifies the concentration of attention on true pathological lesions vs. background noise, ribs, or collarbones.

3. **Intersection over Union (IoU):**
   - Spatial overlap between the thresholded heatmap ($H \ge \tau \cdot \max(H)$) and the ground-truth mask.

---

## 2. Directory Contents

```text
XAI_Evaluation/
│
├── xai_quantitative_metrics.ipynb         # Interactive Jupyter Notebook for single & batch analysis
├── xai_eval_utils.py                      # Core metric definitions (Pointing Game, Energy, IoU, Mask builder)
├── run_xai_quantitative_benchmark.py      # Automated benchmark runner over all 100 Infiltration samples
├── build_xai_eval_notebook.py             # Script to rebuild the notebook programmatically
│
├── output/
│   ├── xai_metrics_comparison_barchart.png # Bar chart comparing Hit Rate, Energy, and IoU across methods
│   ├── xai_energy_boxplot.png             # Boxplot of energy distribution showing statistical significance
│   └── xai_visual_pointing_game_examples.png # Visual cases with doctor BBoxes & peak crosshairs
│
└── reports/
    ├── xai_quantitative_summary.csv       # Aggregated Mean ± Std table across all 4 preprocessing methods
    └── xai_quantitative_metrics_detailed.csv # Per-image granular metric records for all 100 cases
```

---

## 3. How to Run

### Interactive Notebook:
Open `xai_quantitative_metrics.ipynb` in VS Code or Jupyter. Execute the cells to inspect individual patient cases or visualize the statistical distributions.

### Re-run Full 100-Sample Benchmark:
```bash
python run_xai_quantitative_benchmark.py
```
