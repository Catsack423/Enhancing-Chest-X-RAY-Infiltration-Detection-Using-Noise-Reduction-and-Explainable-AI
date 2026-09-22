# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Overview

Research project: Enhancing pneumonia/infiltration detection on chest X-rays using denoising techniques and Explainable AI (Grad-CAM, SHAP).
Single source of truth for experiment specification: `guideline.md`.

## Dataset & Structure

- **Dataset**: NIH ChestX-ray14 (`data/versions/3/`)
  - Image folders: `data/versions/3/images_001/` to `images_012/`
  - Metadata: `Data_Entry_2017.csv`, `BBox_List_2017.csv`
  - Official Splits: `train_val_list.txt`, `test_list.txt` (Patient-level split; do NOT re-split)
- **Task Type**: Binary classification — `Infiltration` vs `No Finding` (Normal only)
  - Negative class must strictly be "No Finding" images (exclude images with other co-occurring pathologies)
- **XAI Evaluation**:
  - Infiltration Bounding Box subset: 123 images (all confirmed in `test_list.txt`)
  - Metrics: IoU / Pointing Game accuracy against ground truth bounding box + 95% bootstrap CI

## Experimental Setup & Architecture

- **Backbone**: Pretrained CNN (ResNet50) with ImageNet weights, frozen early layers. Fixed across all runs.
- **Denoising Methods (4 groups)**:
  1. Baseline (raw image, no denoise)
  2. Single traditional: Median Filter (varying kernel size)
  3. Fusion 1: CLAHE + DWT (varying clip limits / wavelets)
  4. Fusion 2: DAE (custom trained) + CLAHE
- **Key Experiment Variable**: Noise reduction strength levels across methods to study over-smoothing effects on XAI IoU vs classification metrics.
