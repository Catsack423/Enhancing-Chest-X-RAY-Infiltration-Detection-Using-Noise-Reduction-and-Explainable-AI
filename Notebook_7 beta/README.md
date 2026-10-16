# v7 beta — low-estimated-noise class160 CNN pilot

Run **run_v7_colab.ipynb** in Google Colab with a GPU. The notebook embeds its source code;
no clone or new full-dataset upload is needed. `v7_source.zip` is a source-only recovery package.

## Drive paths

- Read-only NIH input: `/content/drive/MyDrive/nih -xray/data/3`
- Durable v7 workspace: `/content/drive/MyDrive/nih -xray/Notebook_7_beta`
- Windows mapping supplied by the user: `G:\My Drive\nih -xray\data\3`

Open the notebook from Drive, select **Runtime → Change runtime type → GPU**, then **Run all**.
Authorize the Drive mount. The full scan runs in Colab, not on the local PC.

The notebook writes these reusable artifacts to the v7 workspace:

| Artifact | Purpose |
|---|---|
| `audit/chunks/` | Atomic 256-image audit checkpoints; safe to resume on another VM |
| `audit/noise_audit.csv`, `audit/summary.json` | Full 112,120-image noise proxy and audit integrity |
| `prepared/manifests/`, `prepared/figures/` | Frozen selections, AP/PA counts, noise distributions and examples |
| `selected_data/`, `v7_selected_data.zip`, `selected_package.json` | Native selected images, XAI reference, manifests and package hashes |
| `runs/v7_beta/seed_42/DAE/` | One DAE's best/last checkpoints, epoch history and corruption logs |
| `runs/v7_beta/seed_42/CNN/<method>/` | Each CNN's best/last checkpoints, true logits, predictions and model specification |
| `runs/v7_beta/seed_42/reports/` | Classification, 95% patient CIs, paired baseline deltas, PSNR/SSIM and figures |
| `runs/v7_beta/seed_42/xai_ready/` | Native input geometry, original 123 bbox manifest and train-only SHAP background metadata; images remain in the selected package |

After a disconnected runtime, Run all again. Completed audit chunks, the selected-image package,
and completed epochs are reused. Only the selected-image ZIP is staged onto the Colab VM for
fast training. This version does not stage the entire NIH corpus or consume `v3_selected_zip`,
which does not cover all 112,120 images required by the audit. Dataset folders may contain
`images_001/images/*.png` or `images_001/*.png`.

## Locked experiment

- **Train 160+160; validation 40+40; test 50+50**. Infiltration-only vs strictly No Finding.
- Sample the lowest estimated noise in each class/frozen partition; break ties by image ID.
  Existing v4.3 patient partitions remain unchanged. No score threshold is relaxed to fill a quota.
- The score is db2 wavelet MAD, measured on 64x64 patches in the central 80% of native images.
  At least 90% of patch intensities must be 5–250 and the patch must be nonconstant.
  Require eight usable patches; take the median sigma from the lowest-gradient 25% of patches.
  Keep global sigma, contrast, sharpness and source image SHA-256 as additional fields.
- **12 v4.3 conditions**: Baseline; Median 3/5/7; DWT→CLAHE threshold 0.5/1/2 with db1/depth3/clip2;
  DAE→CLAHE blend 0.25/0.5/1 with clip2; DWT five-view fusion; DFT three-view fusion.
  There are no strength-L4 conditions. DWT's four decomposition levels are retained.
- **16 corruption states**: clean reference plus Gaussian, Poisson, Poisson–Gaussian,
  salt-and-pepper and speckle at three strengths. Exact parameters are embedded in
  `v7/noise.py` and recorded in each run's `experiment.json`.
- Noise RNG depends on seed/image/family/strength/epoch, never the denoising method.
  Each original train image gets one of the 16 states each epoch; no corrupted copies are
  counted as additional independent train subjects. Validation/test corruption uses epoch 0.
- Work at 256x256, then use the inherited full-frame 224x224 CNN resize and ImageNet normalization.
  No crop is introduced. Frequency views preserve the inherited signed-detail encoding.
- Seed 42 only. ResNet50 ImageNet V2, frozen early layers and all BN statistics, head warmup
  epochs 1–3, layer4+head thereafter, at most 20 epochs total and patience 3 after warmup.
  AdamW and geometric augmentation match v4.3; balanced train uses `pos_weight=1`.
- A DAE is trained from train only, with pre-corruption reference targets, for 20 epochs.
  Effective optimizer batch 64 is accumulated from microbatches of 8 to fit smaller GPUs.
  DAE BatchNorm therefore sees microbatches of 8; this v7 runtime choice is recorded.
- Select checkpoints by balanced BCE across all 16 validation states. Select one Youden
  threshold per model on the same validation mixture; keep it fixed across all test states.
  Report threshold 0.5 as an additional reference. Test never chooses parameters/checkpoints.

## Interpretation

Low estimated noise does not prove absence of noise. Anatomy, blur and processing can affect
the score. The cohort is deliberately selected and does not estimate full-NIH population
performance. Salt-and-pepper/speckle are extra robustness tests, not claims about detector noise.
L1/N1 across different families do not imply equal physical noise power. Record measured
noise mean/std/RMSE and clipping after uint8 quantization for the actual corruptions.

PSNR/SSIM compare a processed image with the same image before synthetic corruption. This is a
reference **proxy**, not a clean detector truth. Infinite PSNR denotes exact identity. Fusion
representations have intentionally unavailable PSNR/SSIM. Main classification metrics and
paired patient-bootstrap 95% CIs are reported separately for each state, with 2,000 draws.

Noise formulas follow [scikit-image](https://scikit-image.org/docs/stable/api/skimage.util.html#skimage.util.random_noise)
and the [Poisson–Gaussian model of Foi et al.](https://doi.org/10.1109/TIP.2008.2001399).
The stated strengths are v7's experimental settings, not reproduced parameter values from those papers.

## Future XAI interface

`v7.runner.load_classifier(experiment, condition_id, device)` restores the validation-selected
model. First open a saved run with `xai_interface.open_experiment(root, data_dir, output)`;
this is read-only and checks the recorded training sources and core versions, so adding a
new explainer script or installing SHAP does not require retraining the CNN.
`v7.dataset.Preprocessor` produces `(V,3,H,W)` from image ID, corruption ID and epoch.
The CNN accepts `(N,V,3,H,W)`, returns `(N,1)` Infiltration logits and exposes `target_layer`
(`backbone.layer4`). Its forward never detaches inputs. Use the same preprocessing, view order,
checkpoint and full-frame coordinate mapping for CNN and XAI.

The selected-image package retains the native 123 bbox images (including mixed cases),
outside train/validation, and a fixed 16+16 background from the selected train images. These
are not extra train images. This version does not calculate Grad-CAM/SHAP or localization metrics.

## Local validation / rebuilding

```powershell
../.gpu/Scripts/python.exe -m unittest discover -s tests -v
../.gpu/Scripts/python.exe run.py smoke --device cuda
../.gpu/Scripts/python.exe build_notebook.py
```

The smoke run uses synthetic data, random CNN initialization and reduced 64px tensors.
It exercises all methods, state evaluation, the warmup transition, checkpoint reload, input
gradients and byte-stable completed resume. These are not research results or a full-size VRAM benchmark.
TF32 is disabled and recorded: its batch-dependent rounding otherwise changes single-image XAI predictions.
The full experiment intentionally runs on Colab. Resume refuses changed code, versions,
config, manifests or image hashes; keep the old workspace and select a new `WORK` for a new protocol.
