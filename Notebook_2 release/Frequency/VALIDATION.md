# Validation — frequency notebook

Date: 9 October 2026. Full research training has not been run.

## Verified on local CPU

- 11 focused tests pass: DWT reconstruction and L1 ordering; constant-image behavior;
  DFT band routing including exact cutoff boundaries; invalid settings; feature-fusion
  equivalence and gradients to every view; Grad-CAM single-view equivalence and view
  permutation; signed multi-view SHAP aggregation; original data/settings; resume
  rejection for changed manifests; notebook schema and Python syntax.
- Existing manifests preserved byte-for-byte: train 320 / 311 patients; validation
  80 / 80; classification test 100 / 90; XAI 123 / 115; SHAP background 32 / 32.
- Baseline preprocessing tensor matches the old implementation exactly.
- Packaged nested data ZIP matches the existing nih_cxr_subset.zip by SHA-256.
  All five packaged manifests match the original run byte-for-byte; ZIP CRC passes.
- No main-run scores are populated and no original experiment results are overwritten.

## GPU validation

Passed on NVIDIA GeForce RTX 4060 8 GB, with Python 3.11, torch 2.13.0+cu130,
torchvision 0.28.0+cu130 and SHAP 0.51.0. CUDA-enabled dependencies are installed
in the project-local `.gpu/` environment. The machine's original Python is CPU-only.

- All **35 tests** pass (11 frequency checks + 24 existing pipeline checks).
- Executed the delivered notebook's setup/data/preview/training/Grad-CAM/SHAP/report
  code cells with isolated smoke overrides: 4 train, 4 validation, 4 test, 2 XAI,
  2 backgrounds, 1 epoch, SHAP nsamples=4. All 3 CNN conditions and both XAI
  explainers completed, in **48.4 seconds** after dependencies/weights were ready.
- The output notebook is `.runtime/frequency_gpu_validation/executed_smoke.ipynb`.
  Its scores are smoke checks and must never be cited as research results.
- Tested the configured original **batch=16 source images**, **SHAP nsamples=200**,
  **background=32** and **SHAP batch_size=8** for each method without out-of-memory.

| Method | Warm training step, batch 16 | SHAP per image, 200 samples | Peak training allocation | Peak SHAP allocation |
|---|---:|---:|---:|---:|
| Baseline | 0.051 s | 1.52 s | 0.52 GiB | 1.02 GiB |
| DWT A4 + D1–D4 | 0.214 s | 8.26 s | 1.18 GiB | 4.15 GiB |
| DFT low/mid/high | 0.133 s | 4.60 s | 0.85 GiB | 2.58 GiB |

Times are small benchmarks using two training steps and one full-budget SHAP image
per method. CUDA allocations exclude other apps and reserved memory. Estimated
full run (three models, up to 50 epochs each, both XAI methods on all 123 images):
**45–90 minutes** on this machine, depending on early stopping, file I/O and other
GPU activity. Full research training has not been started.

Raw timing records are `.runtime/frequency_gpu_validation/benchmark.json` and
the completion marker is `smoke_validation.json` in the same folder.

Reproduce focused checks:

```powershell
python -m unittest discover -s 'Notebook_2 release/tests' -p test_frequency.py -v
```

After notifying the user about expected duration, execute actual notebook cells with
separate smoke overrides and benchmark the configured batch/XAI settings:

```powershell
& '.gpu/Scripts/python.exe' 'Notebook_2 release/Frequency/validate_notebook_gpu.py'
```

This uses a subset of existing IDs (no new split), 4 train / 4 validation / 4 test,
2 XAI, 2 backgrounds, 1 epoch, SHAP nsamples=4. Then it benchmarks batch=16 and
SHAP nsamples=200/background=32 on one image per method. These are hardware/pipeline
checks, with artifacts under `.runtime/frequency_gpu_validation/` only.
