# v7 beta validation — 16 October 2026

## Passed locally

- **15 focused tests passed** in the repository `.gpu` Python 3.11 environment.
- Gaussian sigma calibration (0.01/0.03/0.05), black borders, constant/too-small images,
  corruption formulas and fixed RNG, PSNR/SSIM identity, frozen strict-label selection,
  patient isolation, insufficient quotas, corrupt images, vendor byte hashes and view shapes.
- A separate 135-image synthetic fixture exercises the complete audit → selection →
  native-image package → Drive-like storage → VM staging → pixel-integrity validation path,
  including a complete 123-image bbox manifest and repeatable frozen selections.
- Package corruption and ZIP path escape are rejected. Read-only XAI opening preserves
  the saved experiment record and does not invalidate a run when explainer scripts are added.
- A mounted Drive containing only Notebook_7_beta fails before audit/selection. Remount
  recovery is tested. Missing indexed PNGs fail before checkpoint creation; all-read-error
  audits stop as incomplete, and legacy incorrect complete flags are not reused.
- A populated old mountpoint is preserved while mounting at a fresh empty path;
  the regression fixture reproduces Colab's nonempty-mountpoint rejection.
- The v3 ZIP loader is reused byte for byte (SHA-256
  `5870c0c32c79d62c2170ee9431cc435f586097540ac2b195ba84532ed60a9265`).
  Tests cover copy/extract, archive and pixel hashes, verified resume, local pixel repair,
  corrupt transfer rejection and unsafe member rejection. A synthetic ZIP → local audit
  → selection → selected package round trip preserves all five manifest hashes.
- Subset audit completion is tested with 135 ZIP images and 136 metadata images:
  complete=true for the requested subset, full_nih_complete=false. No original metadata
  rows or official lists are rewritten. Logging emits immediately, heartbeats during
  blocked stages and restores observers on errors without changing engine source bytes.
- Read the real G: v3 catalog: 70,012 unique image IDs cover all 70,012 frozen
  partition/XAI IDs; all five v3 manifest hashes match. Unpacked size is 26.26 GiB.
  This catalog check did not copy or scan the real dataset locally.

## GPU integration passed

Device: **NVIDIA GeForce RTX 4060**, PyTorch 2.13.0+cu130.

- One synthetic DAE epoch and four CNN epochs for **each of 12 conditions**.
- Actual head optimizer updates in epochs 1–3 and layer4+head updates in epoch 4.
- All **16 fixed corruption states** evaluated for each model; threshold selected on
  validation only, classification/patient CIs/paired deltas/image quality and figures generated.
- Independently reload Baseline, DWT five-view and DFT three-view checkpoints; probability
  predictions match saved evaluation values within rtol 1e-5 / atol 1e-6.
- Finite, nonzero input gradients for all three input layouts; output `(N,1)`.
- Reopen and run the complete pipeline again: all completed checkpoint/history/prediction/
  report bytes retain identical SHA-256 hashes.
- TF32 caused a measured batch1/batch4 probability discrepancy during development.
  It is now disabled and recorded in the experiment numeric policy; the final GPU checks passed.

Machine-readable proof: `validation/gpu_smoke.json` (copied from the ignored smoke runtime).
Full checkpoints and plots remain under `.runtime/gpu_smoke_v2/`, excluded from source packages.
This GPU proof predates the Drive I/O and v3 ZIP changes; those paths are covered by
the focused tests above. CNN/DAE training logic is unchanged.

## Bounds of this evidence

The GPU run used synthetic data, random CNN initialization, four images per split,
64x64 working/CNN tensors, a small DAE and 20 bootstrap draws. It proves execution,
provenance and resume behavior, **not** classification accuracy or 256/224px memory capacity.

No real 70,012-image ZIP audit or full ImageNet class160 training has been completed
on Colab in this validation session. The user requested those stages on Colab/Drive;
the notebook performs them when run. No Grad-CAM/SHAP was computed.

The supplied Drive ZIP catalog was confirmed accessible through the user's G: mount,
whose Windows volume label identifies faceyuslay@gmail.com. An earlier Colab mount did
not expose data; the user subsequently reported the old notebook could run. The new
ZIP workflow has not yet been observed running on the user's Colab runtime.
Google Drive mount authorization and Colab GPU runtime selection remain interactive steps.
