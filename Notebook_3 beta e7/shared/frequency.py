"""Spatial frequency views, shared-weight feature fusion, and locked existing IDs.

No CLAHE, no new split, no fitted preprocessing, no new backbone parameters.
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
import pandas as pd
import pywt
import torch

from .data import verify_manifests
from .denoising import Preprocessor, read_gray, working_image
from .evaluation import positive_normalize, resize_map, patient_bootstrap
from .experiment import Experiment, digest, file_hash, write_json
from .models import BinaryResNet50


CONDITIONS = [
    {"id": "baseline", "group": "baseline", "level": 0},
    {"id": "dwt_L1_L4_fusion", "group": "dwt_features", "level": 4,
     "wavelet": "db1", "mode": "periodization", "depth": 4},
    {"id": "dft_low_mid_high_fusion", "group": "dft_features", "level": 0,
     "low_cutoff": 0.125, "high_cutoff": 0.25,
     "frequency_units": "cycles_per_pixel_radial"},
]
MANIFEST_NAMES = ("train", "validation", "test", "xai", "background")


def open_frequency_experiment(root, data_root, source_run, output_root=None):
    """Copy the exact recorded manifests; never call prepare_manifests/sample."""
    root, source_run = Path(root).resolve(), Path(source_run).resolve()
    source_artifacts = source_run / "shared/artifacts"
    saved = json.loads((source_artifacts / "experiment.json").read_text(encoding="utf-8"))
    source_config = saved["config"]
    if digest(source_config) != saved["config_hash"]:
        raise ValueError("Original experiment configuration is corrupted")
    if source_config["run_mode"] != "full":
        raise ValueError("Select the original full run, not smoke manifests")
    source_manifests = source_artifacts / "manifests"
    source_signature = json.loads((source_manifests / "source.json").read_text(encoding="utf-8"))
    if source_signature["config_hash"] != saved["config_hash"]:
        raise ValueError("Original manifest configuration differs from original run")
    data_root = Path(data_root).resolve()
    for name, expected in source_signature["sources"].items():
        if file_hash(data_root / name) != expected:
            raise ValueError(f"NIH source changed: {name}")
    hashes = {f"{name}.csv": file_hash(source_manifests / f"{name}.csv") for name in MANIFEST_NAMES}
    if hashes != saved["manifest_hashes"]:
        raise ValueError("Original manifests changed; refusing to select replacement IDs")
    config = deepcopy(source_config)
    config["conditions"] = deepcopy(CONDITIONS)
    config["output_subdirectory"] = "runs/head_only_e7"
    config["frequency_fusion"] = {
        "version": 1, "fusion": "mean_2048_features_before_same_linear_head",
        "dwt_views": ["A4", "D1", "D2", "D3", "D4"],
        "dft_views": ["low", "mid", "high"],
        "signed_detail_encoding": "0.5 + 0.5 * component; no clip, no per-image scaling",
        "source_config_hash": saved["config_hash"], "source_manifest_hashes": hashes,
        "source_code_hashes": {p.name: file_hash(p) for p in sorted((root / "shared").glob("*.py"))},
    }
    from .beta import configure_head_only
    config = configure_head_only(config, root)
    output_root = Path(output_root or root / config["output_subdirectory"]).resolve()
    if output_root == source_run or output_root.is_relative_to(source_run):
        raise ValueError("Frequency outputs must be separate from the source run")
    exp = Experiment(root, data_root, output_root, config)
    frames = {name: pd.read_csv(source_manifests / f"{name}.csv") for name in MANIFEST_NAMES}
    verify_manifests(exp, frames)
    destination = exp.artifacts / "manifests"
    if (exp.artifacts / "experiment.json").exists():
        exp.lock()  # reject changed settings/code before any writes
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in hashes.items():
        target = destination / name
        if target.exists() and file_hash(target) != expected:
            raise ValueError(f"Existing frequency manifest differs: {name}")
        if not target.exists():
            shutil.copyfile(source_manifests / name, target)
    write_json(destination / "source.json", {
        "config_hash": digest(config), "sources": source_signature["sources"],
        "inherited_manifest_hashes": hashes, "source_config_hash": saved["config_hash"],
    })
    exp.lock()
    return exp, frames


def dwt_components(image, wavelet="db1", depth=4, mode="periodization"):
    """A4 and D1..D4, reconstructed separately in original working coordinates.

    Dj contains all three directional details at level j. Zeroing every other
    coefficient before inverse DWT preserves spatial alignment for localization.
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite grayscale image")
    if depth != 4 or pywt.dwtn_max_level(x.shape, wavelet) < depth:
        raise ValueError("DWT requires four valid levels")
    coeffs = pywt.wavedec2(x, wavelet, mode=mode, level=depth)
    def reconstruct(keep):
        selected = [coeffs[0] if keep == 0 else np.zeros_like(coeffs[0])]
        selected.extend(tuple(c if i == keep else np.zeros_like(c) for c in band)
                        for i, band in enumerate(coeffs[1:], 1))
        return pywt.waverec2(selected, wavelet, mode=mode)[:x.shape[0], :x.shape[1]]
    # wavedec2 orders details D4,D3,D2,D1; expose ascending levels explicitly.
    return np.stack([reconstruct(0)] + [reconstruct(depth - j + 1) for j in range(1, depth + 1)])


def dft_components(image, low_cutoff=0.125, high_cutoff=0.25):
    """Complementary radial masks; inverse FFT returns spatial signed images.

    Frequencies use cycles/pixel: axial Nyquist=.5, corner radius=sqrt(.5).
    Boundaries belong to the higher band. No magnitude spectrum is fed to CNN.
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite grayscale image")
    if not 0 < low_cutoff < high_cutoff < 0.5:
        raise ValueError("Require 0 < low_cutoff < high_cutoff < .5 cycles/pixel")
    fy, fx = np.fft.fftfreq(x.shape[0]), np.fft.fftfreq(x.shape[1])
    radius = np.hypot(fy[:, None], fx[None, :])
    masks = [radius < low_cutoff, (radius >= low_cutoff) & (radius < high_cutoff),
             radius >= high_cutoff]
    spectrum = np.fft.fft2(x)
    return np.stack([np.fft.ifft2(spectrum * mask).real for mask in masks]).astype(np.float32)


class FrequencyPreprocessor(Preprocessor):
    def __init__(self, exp, condition_id, device="cpu"):
        super().__init__(exp, condition_id, device)
        self._tensors = {}

    def components(self, row):
        image = working_image(read_gray(self.exp.data_root / row["relative_path"]), self.exp.config)
        x = image.astype(np.float32) / 255.0
        condition = self.condition
        if condition["group"] == "dwt_features":
            return dwt_components(x, condition["wavelet"], condition["depth"], condition["mode"])
        if condition["group"] == "dft_features":
            return dft_components(x, condition["low_cutoff"], condition["high_cutoff"])
        return x[None]

    def tensor(self, row):
        # Baseline executes the original uint8 resize/normalization exactly.
        if self.condition["group"] == "baseline":
            return super().tensor(row).unsqueeze(0)
        if row["image"] not in self._tensors:
            bands = self.components(row)
            bands[1:] = 0.5 + 0.5 * bands[1:]
            size = self.exp.config["preprocessing"]["cnn_size"]
            resized = np.stack([cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR) for b in bands])
            tensor = torch.from_numpy(np.repeat(resized[:, None], 3, axis=1).astype(np.float32))
            settings = self.exp.config["preprocessing"]
            mean = torch.tensor(settings["mean"]).view(1, 3, 1, 1)
            std = torch.tensor(settings["std"]).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite frequency input")
            # Bound RAM; DWT can otherwise retain >1.5 GB for this small dataset.
            if len(self._tensors) >= 32:
                self._tensors.pop(next(iter(self._tensors)))
            self._tensors[row["image"]] = tensor
        return self._tensors[row["image"]].clone()

    def prepare(self, frames):
        # Transform on demand; do not write frequency images through uint8 cache.
        if self.condition["group"] == "baseline":
            return super().prepare(frames)
        print(f"{self.condition['id']}: spatial frequency views computed on demand", flush=True)


class FeatureFusionResNet50(BinaryResNet50):
    """Same pretrained backbone, trainable layers, head and parameter count.

    For linear head W,b: W mean(f_v)+b == mean(W f_v+b).
    Backbone.forward supplies those per-view logits, so this computes feature
    averaging exactly (not probability averaging) without replacing its head.
    """
    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != 3:
            raise ValueError("Expected (batch, views, 3, height, width)")
        n, views, channels, height, width = x.shape
        logits = self.backbone(x.reshape(n * views, channels, height, width))
        return logits.reshape(n, views, 1).mean(dim=1)


def frequency_gradcam(model, input_tensor, original_shape):
    model.eval()
    captured = []
    handle = model.target_layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        if input_tensor.shape[0] != 1:
            raise ValueError("Explain one source image at a time")
        score = model(input_tensor.detach().requires_grad_(True))[0, 0]
        features = captured[0]  # V,C,h,w, spatially aligned across views
        gradients, = torch.autograd.grad(score, features)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        # Gradients already include 1/V from feature averaging. Sum signed
        # contributions first, then apply one ReLU to the fused explanation.
        native = (weights * features).sum(dim=(0, 1)).relu().detach().cpu().numpy()
        return positive_normalize(resize_map(native, original_shape)), float(score.detach().sigmoid())
    finally:
        handle.remove()


def frequency_shap_maps(explainer, input_tensor, original_shape, config):
    values = explainer.shap_values(input_tensor, nsamples=config["xai"]["shap_nsamples"], rseed=config["seed"])
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected one Infiltration output")
        values = values[0]
    values = np.asarray(values)
    expected = tuple(input_tensor.shape)
    if values.shape == expected + (1,):
        values = values[..., 0]
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Unexpected multi-view SHAP values: {values.shape}")
    channels = values[0].astype(np.float32)
    signed_input = channels.sum(axis=(0, 1))  # views and RGB, before positive-only map
    signed_native = resize_map(signed_input, original_shape)
    return {"view_channel_attributions": channels, "signed_input": signed_input,
            "signed_native": signed_native, "positive_heatmap": positive_normalize(signed_native)}


def train_frequency(exp, manifests, condition_id, device="cuda"):
    from .training import train_condition
    return train_condition(exp, manifests, condition_id, device,
                           model_factory=FeatureFusionResNet50, preprocessor_factory=FrequencyPreprocessor)


def evaluate_frequency(exp, manifests, criterion, condition_id, device="cuda"):
    from .xai import evaluate_condition
    return evaluate_condition(exp, manifests, criterion, condition_id, device,
                              model_factory=FeatureFusionResNet50, preprocessor_factory=FrequencyPreprocessor,
                              gradcam_fn=frequency_gradcam, shap_maps_fn=frequency_shap_maps)


def frequency_reports(exp):
    """Compact measured tables, paired XAI differences, no fabricated scores."""
    destination = exp.output_root / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    rows, summaries, differences, statuses = [], [], [], []
    for condition in exp.condition_ids:
        status = {"condition_id": condition}
        for criterion in ("CNN", "Grad-CAM", "SHAP"):
            folder = exp.results(criterion, condition)
            complete = folder / "complete.json"
            status[criterion] = "not_run"
            if not complete.exists():
                continue
            marker = json.loads(complete.read_text())
            exp.check_provenance(marker["provenance"], condition)
            key = "checkpoint_hash" if criterion == "CNN" else "cnn_checkpoint_hash"
            if marker[key] != file_hash(exp.results("CNN", condition) / "best.pt"):
                raise ValueError("Result refers to a different checkpoint")
            status[criterion] = "complete"
            if criterion == "CNN":
                measured = json.loads((folder / "metrics.json").read_text())
                rows.append({k: v for k, v in measured.items() if k not in ("provenance", "confusion_matrix")})
            else:
                summary = pd.read_csv(folder / "summary.csv")
                summary.insert(0, "explainer", criterion)
                summaries.append(summary)
        statuses.append(status)
    cnn = pd.DataFrame(rows)
    if not cnn.empty and "baseline" in set(cnn.condition_id):
        baseline = cnn.set_index("condition_id").loc["baseline"]
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            cnn[f"delta_{metric}_vs_baseline"] = cnn[metric] - baseline[metric]
    cnn.to_csv(destination / "classification.csv", index=False)
    xai = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    xai.to_csv(destination / "xai.csv", index=False)
    for criterion in ("Grad-CAM", "SHAP"):
        baseline_path = exp.results(criterion, "baseline") / "per_image_metrics.csv"
        if not baseline_path.exists():
            continue
        base = pd.read_csv(baseline_path)
        for condition in exp.condition_ids[1:]:
            path = exp.results(criterion, condition) / "per_image_metrics.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            if set(frame.image) != set(base.image):
                raise ValueError("Paired XAI comparison requires identical image IDs")
            paired = frame.merge(base, on=["image", "patient_id", "risk_flag"], suffixes=("", "_base"), validate="one_to_one")
            if len(paired) != len(base):
                raise ValueError("Paired XAI metadata mismatch")
            for stratum in ("overall", "pure", "mixed_high_risk", "mixed_low_risk"):
                subset = paired.copy() if stratum == "overall" else paired[paired.risk_flag == stratum].copy()
                if subset.empty:
                    continue
                for metric in ["pointing_hit"] + [f"iou_{t:g}" for t in exp.config["xai"]["iou_thresholds"]]:
                    subset["delta"] = subset[metric] - subset[metric + "_base"]
                    low, high = patient_bootstrap(subset, "delta", exp.config["seed"], exp.config["xai"]["bootstrap_replicates"])
                    differences.append({"explainer": criterion, "condition_id": condition, "stratum": stratum,
                                        "metric": metric, "n_images": len(subset), "mean_delta": subset.delta.mean(),
                                        "ci_low": low, "ci_high": high})
    deltas = pd.DataFrame(differences)
    deltas.to_csv(destination / "paired_xai_deltas.csv", index=False)
    status = pd.DataFrame(statuses)
    status.to_csv(destination / "status.csv", index=False)
    return cnn, xai, deltas, status
