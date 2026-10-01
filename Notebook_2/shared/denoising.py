"""One preprocessing path shared by CNN, Grad-CAM, and SHAP."""
from pathlib import Path

import cv2
import numpy as np
import pywt
import torch

from .dae_model import DAE, denoise_image
from .experiment import digest, file_hash


def read_gray(path):
    # imdecode supports Unicode paths on Windows too.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def working_image(raw, config):
    size = config["preprocessing"]["working_size"]
    return cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)


def apply_condition(image, condition, config, dae=None, device="cpu"):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Denoising input must be grayscale uint8")
    group = condition["group"]
    if group == "baseline":
        return image.copy()
    if group == "median":
        return cv2.medianBlur(image, condition["kernel"])
    if group == "clahe_dwt":
        if condition["order"] != "DWT_then_CLAHE":
            raise ValueError("Unexpected DWT/CLAHE order")
        coeffs = pywt.wavedec2(image.astype(np.float64), condition["wavelet"], level=condition["dwt_depth"])
        sigma = np.median(np.abs(coeffs[-1][2])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(image.size)) * condition["threshold_multiplier"]
        details = [tuple(pywt.threshold(c, threshold, mode="soft") if threshold > 0 else c for c in band) for band in coeffs[1:]]
        denoised = pywt.waverec2([coeffs[0]] + details, condition["wavelet"])
        output = np.clip(np.round(denoised[:image.shape[0], :image.shape[1]]), 0, 255).astype(np.uint8)
    elif group == "dae_clahe":
        if dae is None:
            raise RuntimeError("DAE+CLAHE requires the verified project-trained DAE checkpoint")
        output = denoise_image(dae, image, torch.device(device))
    else:
        raise ValueError(f"Unknown condition group: {group}")
    grid = tuple(config["preprocessing"]["clahe_grid"])
    return cv2.createCLAHE(clipLimit=condition["clip_limit"], tileGridSize=grid).apply(output)


def cnn_tensor(image, config):
    prep = config["preprocessing"]
    size = prep["cnn_size"]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.repeat(resized[None], 3, axis=0).astype(np.float32) / 255.0)
    mean = torch.tensor(prep["mean"]).view(3, 1, 1)
    std = torch.tensor(prep["std"]).view(3, 1, 1)
    return (tensor - mean) / std


def load_dae(exp, device):
    checkpoint = exp.load_checkpoint(exp.artifacts / "dae_checkpoint/best.pt")
    if checkpoint["epoch"] < 1:
        raise ValueError("Untrained DAE checkpoint cannot be evaluated")
    model = DAE(base_channels=exp.config["dae"]["base_channels"], residual=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


class Preprocessor:
    """Cache only processed images, never silently substitute a model."""
    def __init__(self, exp, condition_id, device="cpu"):
        self.exp = exp
        self.condition = exp.condition(condition_id)
        self.device = torch.device(device)
        self.dae = None
        dependency = None
        if self.condition["group"] == "dae_clahe":
            self.dae = load_dae(exp, self.device)
            dependency = file_hash(exp.artifacts / "dae_checkpoint/best.pt")
        self.cache = exp.artifacts / "cache" / digest({"config": exp.config, "dae": dependency}) / condition_id

    def image(self, row):
        path = self.cache / row["image"]
        if path.exists():
            result = read_gray(path)
        else:
            raw = read_gray(self.exp.data_root / row["relative_path"])
            base = working_image(raw, self.exp.config)
            result = apply_condition(base, self.condition, self.exp.config, self.dae, self.device)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replacement prevents a disconnected runtime leaving a partial PNG.
            temporary = path.with_suffix(".tmp")
            ok, encoded = cv2.imencode(".png", result)
            if not ok:
                raise RuntimeError("PNG encoding failed")
            encoded.tofile(temporary)
            temporary.replace(path)
        size = self.exp.config["preprocessing"]["working_size"]
        if result.shape != (size, size):
            raise ValueError("Cached preprocessing dimensions changed")
        return result

    def tensor(self, row):
        return cnn_tensor(self.image(row), self.exp.config)

    def prepare(self, frames):
        import pandas as pd
        rows = pd.concat(frames).drop_duplicates("image").to_dict("records")
        for i, row in enumerate(rows, 1):
            self.image(row)
            if i % 500 == 0:
                print(f"{self.condition['id']}: prepared {i}/{len(rows)}", flush=True)
        # Cached images can now be loaded on CPU without keeping DAE on the GPU.
        if self.dae is not None:
            self.dae.cpu()
        self.device = torch.device("cpu")


def synthetic_noise(clean, config, rng):
    noisy = clean + rng.normal(0, config["gaussian_sigma"], clean.shape).astype(np.float32)
    poisson = rng.poisson(np.maximum(noisy * config["poisson_scale"], 0)).astype(np.float32) / config["poisson_scale"]
    return np.clip(0.5 * noisy + 0.5 * poisson, 0, 1).astype(np.float32)
