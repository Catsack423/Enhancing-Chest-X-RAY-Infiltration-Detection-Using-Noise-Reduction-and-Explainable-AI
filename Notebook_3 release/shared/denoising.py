"""Controlled release strengths and explicit v2 legacy compatibility."""
import cv2
import numpy as np
import pywt
import torch

from .dae_model import DAE
from .experiment import file_hash


def read_gray(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode selected image: {path}")
    return image


def working_image(raw, config):
    size = config["preprocessing"]["working_size"]
    return cv2.resize(raw, (size, size), interpolation=cv2.INTER_AREA)


def dwt_max_ratio_denoise(image, condition):
    """Seminae DWT L4: one soft cutoff = ratio * largest detail coefficient magnitude."""
    wavelet = condition.get("wavelet", "db1")
    mode = condition.get("mode", "periodization")
    depth = condition.get("dwt_depth", 1)
    if pywt.dwtn_max_level(image.shape, wavelet) < depth:
        raise ValueError("Insufficient working size for DWT depth")
    ratio = condition["threshold_ratio"]
    if not 0 <= ratio <= 1:
        raise ValueError("DWT threshold ratio must be in [0,1]")
    coeffs = pywt.wavedec2(image.astype(np.float64), wavelet, mode=mode, level=depth)
    cutoff = ratio * max(np.abs(c).max() for band in coeffs[1:] for c in band)
    details = [tuple(pywt.threshold(c, cutoff, mode="soft") if cutoff > 0 else c for c in band) for band in coeffs[1:]]
    result = pywt.waverec2([coeffs[0]] + details, wavelet, mode=mode)[:image.shape[0], :image.shape[1]]
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def dwt_denoise(image, condition):
    if condition.get("threshold_policy") == "max_detail_ratio":
        return dwt_max_ratio_denoise(image, condition)
    legacy = condition.get("profile") == "legacy_v2" or condition.get("threshold_scope") == "image"
    wavelet = condition.get("wavelet", "db1")
    mode = condition.get("mode", "symmetric" if legacy else "periodization")
    depth = condition.get("dwt_depth", condition.get("depth", 3))
    if pywt.dwtn_max_level(image.shape, wavelet) < depth:
        raise ValueError("Insufficient working size for DWT depth")
    coeffs = pywt.wavedec2(image.astype(np.float64), wavelet, mode=mode, level=depth)
    sigma = np.median(np.abs(coeffs[-1][2])) / .6745
    multiplier = condition["threshold_multiplier"]
    if multiplier < 0:
        raise ValueError("Negative threshold multiplier")
    def threshold(coefficient):
        n = image.size if legacy else coefficient.size
        cutoff = multiplier * sigma * np.sqrt(2*np.log(max(n, 1)))
        return pywt.threshold(coefficient, cutoff, mode="soft") if cutoff > 0 else coefficient
    details = [tuple(threshold(c) for c in band) for band in coeffs[1:]]
    result = pywt.waverec2([coeffs[0]] + details, wavelet, mode=mode)[:image.shape[0], :image.shape[1]]
    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def dae_blend(image, dae, alpha, device="cpu", passes=1):
    if dae is None:
        raise RuntimeError("DAE requires the verified project-trained checkpoint")
    if not 0 <= alpha <= 1:
        raise ValueError("DAE blend alpha must be in [0,1]")
    if int(passes) != passes or passes < 1:
        raise ValueError("DAE passes must be a positive integer")
    x = image.astype(np.float32)/255
    tensor = torch.from_numpy(x)[None, None].to(device)
    with torch.no_grad():
        # passes>1 feeds the DAE output back in (L4: stronger than alpha=1).
        for _ in range(int(passes)):
            tensor = dae(tensor)
        denoised = tensor[0, 0].cpu().numpy()
    if denoised.shape != x.shape or not np.isfinite(denoised).all():
        raise ValueError("Invalid DAE output")
    return np.clip(np.round(((1-alpha)*x+alpha*denoised)*255), 0, 255).astype(np.uint8)


def apply_condition(image, condition, config, dae=None, device="cpu"):
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Expected grayscale uint8 denoising input")
    condition = dict(condition)
    condition.setdefault("profile", config.get("profile", "release_controlled"))
    group = condition["group"]
    if group == "baseline":
        return image.copy()
    if group == "median":
        return cv2.medianBlur(image, int(condition["kernel"]))
    if group == "clahe_dwt":
        if condition.get("order", "DWT_then_CLAHE") != "DWT_then_CLAHE":
            raise ValueError("Unexpected DWT/CLAHE order")
        result = dwt_denoise(image, condition)
    elif group == "dae_clahe":
        result = dae_blend(image, dae, condition.get("blend_alpha", condition.get("alpha", 1.0)), device,
                           condition.get("dae_passes", 1))
    else:
        raise ValueError(f"Unknown denoising group {group}")
    return cv2.createCLAHE(clipLimit=float(condition.get("clip_limit", 2.0)),
                           tileGridSize=tuple(config["preprocessing"].get("clahe_grid", [8, 8]))).apply(result)


def cnn_tensor(image, config):
    prep = config["preprocessing"]
    size = prep["cnn_size"]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.repeat(resized[None], 3, axis=0).astype(np.float32)/255)
    return (tensor-torch.tensor(prep["mean"]).view(3, 1, 1))/torch.tensor(prep["std"]).view(3, 1, 1)


def load_dae(exp, device):
    import json
    folder = exp.artifacts / "dae_checkpoint"
    marker = folder / "complete.json"
    if not marker.is_file():
        raise RuntimeError("DAE training is incomplete; finish the configured epoch budget before preprocessing")
    completed = json.loads(marker.read_text(encoding="utf-8"))
    exp.check_provenance(completed["provenance"])
    if completed.get("epochs_run") != exp.config["dae"]["epochs"] or file_hash(folder / "best.pt") != completed.get("checkpoint_hash"):
        raise ValueError("DAE completion/checkpoint fingerprint is invalid")
    checkpoint = exp.load_checkpoint(folder / "best.pt")
    if checkpoint["epoch"] < 1:
        raise ValueError("Refusing untrained DAE checkpoint")
    model = DAE(base_channels=exp.config["dae"]["base_channels"], residual=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def synthetic_noise(clean, config, rng):
    noisy = clean+rng.normal(0, config["gaussian_sigma"], clean.shape).astype(np.float32)
    poisson = rng.poisson(np.maximum(noisy*config["poisson_scale"], 0)).astype(np.float32)/config["poisson_scale"]
    return np.clip(.5*noisy+.5*poisson, 0, 1).astype(np.float32)
