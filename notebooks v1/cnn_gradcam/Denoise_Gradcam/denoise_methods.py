"""Denoising methods matching the CNN research pipeline:
- Baseline (Raw)
- Median Filter (L1=3x3, L2=5x5, L3=7x7)
- CLAHE + DWT (L1, L2, L3)
"""
from typing import Dict, Any, Union
import cv2
import numpy as np
from PIL import Image
import pywt

DEFAULT_WAVELET = "db1"

CLAHE_DWT_LEVELS: Dict[int, Dict[str, Any]] = {
    1: {"clip_limit": 2.0, "dwt_level": 1, "threshold_scale": 0.5},
    2: {"clip_limit": 4.0, "dwt_level": 2, "threshold_scale": 1.0},
    3: {"clip_limit": 8.0, "dwt_level": 3, "threshold_scale": 2.0},
}

MEDIAN_LEVELS: Dict[int, int] = {
    1: 3,  # 3x3
    2: 5,  # 5x5
    3: 7,  # 7x7
}

def to_uint8_gray(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img.convert('L'))
    elif isinstance(img, np.ndarray):
        if img.ndim == 3:
            arr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            arr = img
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
        
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr

def apply_baseline(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Baseline: Raw image, no denoising applied."""
    return to_uint8_gray(img)

def apply_median(img: Union[np.ndarray, Image.Image], level: int = 1) -> np.ndarray:
    """Traditional Median Filter."""
    arr = to_uint8_gray(img)
    ksize = MEDIAN_LEVELS.get(level, 3)
    return cv2.medianBlur(arr, ksize)

def _soft_threshold(coeff: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(coeff) * np.maximum(np.abs(coeff) - thresh, 0.0)

def _dwt_denoise(img_f: np.ndarray, dwt_level: int, threshold_scale: float, wavelet: str = DEFAULT_WAVELET) -> np.ndarray:
    max_level = pywt.dwt_max_level(data_len=min(img_f.shape), filter_len=pywt.Wavelet(wavelet).dec_len)
    actual_level = max(1, min(dwt_level, max_level))

    coeffs = pywt.wavedec2(img_f, wavelet, level=actual_level)
    cA = coeffs[0]
    detail_levels = coeffs[1:]

    if not detail_levels:
        return cA

    finest_cD = detail_levels[-1][2]
    sigma = float(np.median(np.abs(finest_cD)) / 0.6745)
    n = img_f.size
    thresh = sigma * np.sqrt(2.0 * np.log(max(n, 2))) * threshold_scale

    denoised_details = [
        tuple(_soft_threshold(sub, thresh) for sub in (cH, cV, cD))
        for (cH, cV, cD) in detail_levels
    ]
    return pywt.waverec2([cA] + denoised_details, wavelet)

def apply_clahe_dwt(
    img: Union[np.ndarray, Image.Image],
    level: int = 1,
    wavelet: str = DEFAULT_WAVELET,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """Fusion Denoising: DWT Decomposition + VisuShrink Soft-thresholding + CLAHE."""
    arr = to_uint8_gray(img)
    params = CLAHE_DWT_LEVELS.get(level, CLAHE_DWT_LEVELS[1])
    
    clip_limit = float(params["clip_limit"])
    dwt_level = int(params["dwt_level"])
    threshold_scale = float(params["threshold_scale"])

    img_f = arr.astype(np.float64)
    denoised = _dwt_denoise(img_f, dwt_level, threshold_scale, wavelet=wavelet)
    denoised = denoised[: arr.shape[0], : arr.shape[1]]
    denoised_u8 = np.clip(np.round(denoised), 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(denoised_u8)

ALL_METHODS = {
    "Baseline": lambda img: apply_baseline(img),
    "Median L1 (3x3)": lambda img: apply_median(img, level=1),
    "Median L2 (5x5)": lambda img: apply_median(img, level=2),
    "Median L3 (7x7)": lambda img: apply_median(img, level=3),
    "CLAHE+DWT L1": lambda img: apply_clahe_dwt(img, level=1),
    "CLAHE+DWT L2": lambda img: apply_clahe_dwt(img, level=2),
    "CLAHE+DWT L3": lambda img: apply_clahe_dwt(img, level=3),
}
