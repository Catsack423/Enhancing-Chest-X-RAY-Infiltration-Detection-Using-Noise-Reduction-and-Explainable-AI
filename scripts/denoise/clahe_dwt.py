"""Fusion 1 — CLAHE + DWT. Chutia et al. (2023/2024): 97.77% accuracy on
Atelectasis/Pneumothorax; testing transfer to Infiltration here.

Pipeline Order:
  Input -> DWT decompose -> soft-threshold detail coefficients (denoise) ->
  reconstruct -> CLAHE (contrast enhancement on the denoised image).

================================================================================
IMPORTANT NOTE ON SECTION 3.1 & EXPERIMENTAL DESIGN (Ablation Intent):
The grouped presets in `LEVELS` (Levels 1, 2, 3) couple `clip_limit`, `dwt_level`,
and `threshold_scale` together strictly for the Phase 2 joint sanity check (to verify
that the pipeline functions across weak/medium/strong parameter extremes).

In Phase 4–7 full experiment matrix and ablation studies, these three parameters
MUST be swept and ablated in isolation (e.g. varying clip_limit while fixing DWT,
or varying threshold_scale while fixing CLAHE) using the keyword overrides
(`clip_limit`, `dwt_level`, `threshold_scale`) provided in `denoise()`, in order to
rigorously determine which specific factor causes over-smoothing vs XAI-IoU degradation.
================================================================================

Contract:
  All outputs are guaranteed to be 2D np.ndarray, dtype np.uint8, values in [0, 255].
"""
from typing import Optional, Any, Dict
import cv2
import numpy as np
import pywt

DEFAULT_WAVELET = "db1"

# Presets for Phase 2 joint sanity check.
# (See docstring above: Phase 4-7 will ablate each parameter independently via kwargs)
LEVELS: Dict[int, Dict[str, Any]] = {
    1: {"clip_limit": 2.0, "dwt_level": 1, "threshold_scale": 0.5},
    2: {"clip_limit": 4.0, "dwt_level": 2, "threshold_scale": 1.0},
    3: {"clip_limit": 8.0, "dwt_level": 3, "threshold_scale": 2.0},
}


def _enforce_uint8_contract(arr: np.ndarray) -> np.ndarray:
    """Strict contract: All denoising functions MUST return uint8 grayscale [0, 255]."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray output, got {type(arr)}")
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def _soft_threshold(coeff: np.ndarray, thresh: float) -> np.ndarray:
    """Apply soft thresholding to wavelet detail coefficients."""
    return np.sign(coeff) * np.maximum(np.abs(coeff) - thresh, 0.0)


def _dwt_denoise(
    img_f: np.ndarray,
    dwt_level: int,
    threshold_scale: float,
    wavelet: str = DEFAULT_WAVELET
) -> np.ndarray:
    """Decompose with DWT, apply VisuShrink soft thresholding on details, reconstruct."""
    max_level = pywt.dwt_max_level(data_len=min(img_f.shape), filter_len=pywt.Wavelet(wavelet).dec_len)
    actual_level = max(1, min(dwt_level, max_level))

    coeffs = pywt.wavedec2(img_f, wavelet, level=actual_level)
    cA = coeffs[0]
    detail_levels = coeffs[1:]

    if not detail_levels:
        return cA

    # VisuShrink universal threshold estimated from finest-level diagonal detail
    finest_cD = detail_levels[-1][2]
    sigma = float(np.median(np.abs(finest_cD)) / 0.6745)
    n = img_f.size
    thresh = sigma * np.sqrt(2.0 * np.log(max(n, 2))) * threshold_scale

    denoised_details = [
        tuple(_soft_threshold(sub, thresh) for sub in (cH, cV, cD))
        for (cH, cV, cD) in detail_levels
    ]
    return pywt.waverec2([cA] + denoised_details, wavelet)


def denoise(
    img: np.ndarray,
    level: int = 1,
    clip_limit: Optional[float] = None,
    dwt_level: Optional[int] = None,
    threshold_scale: Optional[float] = None,
    wavelet: str = DEFAULT_WAVELET,
    tile_grid_size: tuple = (8, 8),
    **kwargs: Any
) -> np.ndarray:
    """Apply DWT denoise followed by CLAHE contrast enhancement.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    level : int, optional
        Predefined strength level {1, 2, 3} for joint sanity sweep. Default is 1.
    clip_limit : float, optional
        Override CLAHE clip limit (e.g. 2.0, 4.0, 8.0) for isolated ablation.
    dwt_level : int, optional
        Override DWT decomposition level (1, 2, 3) for isolated ablation.
    threshold_scale : float, optional
        Override VisuShrink threshold multiplier for isolated ablation.
    wavelet : str, optional
        Wavelet family name (default 'db1').
    tile_grid_size : tuple, optional
        CLAHE grid size (default (8, 8)).

    Returns
    -------
    np.ndarray
        Enhanced and denoised uint8 grayscale image strictly in [0, 255].
    """
    img_u8 = _enforce_uint8_contract(img)

    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level}")

    params = LEVELS[level]
    eff_clip_limit = float(clip_limit if clip_limit is not None else params["clip_limit"])
    eff_dwt_level = int(dwt_level if dwt_level is not None else params["dwt_level"])
    eff_threshold_scale = float(threshold_scale if threshold_scale is not None else params["threshold_scale"])

    img_f = img_u8.astype(np.float64)
    denoised = _dwt_denoise(img_f, eff_dwt_level, eff_threshold_scale, wavelet=wavelet)

    # waverec2 can differ by 1 pixel along edges due to padding; crop back exactly
    denoised = denoised[: img_u8.shape[0], : img_u8.shape[1]]
    denoised_u8 = _enforce_uint8_contract(denoised)

    clahe = cv2.createCLAHE(clipLimit=eff_clip_limit, tileGridSize=tile_grid_size)
    out = clahe.apply(denoised_u8)
    return _enforce_uint8_contract(out)
