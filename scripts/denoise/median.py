"""Traditional single filter — Median Filter. Chattopadhyay (2022): best traditional
filter for X-ray Poisson noise, PSNR 34.52 dB. Strength = kernel size."""
from typing import Optional, Any
import cv2
import numpy as np

# Section 3.1: kernel size controls strength. Must be odd positive integer.
LEVELS = {1: 3, 2: 5, 3: 7}


def _enforce_uint8_contract(arr: np.ndarray) -> np.ndarray:
    """Strict contract: All denoising functions MUST return uint8 grayscale [0, 255]."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr)}")
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def denoise(
    img: np.ndarray,
    level: int = 1,
    ksize: Optional[int] = None,
    **kwargs: Any
) -> np.ndarray:
    """Apply median filter.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    level : int, optional
        Predefined strength level {1: kernel 3, 2: kernel 5, 3: kernel 7}. Default is 1.
    ksize : int, optional
        Explicit kernel size override (must be odd positive integer). If None, inferred from `level`.

    Returns
    -------
    np.ndarray
        Denoised uint8 grayscale image strictly in [0, 255].
    """
    img_u8 = _enforce_uint8_contract(img)

    if ksize is None:
        if level not in LEVELS:
            raise ValueError(f"level must be one of {list(LEVELS)}, got {level}")
        ksize = LEVELS[level]

    if not isinstance(ksize, int) or ksize <= 0 or ksize % 2 == 0:
        raise ValueError(f"ksize must be an odd positive integer, got {ksize}")

    out = cv2.medianBlur(img_u8, ksize)
    return _enforce_uint8_contract(out)
