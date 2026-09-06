"""Baseline group — raw image, no denoising. Reference line for all comparisons."""
from typing import Any
import numpy as np

LEVELS = {1: None, 2: None, 3: None}  # no strength variable — baseline is fixed


def _enforce_uint8_contract(arr: np.ndarray) -> np.ndarray:
    """Strict contract: All denoising functions MUST return uint8 grayscale [0, 255]."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr)}")
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def denoise(img: np.ndarray, level: int = 1, **kwargs: Any) -> np.ndarray:
    """No-op baseline.

    Guarantees strict uint8 [0, 255] contract across all calls.
    """
    out = _enforce_uint8_contract(img)
    return out.copy()
