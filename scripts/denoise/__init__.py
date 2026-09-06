"""Denoising pipeline — Section 3 of guideline.md.

4 groups, each exposing denoise(img, level, **kwargs) -> np.uint8 grayscale image:
  - baseline    : no-op
  - median      : Median Filter (kernel size: 3, 5, 7)
  - clahe_dwt   : CLAHE + DWT fusion (clip limit: 2.0, 4.0, 8.0; threshold scale: 0.5, 1.0, 2.0)
  - dae_clahe   : DAE + CLAHE fusion (DAE architecture implemented; untrained in this phase)

Contract:
  Every method strictly returns a 2D np.ndarray, dtype np.uint8, values in [0, 255].
"""
from typing import Any
import numpy as np

from . import baseline, median, clahe_dwt, dae_clahe
from .dae_model import DAE

REGISTRY = {
    "baseline": baseline,
    "median": median,
    "clahe_dwt": clahe_dwt,
    "dae_clahe": dae_clahe,
}


def apply_denoise(
    img: np.ndarray,
    method: str = "baseline",
    level: int = 1,
    **kwargs: Any
) -> np.ndarray:
    """Convenience dispatcher to apply any registered denoising method.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    method : str
        One of 'baseline', 'median', 'clahe_dwt', 'dae_clahe'.
    level : int
        Strength level in {1, 2, 3}.
    **kwargs
        Additional method-specific parameter overrides.

    Returns
    -------
    np.ndarray
        Denoised 2D uint8 grayscale image strictly in [0, 255].
    """
    if method not in REGISTRY:
        raise ValueError(
            f"Unknown denoising method '{method}'. Valid options are: {list(REGISTRY.keys())}"
        )
    out = REGISTRY[method].denoise(img, level=level, **kwargs)

    # Validate global contract
    if not isinstance(out, np.ndarray):
        raise TypeError(f"Denoising method '{method}' returned {type(out)}, expected np.ndarray")
    if out.dtype != np.uint8:
        raise TypeError(f"Denoising method '{method}' violated contract: expected dtype uint8, got {out.dtype}")
    if out.ndim != 2:
        raise ValueError(f"Denoising method '{method}' violated contract: expected 2D grayscale, got shape {out.shape}")

    return out


__all__ = ["baseline", "median", "clahe_dwt", "dae_clahe", "DAE", "REGISTRY", "apply_denoise"]
