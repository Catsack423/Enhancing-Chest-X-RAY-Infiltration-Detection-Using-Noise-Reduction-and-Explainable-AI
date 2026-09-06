"""Fusion 2 — DAE + CLAHE.

Reference: Thamilarasi et al. (2025)
"Enhanced ensemble segmentation of lung chest X-ray images by denoising autoencoder and CLAHE."

Pipeline:
  Input Image -> DAE (Denoising Autoencoder) -> CLAHE (Contrast Enhancement)

Strength (Section 3.1):
  - clip_limit: CLAHE clip limit (2.0, 4.0, 8.0)
  - Trained DAE acts as deep feature-preserving denoising filter.
  - In Phase 2, DAE weights are untrained; the architecture runs in zero-residual
    identity mode, smoothly preparing for Phase 3/dedicated DAE training.

Contract:
  All outputs are guaranteed to be 2D np.ndarray, dtype np.uint8, values in [0, 255].
"""
from typing import Optional, Any, Dict
import os
import cv2
import numpy as np
import torch

from .dae_model import DAE, denoise_image

LEVELS: Dict[int, Dict[str, Any]] = {
    1: {"clip_limit": 2.0},
    2: {"clip_limit": 4.0},
    3: {"clip_limit": 8.0},
}

# Cached singleton model instance for inference efficiency
_CACHED_MODEL: Optional[DAE] = None
_CACHED_MODEL_IS_UNTRAINED: bool = True
_WARNED_UNTRAINED: bool = False


def _enforce_uint8_contract(arr: np.ndarray) -> np.ndarray:
    """Strict contract: All denoising functions MUST return uint8 grayscale [0, 255]."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray output, got {type(arr)}")
    if arr.dtype != np.uint8:
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def get_default_dae(checkpoint_path: Optional[str] = None) -> DAE:
    """Retrieve or initialize default DAE model instance."""
    global _CACHED_MODEL, _CACHED_MODEL_IS_UNTRAINED, _WARNED_UNTRAINED
    if _CACHED_MODEL is None:
        model = DAE(in_channels=1, base_channels=32, residual=True)
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.load_checkpoint(checkpoint_path, device=device)
            _CACHED_MODEL_IS_UNTRAINED = False
            print(f"[DAE+CLAHE] Loaded trained DAE checkpoint from: {checkpoint_path}")
        else:
            _CACHED_MODEL_IS_UNTRAINED = True
            if not _WARNED_UNTRAINED:
                print("[DAE+CLAHE] NOTICE: DAE operating in UNTRAINED residual mode "
                      "(zero-residual identity pass pending Phase 3/dedicated training). "
                      "Output reflects CLAHE enhancement with identity DAE pass.")
                _WARNED_UNTRAINED = True
        _CACHED_MODEL = model
    return _CACHED_MODEL


def is_dae_untrained(checkpoint_path: Optional[str] = None, model: Optional[DAE] = None) -> bool:
    """Check whether DAE is operating in untrained identity mode."""
    global _CACHED_MODEL_IS_UNTRAINED
    if model is not None:
        return getattr(model, "is_untrained", False)
    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        return False
    return _CACHED_MODEL_IS_UNTRAINED


def denoise(
    img: np.ndarray,
    level: int = 1,
    clip_limit: Optional[float] = None,
    tile_grid_size: tuple = (8, 8),
    checkpoint_path: Optional[str] = None,
    model: Optional[DAE] = None,
    **kwargs: Any
) -> np.ndarray:
    """Apply DAE denoising followed by CLAHE contrast enhancement.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (uint8).
    level : int, optional
        Predefined strength level {1, 2, 3}. Default is 1.
    clip_limit : float, optional
        Override CLAHE clip limit (e.g. 2.0, 4.0, 8.0).
    tile_grid_size : tuple, optional
        CLAHE grid size (default (8, 8)).
    checkpoint_path : str, optional
        Path to trained DAE weights checkpoint.
    model : DAE, optional
        Pre-instantiated DAE model. If None, default model is used.

    Returns
    -------
    np.ndarray
        Denoised and enhanced uint8 grayscale image strictly in [0, 255].
    """
    img_u8 = _enforce_uint8_contract(img)

    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level}")

    params = LEVELS[level]
    eff_clip_limit = float(clip_limit if clip_limit is not None else params["clip_limit"])

    # 1. DAE Stage
    dae_net = model if model is not None else get_default_dae(checkpoint_path)
    dae_out = denoise_image(dae_net, img_u8)
    dae_out_u8 = _enforce_uint8_contract(dae_out)

    # 2. CLAHE Stage
    clahe = cv2.createCLAHE(clipLimit=eff_clip_limit, tileGridSize=tile_grid_size)
    final_out = clahe.apply(dae_out_u8)

    return _enforce_uint8_contract(final_out)
