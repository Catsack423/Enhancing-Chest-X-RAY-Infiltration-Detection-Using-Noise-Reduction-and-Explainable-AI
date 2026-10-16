"""Unmodified v4.3 helper definitions; full runner replaced by v7."""
from copy import deepcopy

from dataclasses import dataclass

from hashlib import sha256

from importlib import metadata

from pathlib import Path

import json

import os

import random

import numpy as np

import torch

def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def file_hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def versions():
    result = {}
    for name in ("torch", "torchvision", "shap", "numpy", "pandas", "opencv-python", "opencv-python-headless", "PyWavelets", "scikit-learn"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result

def deterministic_backend(enabled=True):
    # Disabling trades bitwise-repeatable GPU arithmetic for faster cuDNN kernels;
    # seeds still fix data order, noise and initial weights.
    torch.backends.cudnn.benchmark = not enabled
    torch.backends.cudnn.deterministic = enabled
    torch.use_deterministic_algorithms(enabled, warn_only=True)

def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic_backend(deterministic)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def save_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)
