"""Shared image/corruption/preprocessing interface for CNN and later XAI."""
from collections import OrderedDict
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .noise import STATES, corrupt, training_state, rng_for
from .prepare import MANIFESTS, validate_frames
from .release.denoising import apply_condition, cnn_tensor
from .release.frequency import dwt_components, dft_components
from .release.training import random_affine
from .release.experiment import file_hash


class InputData:
    def __init__(self, directory, config):
        self.directory = Path(directory).resolve()
        self.config = config
        self.receipt = json.loads((self.directory / "dataset.json").read_text())
        self.frames = {name: pd.read_csv(self.directory / f"{name}.csv") for name in MANIFESTS}
        for name in MANIFESTS:
            if file_hash(self.directory / f"{name}.csv") != self.receipt["manifest_hashes"][name]:
                raise ValueError(f"Manifest integrity failed: {name}")
        validate_frames(self.frames, config["counts_per_class"])
        if len(self.frames["xai"]) < 1:
            raise ValueError("XAI reference must retain at least 1 infiltration image")
        self.working, self.geometry = {}, {}
        combined = pd.concat(self.frames.values())
        for image_id, copies in combined.groupby("image", sort=True):
            if copies.sha256.nunique() != 1 or copies.patient_id.astype(str).nunique() != 1:
                raise ValueError("Image identity differs between manifests")
            row = copies.iloc[0]
            path = (self.directory / row.relative_path).resolve()
            if not path.is_relative_to(self.directory) or file_hash(path) != row.sha256:
                raise ValueError(f"Input image integrity failed: {image_id}")
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (int(row.height), int(row.width)):
                raise ValueError(f"Image dimensions changed: {image_id}")
            size = config["preprocessing"]["working_size"]
            self.working[image_id] = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
            self.geometry[image_id] = {
                "native_width": int(row.width),
                "native_height": int(row.height),
                "working_width": size,
                "working_height": size,
                "cnn_width": config["preprocessing"]["cnn_size"],
                "cnn_height": config["preprocessing"]["cnn_size"],
                "mapping": "full-frame resize; no crop",
                "image_sha256": row.sha256,
            }

    def noisy(self, image_id, state_id, epoch=0):
        return corrupt(self.working[image_id], state_id, seed=self.config["seed"], image_id=image_id, epoch=epoch)


def expanded(frame):
    """Exactly one copy of each source image in every fixed corruption state."""
    return pd.concat([frame.assign(corruption_id=state["id"]) for state in STATES], ignore_index=True)


class Preprocessor:
    def __init__(self, data, condition, config, dae=None, device="cpu"):
        self.data, self.condition, self.config = data, condition, config
        self.dae, self.device = dae, torch.device(device)
        self.cache = OrderedDict()

    @property
    def view_order(self):
        group = self.condition["group"]
        return (["DWT_A4", "DWT_D1", "DWT_D2", "DWT_D3", "DWT_D4"] if group == "dwt_features" else
                ["DFT_low", "DFT_mid", "DFT_high"] if group == "dft_features" else ["image"])

    def image(self, image_id, state_id, epoch=0):
        if self.condition["group"] in ("dwt_features", "dft_features"):
            raise ValueError("Frequency fusion has signed views, not a denoised image")
        key = (image_id, state_id)
        if epoch == 0 and key in self.cache:
            return self.cache[key].copy()
        noisy, _ = self.data.noisy(image_id, state_id, epoch)
        result = apply_condition(noisy, self.condition, self.config, self.dae, str(self.device))
        if epoch == 0:
            self.cache[key] = result
            while len(self.cache) > 2048:
                self.cache.popitem(last=False)
        return result.copy()

    def tensor(self, image_id, state_id="clean", epoch=0):
        """Return (V,3,H,W); input gradients remain enabled in the CNN forward."""
        group = self.condition["group"]
        if group not in ("dwt_features", "dft_features"):
            return cnn_tensor(self.image(image_id, state_id, epoch), self.config).unsqueeze(0)
        noisy, _ = self.data.noisy(image_id, state_id, epoch)
        c = self.condition
        x = noisy.astype(np.float32) / 255
        if group == "dwt_features":
            bands = dwt_components(x, c["wavelet"], c["depth"], c["mode"])
        else:
            bands = dft_components(x, c["low_cutoff"], c["high_cutoff"])
        bands[1:] = .5 + .5 * bands[1:]
        settings = self.config["preprocessing"]
        size = settings["cnn_size"]
        views = np.stack([cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR) for b in bands])
        tensor = torch.from_numpy(np.repeat(views[:, None], 3, axis=1).astype(np.float32))
        tensor = (tensor - torch.tensor(settings["mean"]).view(1, 3, 1, 1)) / torch.tensor(settings["std"]).view(1, 3, 1, 1)
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite multi-view tensor")
        return tensor


class ClassificationDataset(Dataset):
    def __init__(self, frame, preprocessor, epoch=0):
        self.records = frame.to_dict("records")
        self.preprocessor, self.epoch = preprocessor, int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        seed = self.preprocessor.config["seed"]
        state = training_state(seed, row["image"], self.epoch) if self.epoch else row["corruption_id"]
        views = self.preprocessor.tensor(row["image"], state, self.epoch)
        if self.epoch:
            views = random_affine(views, rng_for(seed, row["image"], "geometry", self.epoch, "geometry"),
                                  self.preprocessor.config["cnn"]["augmentation"])
        return views, torch.tensor([float(row["label"])])


class ReconstructionDataset(Dataset):
    def __init__(self, frame, data, epoch=0):
        self.records, self.data, self.epoch = frame.to_dict("records"), data, int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        state = training_state(self.data.config["seed"], row["image"], self.epoch) if self.epoch else row["corruption_id"]
        noisy, _ = self.data.noisy(row["image"], state, self.epoch)
        return torch.from_numpy(noisy[None].astype(np.float32) / 255), torch.from_numpy(
            self.data.working[row["image"]][None].astype(np.float32) / 255)
