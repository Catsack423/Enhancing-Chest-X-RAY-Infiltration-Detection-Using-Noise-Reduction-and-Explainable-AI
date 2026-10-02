"""Configuration, provenance, and durable writes. Importing never starts a run."""
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
    for name in ("torch", "torchvision", "shap", "numpy", "pandas", "opencv-python", "PyWavelets", "scikit-learn"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


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


@dataclass
class Experiment:
    root: Path
    data_root: Path
    output_root: Path
    config: dict

    @classmethod
    def open(cls, root, data_root=None, output_root=None, smoke=False):
        root = Path(root).resolve()
        config = json.loads((root / "shared/config.json").read_text(encoding="utf-8"))
        if smoke:
            config = deepcopy(config)
            config["data"].update(train_per_class=2, validation_per_class=2)
            config["data"].update(test_per_class=2, expected_test_images=4)
            config["training"]["images_per_epoch"] = 4
            config["dae"].update(epochs=1, batch_size=2)
            config["cnn"].update(epochs=1, batch_size=2)
            config["xai"].update(background_per_class=1, shap_nsamples=4, bootstrap_replicates=20, visual_examples=1)
        config["run_mode"] = "smoke" if smoke else "full"
        if data_root is None:
            data_root = os.environ.get("CXR_DATA_ROOT")
        if data_root is None:
            local = next((parent / "data/versions/3" for parent in root.parents
                          if (parent / "data/versions/3/Data_Entry_2017.csv").is_file()), None)
            data_root = local or config["data"]["colab_data_root"]
        output_root = Path(output_root or root / config.get("output_subdirectory", "")).resolve()
        if smoke:
            output_root = output_root / "smoke_runs"
        exp = cls(root, Path(data_root).resolve(), output_root, config)
        if not (exp.data_root / "Data_Entry_2017.csv").is_file():
            raise FileNotFoundError(f"Set DATA_ROOT to the NIH data/versions/3 directory: {exp.data_root}")
        return exp

    @property
    def artifacts(self):
        return self.output_root / "shared/artifacts"

    @property
    def condition_ids(self):
        return [c["id"] for c in self.config["conditions"]]

    def condition(self, condition_id):
        return next(c for c in self.config["conditions"] if c["id"] == condition_id)

    def results(self, criterion, condition_id):
        self.condition(condition_id)
        return self.output_root / criterion / "results" / condition_id

    def provenance(self, condition_id=None):
        manifest_dir = self.artifacts / "manifests"
        hashes = {p.name: file_hash(p) for p in sorted(manifest_dir.glob("*.csv"))}
        if set(hashes) != {"train.csv", "validation.csv", "test.csv", "xai.csv", "background.csv"}:
            raise RuntimeError("Prepare and verify all five manifests first")
        result = {
            "config_hash": digest(self.config), "config": self.config,
            "manifest_hashes": hashes, "condition_id": condition_id,
            "versions": versions(),
        }
        if condition_id and self.condition(condition_id)["group"] == "dae_clahe":
            result["dae_checkpoint_hash"] = file_hash(self.artifacts / "dae_checkpoint/best.pt")
        return result

    def check_provenance(self, saved, condition_id=None):
        expected = self.provenance(condition_id)
        for key in ("config_hash", "manifest_hashes", "condition_id", "dae_checkpoint_hash"):
            if saved.get(key) != expected.get(key):
                raise ValueError(f"Incompatible experiment {key}; use a separate OUTPUT_ROOT")
        if saved.get("versions") != expected["versions"]:
            raise ValueError("Package versions changed; restore the recorded environment or use a new OUTPUT_ROOT")

    def lock(self):
        path = self.artifacts / "experiment.json"
        if path.exists():
            self.check_provenance(json.loads(path.read_text(encoding="utf-8")))
        else:
            write_json(path, self.provenance())

    def load_checkpoint(self, path, condition_id=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.check_provenance(checkpoint["provenance"], condition_id)
        return checkpoint
