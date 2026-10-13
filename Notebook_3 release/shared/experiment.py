"""Configuration, provenance, and durable writes. Importing never starts a run."""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from pathlib import Path
import json
import os
import random
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


@dataclass
class Experiment:
    root: Path
    data_root: Path
    output_root: Path
    config: dict
    cache_root: Path = None
    staging_root: Path = None
    cache: object = None

    @classmethod
    def open(cls, root, data_root=None, output_root=None, smoke=False, seed=42,
             cache_root=None, profile="release_controlled", model_protocol=None):
        root = Path(root).resolve()
        config = json.loads((root / "configs/release_full.json").read_text(encoding="utf-8"))
        if profile not in ("release_controlled", "legacy_v2"):
            raise ValueError("Unknown method profile")
        if seed not in config["training_seeds"]:
            raise ValueError("Training seed must be one of the fixed release seeds")
        config["seed"] = int(seed)
        config["profile"] = profile
        config["conditions"] = json.loads((root / f"configs/methods_{profile}.json").read_text(encoding="utf-8"))
        if model_protocol is not None:
            if model_protocol not in ("layer4_after_warmup", "head_only_legacy"):
                raise ValueError("Unknown model protocol")
            config["cnn"]["trainable_layers"] = model_protocol
        if smoke:
            config["data"]["expected_counts"] = {"train": 4, "validation": 4, "test": 4, "xai": 2, "background": 2}
            config["data"]["expected_class_counts"] = {k: {"0": 2, "1": 2} for k in ("train", "validation", "test")}
            config["data"].update(train_per_class=2, validation_per_class=2, test_per_class=2)
            config["training"]["images_per_epoch"] = 4
            config["dae"].update(epochs=1, batch_size=4, amp=False)
            config["cnn"].update(epochs=4, batch_size=4, accumulation_steps=1, effective_batch_size=4, pretrained=False, amp=False)
            config["xai"].update(background_per_class=1, shap_nsamples=4, shap_batch_size=2, bootstrap_replicates=20, visual_examples=1)
            config["reporting"]["classification_bootstrap_replicates"] = 20
            config["runtime_cache"]["shard_size"] = 4
        config["run_mode"] = "smoke" if smoke else "full"
        source_files = sorted([*root.glob("*.py"), *(root / "shared").glob("*.py")])
        config["source_code_hashes"] = {p.relative_to(root).as_posix(): file_hash(p) for p in source_files}
        if data_root is None:
            data_root = os.environ.get("CXR_DATA_ROOT")
        if data_root is None:
            local = next((parent / "data/versions/3" for parent in root.parents
                          if (parent / "data/versions/3/Data_Entry_2017.csv").is_file()), None)
            data_root = local or config["data"]["colab_data_root"]
        is_colab = Path("/content/drive/MyDrive").is_dir()
        output_root = Path(output_root or (config["data"]["colab_output_root"] if is_colab else root / config["output_subdirectory"])).resolve()
        if smoke:
            output_root = output_root / "smoke_runs"
        if profile != "release_controlled":
            output_root = output_root / profile
        if config["cnn"]["trainable_layers"] != "layer4_after_warmup":
            output_root = output_root / config["cnn"]["trainable_layers"]
        base_output = output_root
        output_root = output_root / f"seed_{seed}"
        data_root = Path(data_root).resolve()
        cache_root = Path(cache_root or os.environ.get("CXR_CACHE_ROOT") or
                          (config["runtime_cache"]["colab_root"] if is_colab else root / ".runtime/cache")).resolve()
        if output_root.is_relative_to(data_root) or cache_root.is_relative_to(data_root):
            raise ValueError("Dataset directory is read-only; choose separate output/cache roots")
        staging_root = cache_root / "staging" / digest(config) / f"seed_{seed}"
        staging_root.mkdir(parents=True, exist_ok=True)
        exp = cls(root, data_root, output_root, config, cache_root, staging_root)
        exp.base_output_root = base_output
        if not (exp.data_root / "Data_Entry_2017.csv").is_file():
            raise FileNotFoundError(f"Mount Drive and set DATA_ROOT to nih -xray/data/3: {exp.data_root}")
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
        hashes = getattr(self, "_manifest_hashes", None)
        if hashes is None:
            hashes = {p.name: file_hash(p) for p in sorted(manifest_dir.glob("*.csv"))}
            self._manifest_hashes = hashes
        if set(hashes) != {"train.csv", "validation.csv", "test.csv", "xai.csv", "background.csv"}:
            raise RuntimeError("Prepare and verify all five manifests first")
        result = {
            "config_hash": digest(self.config), "config": self.config,
            "manifest_hashes": hashes, "condition_id": condition_id,
            "versions": getattr(self, "_versions", None) or versions(),
        }
        pixel_marker = self.artifacts / "input_source.json"
        pixel_hash = getattr(self, "_input_source_hash", None)
        if pixel_hash is None and pixel_marker.is_file():
            pixel_hash = file_hash(pixel_marker)
            self._input_source_hash = pixel_hash
        result["input_source_hash"] = pixel_hash
        self._versions = result["versions"]
        if condition_id and self.condition(condition_id)["group"] == "dae_clahe":
            path = self.artifacts / "dae_checkpoint/best.pt"
            stat = path.stat()
            fingerprint = (stat.st_size, stat.st_mtime_ns)
            old = getattr(self, "_dae_hash", None)
            if old is None or old[0] != fingerprint:
                old = (fingerprint, file_hash(path))
                self._dae_hash = old
            result["dae_checkpoint_hash"] = old[1]
        return result

    def check_provenance(self, saved, condition_id=None):
        expected = self.provenance(condition_id)
        for key in ("config_hash", "manifest_hashes", "condition_id", "dae_checkpoint_hash", "input_source_hash"):
            if saved.get(key) != expected.get(key):
                raise ValueError(f"Incompatible experiment {key}; use a separate OUTPUT_ROOT")
        if saved.get("versions") != expected["versions"]:
            raise ValueError("Package versions changed; restore the recorded environment or use a new OUTPUT_ROOT")

    def lock(self):
        path = self.artifacts / "experiment.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            current = self.provenance()
            if saved.get("input_source_hash") is None and current.get("input_source_hash") is not None:
                checkpoints = [*(self.output_root / "CNN/results").glob("*/*.pt"),
                               *(self.artifacts / "dae_checkpoint").glob("*.pt")]
                if checkpoints:
                    raise ValueError("Existing checkpoints lack pixel provenance; choose a new OUTPUT_ROOT")
                self.check_provenance({**saved, "input_source_hash": current["input_source_hash"]})
                write_json(path, current)
            else:
                self.check_provenance(saved)
        else:
            write_json(path, self.provenance())

    def load_checkpoint(self, path, condition_id=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.check_provenance(checkpoint["provenance"], condition_id)
        return checkpoint
