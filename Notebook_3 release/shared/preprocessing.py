"""Common multi-view inputs for CNN, Grad-CAM and SHAP; bounded runtime cache."""
from collections import OrderedDict
from pathlib import Path
import json
import shutil

import cv2
import numpy as np
import torch

from .denoising import apply_condition, cnn_tensor, load_dae
from .experiment import digest, file_hash, write_json
from .frequency import dwt_components, dft_components
from .runtime_cache import RuntimeCache, shard_valid


class ReleasePreprocessor:
    def __init__(self, exp, condition_id, device="cpu"):
        self.exp = exp
        self.condition = dict(exp.condition(condition_id))
        self.condition["profile"] = exp.config.get("profile", "release_controlled")
        self.device = torch.device(device)
        self.dae = None
        self._images = OrderedDict()
        self._shards = OrderedDict()
        self.directory = None
        self.dependency = None
        if self.condition["group"] == "dae_clahe":
            self.dae = load_dae(exp, self.device)
            self.dependency = file_hash(exp.artifacts/"dae_checkpoint/best.pt")

    def _cache(self):
        if self.exp.cache is None:
            raise RuntimeError("Prepare RuntimeCache with all five manifests before preprocessing")
        return self.exp.cache

    def image(self, row):
        source = self._cache()
        if self.condition["group"] in ("dwt_features", "dft_features"):
            raise ValueError("Frequency conditions have signed components, not a uint8 processed image")
        if self.condition["group"] == "baseline":
            return source.working(row)
        if self.directory is not None:
            index = source.indices[row["image"]]
            shard, offset = divmod(index, source.shard_size)
            if shard not in self._shards:
                self._shards[shard] = np.load(self.directory/f"shard_{shard:05}.npy", mmap_mode="r", allow_pickle=False)
                while len(self._shards) > 4:
                    self._shards.popitem(last=False)
            return np.array(self._shards[shard][offset], copy=True)
        image_id = row["image"]
        if image_id not in self._images:
            self._images[image_id] = apply_condition(source.working(row), self.condition, self.exp.config, self.dae, self.device)
            while len(self._images) > 16:
                self._images.popitem(last=False)
        self._images.move_to_end(image_id)
        return self._images[image_id].copy()

    def components(self, row):
        x = self._cache().working(row).astype(np.float32)/255
        c = self.condition
        if c["group"] == "dwt_features":
            return dwt_components(x, c.get("wavelet", "db1"), c.get("depth", 4), c.get("mode", "periodization"))
        if c["group"] == "dft_features":
            return dft_components(x, c.get("low_cutoff", .125), c.get("high_cutoff", .25))
        return self.image(row).astype(np.float32)[None]/255

    def tensor(self, row):
        if self.condition["group"] not in ("dwt_features", "dft_features"):
            return cnn_tensor(self.image(row), self.exp.config).unsqueeze(0)
        bands = self.components(row)
        # Preserve signed details, without uint8 conversion or per-image scaling.
        bands[1:] = .5+.5*bands[1:]
        prep = self.exp.config["preprocessing"]
        size = prep["cnn_size"]
        resized = np.stack([cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR) for b in bands])
        tensor = torch.from_numpy(np.repeat(resized[:, None], 3, axis=1).astype(np.float32))
        tensor = (tensor-torch.tensor(prep["mean"]).view(1, 3, 1, 1))/torch.tensor(prep["std"]).view(1, 3, 1, 1)
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite frequency input")
        return tensor

    def prepare(self, frames):
        self._shards.clear()
        self.directory = None
        if self.exp.cache is None or not getattr(self.exp.cache, "ready", True):
            RuntimeCache(self.exp).prepare(frames)
        source = self._cache()
        if self.condition["group"] in ("baseline", "dwt_features", "dft_features"):
            return self
        key = digest({"source_cache": source.key, "preprocessing": self.exp.config["preprocessing"],
                      "condition": self.condition, "dae_sha256": self.dependency,
                      "implementation_sha256": {name: file_hash(Path(__file__).with_name(name))
                           for name in ("preprocessing.py", "denoising.py", "dae_model.py")}})
        parent = Path(self.exp.cache_root)/"derived"
        directory = parent/key
        parent.mkdir(parents=True, exist_ok=True)
        settings = self.exp.config.get("runtime_cache", self.exp.config.get("cache", {}))
        budget = int(settings.get("derived_budget_bytes", settings.get("derived_max_gib", 6)*1024**3))
        needed = len(source.rows)*source.size*source.size+len(source.shards)*256
        if needed > budget:
            raise OSError(f"Derived cache needs {needed/1024**3:.2f} GiB, exceeding derived_max_gib")
        # Retire only our generated, completed derivative directories on VM disk.
        siblings = sorted((p for p in parent.iterdir() if p.is_dir() and p != directory), key=lambda p: p.stat().st_mtime)
        used = sum(f.stat().st_size for p in siblings for f in p.rglob("*") if f.is_file())
        for sibling in siblings:
            if used+needed <= budget:
                break
            marker = sibling/"complete.json"
            if not marker.exists():
                continue
            record = json.loads(marker.read_text())
            if record.get("cache_schema") != 1 or record.get("key") != sibling.name:
                raise ValueError("Refusing to retire unrecognized derivative cache")
            released = sum(f.stat().st_size for f in sibling.rglob("*") if f.is_file())
            if not sibling.resolve().is_relative_to(parent.resolve()):
                raise ValueError("Unsafe derivative cache path")
            shutil.rmtree(sibling)
            used -= released
        if used+needed > budget:
            raise OSError("Incomplete derivative caches exceed budget; remove their runtime-only folders before retry")
        directory.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(parent).free < needed and not (directory/"complete.json").exists():
            raise OSError("Insufficient VM disk for derived preprocessing cache")
        for shard, start in enumerate(range(0, len(source.rows), source.shard_size)):
            stop = min(start+source.shard_size, len(source.rows))
            path = directory/f"shard_{shard:05}.npy"
            marker = path.with_suffix(".json")
            shape = (stop-start, source.size, source.size)
            signature = digest({"key": key, "start": start, "stop": stop})
            if shard_valid(path, marker, signature, shape):
                continue
            marker.unlink(missing_ok=True)
            temporary = path.with_suffix(".npy.tmp")
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
            try:
                for offset, row in enumerate(source.rows[start:stop]):
                    array[offset] = apply_condition(source.working(row), self.condition, self.exp.config, self.dae, self.device)
                array.flush()
            finally:
                del array
            temporary.replace(path)
            write_json(marker, {"signature": signature, "shape": list(shape), "sha256": file_hash(path)})
            if (shard+1) % 10 == 0:
                print(f"{self.condition['id']}: {stop}/{len(source.rows)} prepared", flush=True)
        write_json(directory/"complete.json", {"cache_schema": 1, "key": key, "images": len(source.rows)})
        self.directory = directory
        self._images.clear()
        if self.dae is not None:
            self.dae.cpu()
        self.device = torch.device("cpu")
        return self


# Backward compatible import for previous notebooks' single preprocessing owner.
Preprocessor = ReleasePreprocessor
