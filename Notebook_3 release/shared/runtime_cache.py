"""Verified uint8 shards on runtime disk; original Drive files are read only."""
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import json
import shutil
import time

import cv2
import numpy as np
import pandas as pd

from .data import validate_relative_path
from .denoising import working_image
from .experiment import digest, file_hash, write_json


def unique_rows(frames):
    frames = list(frames.values()) if isinstance(frames, dict) else list(frames)
    combined = pd.concat(frames, ignore_index=True)
    for image, group in combined.groupby("image", sort=False):
        if group.relative_path.nunique() != 1:
            raise ValueError(f"Conflicting paths for {image}")
    return combined.drop_duplicates("image").sort_values("image").to_dict("records")


def shard_valid(path, marker, signature, shape):
    """Completion is insufficient without checking actual payload and geometry."""
    try:
        saved = json.loads(marker.read_text())
        if saved.get("signature") != signature or saved.get("shape") != list(shape):
            return False
        if file_hash(path) != saved["sha256"]:
            return False
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        good = array.dtype == np.uint8 and array.shape == shape
        del array
        return good
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


class RuntimeCache:
    def __init__(self, exp):
        self.exp = exp
        self.root = Path(exp.cache_root).resolve()
        data = exp.data_root.resolve()
        if self.root == data or self.root.is_relative_to(data):
            raise ValueError("Runtime cache must be outside the read-only dataset")
        # Mounted Drive must not receive tens of thousands of intermediate writes.
        if "/content/drive/" in self.root.as_posix():
            raise ValueError("CACHE_ROOT must be on the Colab VM, outside mounted Drive")
        self.settings = exp.config.get("runtime_cache", exp.config.get("cache", {}))
        self.shard_size = int(self.settings.get("shard_size", 1024))
        if self.shard_size < 1:
            raise ValueError("Cache shard size must be positive")
        self._maps = OrderedDict()
        self._native_shapes = {}
        self.ready = False

    def _layout(self, rows):
        folders = {}
        for row in rows:
            relative = validate_relative_path(row)
            group = relative.parts[0]
            if group not in folders:
                # Resolve one directory per original image shard, never glob Drive.
                candidates = [self.exp.data_root/group/"images", self.exp.data_root/group]
                existing = next((p for p in candidates if p.is_dir()), None)
                if existing is None:
                    raise FileNotFoundError(f"Missing selected image folder: {group}")
                folders[group] = existing
        return folders

    def _read(self, row):
        folder = self.folders[validate_relative_path(row).parts[0]]
        path = folder/row["image"]
        tries = int(self.settings.get("read_retries", 3))
        if not 1 <= tries <= 5:
            raise ValueError("read_retries must be 1..5")
        for attempt in range(tries):
            try:
                encoded = np.fromfile(path, dtype=np.uint8)
                raw = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
                if raw is None:
                    raise ValueError(f"Cannot decode selected image: {path}")
                return raw, sha256(encoded.tobytes()).hexdigest()
            except FileNotFoundError:
                raise FileNotFoundError(f"Missing selected image: {path}") from None
            except (OSError, ValueError):
                if attempt+1 == tries:
                    raise
                time.sleep(min(.5*(attempt+1), 2))

    def _native_path(self, image):
        if Path(image).name != image:
            raise ValueError("Unsafe native cache image ID")
        return self.directory/"native"/(image+".npy")

    def _save_native(self, row, raw, source_sha):
        path = self._native_path(row["image"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, raw, allow_pickle=False)
        temporary.replace(path)
        write_json(path.with_suffix(".json"), {"signature": self.key, "shape": list(raw.shape),
                   "sha256": file_hash(path), "source_sha256": source_sha})

    def prepare(self, frames):
        self.ready = False
        self._maps.clear()
        self._native_shapes.clear()
        self.rows = unique_rows(frames)
        xai = frames.get("xai") if isinstance(frames, dict) else None
        if xai is None:
            xai_frames = [f for f in (frames.values() if isinstance(frames, dict) else frames) if "bbox_x" in f.columns]
            xai = pd.concat(xai_frames) if xai_frames else pd.DataFrame({"image": []})
        self.native_ids = set(xai.image)
        self.size = int(self.exp.config["preprocessing"]["working_size"])
        self.signature = {"cache_schema": 1, "source_hashes": self.exp.config["data"]["source_hashes"],
            "manifest_hash": digest([(r["image"], r["relative_path"]) for r in self.rows]),
            "native_ids": sorted(self.native_ids), "working_size": self.size,
            "resize": "opencv_INTER_AREA_uint8", "opencv_version": cv2.__version__,
            "resize_code": digest(working_image.__code__.co_code.hex()), "shard_size": self.shard_size}
        self.key = digest(self.signature)
        self.directory = self.root/"source"/self.key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.folders = self._layout(self.rows)
        required = len(self.rows)*self.size*self.size
        missing_payload = sum(min(self.shard_size, len(self.rows)-start)*self.size*self.size
                              for start in range(0, len(self.rows), self.shard_size)
                              if not (self.directory/f"shard_{start//self.shard_size:05}.npy").exists())
        if shutil.disk_usage(self.root).free < missing_payload+min(required, 256*1024**2):
            raise OSError(f"Insufficient runtime disk for source cache; need about {required/1024**3:.2f} GiB pixel payload")
        self.indices = {r["image"]: i for i, r in enumerate(self.rows)}
        self.shards = []
        raw_hashes, working_hashes = {}, []
        for start in range(0, len(self.rows), self.shard_size):
            stop = min(start+self.shard_size, len(self.rows))
            path = self.directory/f"shard_{start//self.shard_size:05}.npy"
            marker = path.with_suffix(".json")
            shape = (stop-start, self.size, self.size)
            signature = digest({"key": self.key, "start": start, "stop": stop})
            valid = shard_valid(path, marker, signature, shape)
            if valid:
                saved = json.loads(marker.read_text())
                records = saved.get("source_image_sha256", [])
                if not isinstance(records, list):
                    records = []
                expected_ids = [r["image"] for r in self.rows[start:stop]]
                valid = ([entry[0] for entry in records if isinstance(entry, list) and len(entry) == 2] == expected_ids
                         and all(isinstance(entry[1], str) and len(entry[1]) == 64
                                 and all(c in "0123456789abcdef" for c in entry[1]) for entry in records))
            if not valid:
                marker.unlink(missing_ok=True)
                temporary = path.with_suffix(".npy.tmp")
                array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
                source_checksums = []
                try:
                    for offset, row in enumerate(self.rows[start:stop]):
                        raw, source_sha = self._read(row)
                        array[offset] = working_image(raw, self.exp.config)
                        source_checksums.append([row["image"], source_sha])
                        if row["image"] in self.native_ids:
                            self._save_native(row, raw, source_sha)
                    array.flush()
                finally:
                    del array
                temporary.replace(path)
                write_json(marker, {"signature": signature, "shape": list(shape), "sha256": file_hash(path),
                           "source_image_sha256": source_checksums})
            saved = json.loads(marker.read_text())
            raw_hashes.update(saved["source_image_sha256"])
            working_hashes.append([path.name, saved["sha256"]])
            self.shards.append(path)
            if (start//self.shard_size+1) % 10 == 0:
                print(f"Source cache: {stop}/{len(self.rows)} checked", flush=True)
        for row in self.rows:
            if row["image"] in self.native_ids:
                path = self._native_path(row["image"])
                try:
                    marker = json.loads(path.with_suffix(".json").read_text())
                    shape = tuple(marker["shape"])
                    if marker.get("source_sha256") != raw_hashes[row["image"]]:
                        raise ValueError("Native/source pixel fingerprints differ")
                    if not shard_valid(path, path.with_suffix(".json"), self.key, shape):
                        raise ValueError("Invalid native payload")
                except (OSError, ValueError, KeyError):
                    raw, source_sha = self._read(row)
                    if source_sha != raw_hashes[row["image"]]:
                        raise ValueError(f"Source pixels changed while repairing native cache: {row['image']}; use a new output root")
                    self._save_native(row, raw, source_sha)
                    shape = raw.shape
                self._native_shapes[row["image"]] = tuple(shape)
        # This survives VM resets, unlike cache sidecars. A fresh cache cannot
        # silently resume checkpoints after even one original PNG has changed.
        native_hashes = [[image, file_hash(self._native_path(image))] for image in sorted(self.native_ids)]
        input_record = {"schema_version": 1, "source_metadata_hashes": self.exp.config["data"]["source_hashes"],
                        "source_image_sha256_digest": digest(sorted(raw_hashes.items())),
                        "working_pixel_sha256_digest": digest(working_hashes),
                        "native_pixel_sha256_digest": digest(native_hashes),
                        "source_images": len(self.rows), "native_images": len(self.native_ids),
                        "cache_key": self.key, "cache_signature": self.signature}
        input_record["fingerprint"] = digest(input_record)
        durable = self.exp.artifacts / "input_source.json"
        if durable.exists():
            if json.loads(durable.read_text()) != input_record:
                raise ValueError("Source/working/native pixel provenance changed; refuse resume and use a new output root")
        else:
            write_json(durable, input_record)
        self.input_source_hash = file_hash(durable)
        self.exp._input_source_hash = self.input_source_hash
        write_json(self.directory/"complete.json", {"key": self.key, "signature": self.signature,
                   "images": len(self.rows), "native_images": len(self.native_ids)})
        self.ready = True
        self.exp.cache = self
        return self

    def working(self, row):
        if not self.ready:
            raise RuntimeError("Source cache is incomplete; finish prepare before use")
        if row["image"] not in self.indices:
            raise KeyError(f"Image not in prepared runtime cache: {row['image']}")
        index = self.indices[row["image"]]
        shard, offset = divmod(index, self.shard_size)
        if shard not in self._maps:
            self._maps[shard] = np.load(self.shards[shard], mmap_mode="r", allow_pickle=False)
            while len(self._maps) > int(self.settings.get("open_shards", 4)):
                self._maps.popitem(last=False)
        self._maps.move_to_end(shard)
        # Do not expose writable/long-lived views into shared mmap buffers.
        return np.array(self._maps[shard][offset], copy=True)

    def native(self, row):
        if not self.ready:
            raise RuntimeError("Source cache is incomplete; finish prepare before use")
        if row["image"] not in self.native_ids:
            raise ValueError("Native pixels are cached only for frozen XAI images")
        image = np.load(self._native_path(row["image"]), allow_pickle=False)
        if image.dtype != np.uint8 or image.shape != self._native_shapes[row["image"]]:
            raise ValueError("Native cache shape/dtype changed")
        return image
