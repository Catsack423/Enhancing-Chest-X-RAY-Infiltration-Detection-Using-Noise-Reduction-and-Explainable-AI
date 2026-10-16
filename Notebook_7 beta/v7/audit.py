"""Resumable all-image noise audit. Never changes the dataset or its partition."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
import pandas as pd
import pywt

from .release.experiment import digest, file_hash, write_json

AUDIT_VERSION = "db2-patch64-central80-v1"


def wavelet_sigma(values):
    hh = pywt.dwt2(values.astype(np.float64), "db2", mode="periodization", axes=(-2, -1))[1][2]
    # Omit coefficients touching the artificial patch boundary.
    hh = hh[..., 2:-2, 2:-2]
    center = np.median(hh, axis=(-2, -1), keepdims=True)
    return np.median(np.abs(hh - center), axis=(-2, -1)) / .67448975


def estimate(image):
    if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 64:
        raise ValueError("Expected grayscale uint8 image of at least 64 pixels per side")
    height, width = image.shape
    top, left = int(height * .1), int(width * .1)
    center = image[top:height-top, left:width-left]
    rows, cols = center.shape[0] // 64, center.shape[1] // 64
    patches = center[:rows*64, :cols*64].reshape(rows, 64, cols, 64).transpose(0, 2, 1, 3).reshape(-1, 64, 64)
    valid = (((patches >= 5) & (patches <= 250)).mean(axis=(1, 2)) >= .9) & (patches.std(axis=(1, 2)) > 0)
    usable = patches[valid].astype(np.float64)
    result = {"global_sigma": float(wavelet_sigma(image) / 255),
              "contrast": float(image.std() / 255),
              "sharpness": float(cv2.Laplacian(image.astype(np.float32) / 255, cv2.CV_32F).var()),
              "valid_patches": len(usable), "selected_patches": 0, "noise_sigma": None,
              "status": "insufficient_patches"}
    if len(usable) >= 8:
        gradient = (np.median(np.abs(np.diff(usable, axis=1)), axis=(1, 2)) +
                    np.median(np.abs(np.diff(usable, axis=2)), axis=(1, 2)))
        count = math.ceil(len(usable) * .25)
        selected = usable[np.argsort(gradient, kind="stable")[:count]]
        result.update(noise_sigma=float(np.median(wavelet_sigma(selected)) / 255),
                      selected_patches=count, status="ok")
    return result


def image_index(data_root):
    root = Path(data_root).resolve()
    result = {}
    for path in sorted(set(root.glob("images_*/images/*.png")) | set(root.glob("images_*/*.png"))):
        if path.name in result:
            raise ValueError(f"Duplicate image filename: {path.name}")
        if not path.resolve().is_relative_to(root):
            raise ValueError("Image path escapes the dataset")
        result[path.name] = path
    return result


def read_image(path):
    raw = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image decoding failed")
    from hashlib import sha256
    return image, sha256(raw).hexdigest()


def audit(data_root, output, partition_root, *, workers=4, limit=None, reuse_complete=False, image_ids=None):
    data_root, output, partition_root = Path(data_root).resolve(), Path(output), Path(partition_root)
    if output.resolve().is_relative_to(data_root):
        raise ValueError("Audit output must be outside the read-only dataset")
    cv2.setNumThreads(1)
    metadata = pd.read_csv(data_root / "Data_Entry_2017.csv")
    if metadata["Image Index"].duplicated().any():
        raise ValueError("Duplicate metadata image IDs")
    official_test = set((data_root / "test_list.txt").read_text().split())
    official_train = set((data_root / "train_val_list.txt").read_text().split())
    if official_test & official_train or set(metadata["Image Index"]) != official_test | official_train:
        raise ValueError("Official image lists do not partition the metadata")
    patients_test = set(metadata.loc[metadata["Image Index"].isin(official_test), "Patient ID"])
    patients_train = set(metadata.loc[metadata["Image Index"].isin(official_train), "Patient ID"])
    if patients_test & patients_train:
        raise ValueError("Official patients overlap")
    requested = metadata
    if image_ids is not None:
        image_ids = set(image_ids)
        if not image_ids or not image_ids <= set(metadata["Image Index"]):
            raise ValueError("ZIP image IDs are empty or outside the official metadata")
        requested = metadata[metadata["Image Index"].isin(image_ids)]
    partition = {}
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(partition_root / f"{split}.csv")
        partition.update(dict.fromkeys(frame.image, split))
    sources = [data_root / name for name in ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt")]
    sources += sorted(partition_root.glob("*.csv"))
    signature = digest({"algorithm": AUDIT_VERSION, "source": file_hash(__file__),
                        "inputs": {p.name: file_hash(p) for p in sources},
                        "image_ids": sorted(image_ids) if image_ids is not None else None})
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "audit_config.json"
    if lock.exists() and json.loads(lock.read_text())["signature"] != signature:
        raise ValueError("Audit algorithm/metadata/partition changed; use a new audit directory")
    write_json(lock, {"signature": signature})
    completed = output / "summary.json"
    if reuse_complete and limit is None and completed.exists():
        receipt = json.loads(completed.read_text())
        if receipt["complete"] and receipt["signature"] == signature and receipt["statuses"].get("error", 0) == 0:
            if file_hash(output / "noise_audit.csv") != receipt["csv_sha256"]:
                raise ValueError("Frozen audit CSV changed")
            print("Reusing the complete, verified Drive audit.", flush=True)
            return pd.read_csv(output / "noise_audit.csv")
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    known = {}
    for path in sorted(chunks.glob("*.json")):
        saved = json.loads(path.read_text())
        if saved["signature"] != signature:
            raise ValueError("Incompatible audit chunk")
        known.update({record["image"]: record for record in saved["records"]})
    lookup = image_index(data_root)
    items = requested.to_dict("records")[:limit]
    missing_images = [row["Image Index"] for row in items if row["Image Index"] not in lookup]
    if missing_images:
        raise FileNotFoundError(
            f"Image index found {len(lookup):,} PNGs under {data_root}; "
            f"{len(missing_images):,} requested images are missing, e.g. {missing_images[:3]}. "
            "Check the mounted Google account, folder access and images_001/images/*.png layout. "
            "No image audit batches were written.")
    started = time.monotonic()

    def inspect(entry):
        name = entry["Image Index"]
        view = entry.get("View Position", "")
        record = {"image": name, "patient_id": int(entry["Patient ID"]), "finding_labels": entry["Finding Labels"],
                  "view_position": "" if pd.isna(view) else str(view),
                  "official_split": "test" if name in official_test else "train_val",
                  "split": partition.get(name, "unassigned_train_val" if name in official_train else "test")}
        try:
            path = lookup[name]
            stat = path.stat()
            old = known.get(name)
            if old and old.get("status") in ("ok", "insufficient_patches") and old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
                return old
            image, sha = read_image(path)
            record.update(relative_path=path.relative_to(data_root).as_posix(), sha256=sha,
                          width=image.shape[1], height=image.shape[0], size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                          **estimate(image))
        except (KeyError, OSError, ValueError, cv2.error) as error:
            record.update(status="error", error=str(error)[:160], noise_sigma=None)
        return record

    print(f"Audit: {len(items):,}/{len(metadata):,} images; {len(known):,} checkpointed; {workers} workers", flush=True)
    failed_batch = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), 256):
            if (output / "STOP").exists():
                print("Audit stopped at a durable batch boundary; checkpoint retained.", flush=True)
                break
            records = list(pool.map(inspect, items[start:start+256]))
            path = chunks / f"{start:06d}.json"
            value = {"signature": signature, "records": records}
            if not path.exists() or json.loads(path.read_text()) != value:
                write_json(path, value)
            known.update({r["image"]: r for r in records})
            done = min(start+256, len(items))
            errors = sum(row["status"] == "error" for row in records)
            print(f"Audit {done:,}/{len(items):,}; batch read {len(records)-errors}/{len(records)}, "
                  f"errors {errors}; elapsed {time.monotonic()-started:.1f}s", flush=True)
            if errors == len(records):
                failed_batch = True
                print("All reads in this batch failed; stopping. Examples: " +
                      str([{key: row.get(key) for key in ("image", "error")} for row in records[:3]]), flush=True)
                break
    frame = pd.DataFrame([known[name] for name in sorted(known)])
    if frame.empty:
        frame = pd.DataFrame(columns=["image", "status", "noise_sigma"])
    temporary = output / "noise_audit.csv.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(output / "noise_audit.csv")
    error_count = int((frame.status == "error").sum())
    summary = {"signature": signature, "algorithm": AUDIT_VERSION, "metadata_images": len(metadata),
               "requested_images": len(requested),
               "scope": "v3_zip_originals" if image_ids is not None else "full_metadata",
               "audited_images": len(frame), "complete": len(frame) == len(requested) and error_count == 0,
               "full_nih_complete": len(frame) == len(metadata) and error_count == 0,
               "readable_images": int(frame.status.isin(["ok", "insufficient_patches"]).sum()),
               "statuses": {str(k): int(v) for k, v in frame.status.value_counts().items()},
               "csv_sha256": file_hash(output / "noise_audit.csv"), "estimate_units": "intensity normalized to [0,1]",
               "interpretation": "Estimated noise proxy; low score does not prove noise-free anatomy"}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed_batch:
        raise OSError("All image reads failed in an audit batch. Check the Drive mount and permissions; "
                      "failed rows are checkpointed for retry and are not a completed audit.")
    return frame
