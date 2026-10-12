"""Frozen manifest consumption. Never select a new patient partition on Drive."""
from pathlib import Path, PurePosixPath
import json
import shutil

import numpy as np
import pandas as pd

from .experiment import digest, file_hash, write_json

MANIFEST_NAMES = ("train", "validation", "test", "xai", "background")
SOURCE_NAMES = ("Data_Entry_2017.csv", "BBox_List_2017.csv", "train_val_list.txt", "test_list.txt")
FULL_COUNTS = {"train": {0: 40371, 1: 5737}, "validation": {0: 10129, 1: 1590},
               "test": {0: 9861, 1: 2220}, "background": {0: 16, 1: 16}}


def source_hashes(exp):
    hashes = {name: file_hash(exp.data_root / name) for name in SOURCE_NAMES}
    expected = exp.config["data"]["source_hashes"]
    if hashes != expected:
        changed = [name for name in SOURCE_NAMES if hashes[name] != expected.get(name)]
        raise ValueError(f"NIH metadata/split checksums changed: {changed}")
    return hashes


def load_metadata(exp):
    source_hashes(exp)
    meta = pd.read_csv(exp.data_root / "Data_Entry_2017.csv", usecols=["Image Index", "Finding Labels", "Patient ID"])
    meta["Patient ID"] = meta["Patient ID"].astype(int)
    if meta["Image Index"].duplicated().any():
        raise ValueError("Duplicate metadata image IDs")
    official = {key: set((exp.data_root / filename).read_text().splitlines()) for key, filename in
                (("train_val", "train_val_list.txt"), ("test", "test_list.txt"))}
    if official["train_val"] & official["test"]:
        raise ValueError("Official split images overlap")
    if any(not ids <= set(meta["Image Index"]) for ids in official.values()):
        raise ValueError("Official split contains IDs missing from metadata")
    patients = {key: set(meta.loc[meta["Image Index"].isin(ids), "Patient ID"]) for key, ids in official.items()}
    if patients["train_val"] & patients["test"]:
        raise ValueError("Official split patients overlap")
    return meta, official


def validate_relative_path(row):
    path = PurePosixPath(str(row["relative_path"]).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or len(path.parts) not in (2, 3):
        raise ValueError(f"Unsafe manifest image path: {path}")
    if not path.parts[0].startswith("images_") or path.name != row["image"]:
        raise ValueError(f"Image path/ID mismatch: {path}")
    if len(path.parts) == 3 and path.parts[1] != "images":
        raise ValueError(f"Unsupported image layout: {path}")
    return path


def resolve_image_path(exp, row):
    """Try the two documented layouts inside the exact recorded shard only."""
    relative = validate_relative_path(row)
    candidates = [exp.data_root / relative.as_posix(),
                  exp.data_root / relative.parts[0] / relative.name,
                  exp.data_root / relative.parts[0] / "images" / relative.name]
    for path in dict.fromkeys(candidates):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing selected image {row['image']}; checked {[str(p) for p in dict.fromkeys(candidates)]}")


def balanced_sample(frame, per_class=2):
    # Smoke is a deterministic slice of frozen IDs, independent of training seed.
    groups = [frame.loc[frame.label == label].sort_values("image").head(per_class) for label in (0, 1)]
    if any(len(group) != per_class for group in groups):
        raise ValueError("Insufficient frozen images for smoke sample")
    return pd.concat(groups).sort_values("image").reset_index(drop=True)


def _smoke_frames(frames):
    selected = {name: balanced_sample(frames[name]) for name in ("validation", "test")}
    background = balanced_sample(frames["background"], 1)
    # Retain exactly the selected background in smoke training, then one more/class.
    extras = balanced_sample(frames["train"].loc[~frames["train"].image.isin(background.image)], 1)
    selected["train"] = pd.concat([background, extras]).sort_values("image").reset_index(drop=True)
    selected["background"] = background
    selected["xai"] = frames["xai"].sort_values("image").head(2).reset_index(drop=True)
    return {name: selected[name] for name in MANIFEST_NAMES}


def verify_manifests(exp, frames):
    if set(frames) != set(MANIFEST_NAMES):
        raise ValueError("All five manifests are required")
    meta, official = load_metadata(exp)
    metadata = meta.set_index("Image Index")
    for name, frame in frames.items():
        if frame.empty or frame.image.duplicated().any():
            raise ValueError(f"Empty/duplicate image IDs in {name}")
        allowed = official["train_val" if name in ("train", "validation", "background") else "test"]
        if not set(frame.image) <= allowed:
            raise ValueError(f"Wrong official split in {name}")
        for row in frame.to_dict("records"):
            record = metadata.loc[row["image"]]
            if int(row["patient_id"]) != record["Patient ID"]:
                raise ValueError(f"Wrong patient for {row['image']}")
            if row["finding_labels"] != record["Finding Labels"]:
                raise ValueError(f"Finding label changed for {row['image']}")
            if row["label"] not in (0, 1):
                raise ValueError("Invalid binary label")
            expected = "Infiltration" if row["label"] == 1 else "No Finding"
            if name != "xai" and row["finding_labels"] != expected:
                raise ValueError(f"Invalid pure-only binary label for {row['image']}")
            validate_relative_path(row)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"),
                        ("train", "xai"), ("validation", "xai")):
        if set(frames[left].patient_id) & set(frames[right].patient_id):
            raise ValueError(f"Patient leakage: {left}/{right}")
    background = frames["background"]
    if not set(background.image) <= set(frames["train"].image):
        raise ValueError("SHAP backgrounds must be a training subset")
    smoke = exp.config.get("run_mode") == "smoke"
    expected_lengths = exp.config["data"].get("expected_counts", {})
    if any(len(frames[name]) != int(value) for name, value in expected_lengths.items()):
        raise ValueError("Manifest row counts differ from configured release")
    counts = {name: {0: 2, 1: 2} for name in ("train", "validation", "test")} if smoke else FULL_COUNTS.copy()
    counts["background"] = {0: 1, 1: 1} if smoke else {0: 16, 1: 16}
    overrides = exp.config["data"].get("expected_class_counts", {}) if not smoke else {}
    for name, expected in counts.items():
        expected = {int(k): int(v) for k, v in overrides.get(name, expected).items()}
        if frames[name].label.value_counts().to_dict() != expected:
            raise ValueError(f"Unexpected class counts in {name}")
    boxes = pd.read_csv(exp.data_root / "BBox_List_2017.csv").iloc[:, :6]
    boxes.columns = ["image", "finding", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    infiltration = boxes.loc[boxes.finding == "Infiltrate"].set_index("image")
    xai = frames["xai"]
    if len(xai) != (2 if smoke else 123) or (not smoke and xai.patient_id.nunique() != 115):
        raise ValueError("XAI size/patients differ from frozen release")
    if not smoke and set(xai.image) != set(infiltration.index):
        raise ValueError("XAI IDs differ from original infiltration bounding boxes")
    for row in xai.to_dict("records"):
        if row["label"] != 1 or "Infiltration" not in row["finding_labels"].split("|"):
            raise ValueError("Invalid XAI positive label")
        labels = set(row["finding_labels"].split("|"))
        expected_risk = "pure" if labels == {"Infiltration"} else "mixed_high_risk" if labels & {"Effusion", "Atelectasis"} else "mixed_low_risk"
        if row["risk_flag"] != expected_risk:
            raise ValueError("XAI risk stratum changed")
        for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            if row["image"] not in infiltration.index or not np.isclose(row[key], float(infiltration.loc[row["image"], key])):
                raise ValueError("XAI bounding box changed")
        if row["bbox_w"] <= 0 or row["bbox_h"] <= 0:
            raise ValueError("Invalid bounding box dimensions")
    # Compare against the bundled partition, without reconstructing or randomizing it.
    for name in MANIFEST_NAMES:
        bundled = pd.read_csv(exp.root / "manifests" / f"{name}.csv")
        if not set(frames[name].image) <= set(bundled.image):
            raise ValueError(f"Image left frozen {name} partition")
        if not smoke and set(frames[name].image) != set(bundled.image):
            raise ValueError(f"Frozen {name} partition changed")
    return True


def prepare_manifests(exp):
    """Copy checked bundled manifests; fail closed on incompatible resume."""
    bundled = exp.root / "manifests"
    hashes = {f"{name}.csv": file_hash(bundled / f"{name}.csv") for name in MANIFEST_NAMES}
    expected = exp.config["data"].get("manifest_hashes")
    if expected is None and (bundled / "source_hashes.json").exists():
        frozen = json.loads((bundled / "source_hashes.json").read_text())
        expected = frozen["manifests"]
        if frozen["sources"] != exp.config["data"]["source_hashes"]:
            raise ValueError("Bundled source checksums differ from configuration")
    if expected and hashes != expected:
        raise ValueError("Bundled manifest checksums changed")
    sources = source_hashes(exp)
    frames = {name: pd.read_csv(bundled / f"{name}.csv") for name in MANIFEST_NAMES}
    if exp.config.get("run_mode") == "smoke":
        frames = _smoke_frames(frames)
    verify_manifests(exp, frames)
    folder = exp.artifacts / "manifests"
    signature = {"config_hash": digest(exp.config), "sources": sources, "bundled_manifest_hashes": hashes,
                 "run_mode": exp.config.get("run_mode", "full")}
    marker = folder / "source.json"
    if marker.exists():
        if json.loads(marker.read_text()) != signature:
            raise ValueError("Manifest/config/source provenance changed; use another output directory")
        saved = {name: pd.read_csv(folder / f"{name}.csv") for name in MANIFEST_NAMES}
        for name in MANIFEST_NAMES:
            pd.testing.assert_frame_equal(saved[name], frames[name], check_dtype=False)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            path = folder / f"{name}.csv"
            if exp.config.get("run_mode") != "smoke":
                if path.exists() and file_hash(path) != hashes[path.name]:
                    raise ValueError(f"Partial manifest differs: {path.name}")
                shutil.copyfile(bundled / path.name, path)
            else:
                temporary = path.with_suffix(".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(path)
        write_json(marker, signature)
    exp.lock()
    return frames
