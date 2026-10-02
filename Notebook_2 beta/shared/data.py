"""Official split filtering, patient-disjoint sampling, and manifest verification."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .experiment import digest, file_hash, write_json


def _source_hashes(exp):
    return {name: file_hash(exp.data_root / name) for name in (
        "Data_Entry_2017.csv", "BBox_List_2017.csv", "train_val_list.txt", "test_list.txt")}


def load_metadata(exp):
    meta = pd.read_csv(exp.data_root / "Data_Entry_2017.csv", usecols=["Image Index", "Finding Labels", "Patient ID"])
    meta["Patient ID"] = meta["Patient ID"].astype(int)
    if meta["Image Index"].duplicated().any():
        raise ValueError("Duplicate metadata image IDs")
    lists = {name: set((exp.data_root / filename).read_text().splitlines()) for name, filename in (
        ("train_val", "train_val_list.txt"), ("test", "test_list.txt"))}
    if lists["train_val"] & lists["test"]:
        raise ValueError("Official split lists overlap")
    patients = {name: set(meta.loc[meta["Image Index"].isin(ids), "Patient ID"]) for name, ids in lists.items()}
    if patients["train_val"] & patients["test"]:
        raise ValueError("Official splits contain overlapping patients")
    return meta, lists


def balanced_sample(frame, per_class, seed, stream=0):
    """Deterministic IDs, independent of condition, file availability, and prior RNG use."""
    if per_class < 1:
        raise ValueError("Balanced sample size must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    selected = []
    for label in (0, 1):
        group = frame[frame.label == label].sort_values("image")
        if len(group) < per_class:
            raise ValueError(f"Insufficient class {label} for balanced subset")
        selected.append(group.iloc[rng.permutation(len(group))[:per_class]])
    return pd.concat(selected).sort_values("image").reset_index(drop=True)


def verify_manifests(exp, manifests):
    meta, official = load_metadata(exp)
    labels = meta.set_index("Image Index")["Finding Labels"].to_dict()
    patient_ids = meta.set_index("Image Index")["Patient ID"].to_dict()
    for name, frame in manifests.items():
        if frame["image"].duplicated().any():
            raise ValueError(f"Duplicate images in {name}")
        allowed = official["train_val" if name in ("train", "validation", "background") else "test"]
        if not set(frame.image) <= allowed:
            raise ValueError(f"Wrong official split in {name}")
        for row in frame.itertuples(index=False):
            if patient_ids[row.image] != row.patient_id:
                raise ValueError(f"Wrong patient for {row.image}")
            if name != "xai" and labels[row.image] != ("Infiltration" if row.label == 1 else "No Finding"):
                raise ValueError(f"Invalid binary label for {row.image}")
            if not (exp.data_root / row.relative_path).is_file():
                raise FileNotFoundError(row.relative_path)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"), ("train", "xai"), ("validation", "xai")):
        if set(manifests[left].patient_id) & set(manifests[right].patient_id):
            raise ValueError(f"Patient leakage: {left}/{right}")
        if set(manifests[left].image) & set(manifests[right].image):
            raise ValueError(f"Image leakage: {left}/{right}")
    for name, count in (("train", exp.config["data"]["train_per_class"]), ("validation", exp.config["data"]["validation_per_class"]), ("background", exp.config["xai"]["background_per_class"])):
        if manifests[name].label.value_counts().to_dict() != {0: count, 1: count}:
            raise ValueError(f"Unexpected balanced counts: {name}")
    if not set(manifests["background"].image) <= set(manifests["train"].image):
        raise ValueError("SHAP backgrounds must come from training only")
    if exp.config["run_mode"] == "full":
        if len(manifests["test"]) != exp.config["data"]["expected_test_images"]:
            raise ValueError("Official filtered test count changed")
        xai = manifests["xai"]
        if len(xai) != exp.config["data"]["expected_xai_images"] or xai.patient_id.nunique() != exp.config["data"]["expected_xai_patients"]:
            raise ValueError("XAI set must be the original 123 images / 115 patients")
        official_binary = meta.loc[meta["Image Index"].isin(official["test"]) & meta["Finding Labels"].isin(["Infiltration", "No Finding"])].rename(columns={"Image Index": "image"}).copy()
        if len(official_binary) != exp.config["data"].get("official_test_images", 12081):
            raise ValueError("Official binary test population changed")
        official_binary["label"] = (official_binary["Finding Labels"] == "Infiltration").astype(int)
        if exp.config["data"].get("test_scope") == "official_balanced_subset":
            official_binary = balanced_sample(official_binary, exp.config["data"]["test_per_class"], exp.config["seed"], stream=1)
        expected_test = set(official_binary.image)
        if set(manifests["test"].image) != expected_test:
            raise ValueError("Test manifest does not match the configured official test selection")
    boxes = pd.read_csv(exp.data_root / "BBox_List_2017.csv").iloc[:, :6]
    boxes.columns = ["image", "finding", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    infiltration = boxes[boxes.finding == "Infiltrate"].set_index("image")
    if exp.config["run_mode"] == "full" and set(manifests["xai"].image) != set(infiltration.index):
        raise ValueError("XAI image IDs changed")
    for r in manifests["xai"].itertuples(index=False):
        for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            if not np.isclose(getattr(r, key), float(infiltration.loc[r.image, key])):
                raise ValueError("XAI bounding box changed")


def prepare_manifests(exp):
    folder = exp.artifacts / "manifests"
    source_hashes = _source_hashes(exp)
    signature = {"config_hash": digest(exp.config), "sources": source_hashes}
    marker = folder / "source.json"
    names = ("train", "validation", "test", "xai", "background")
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != signature:
            raise ValueError("Configuration or source files changed; choose a new OUTPUT_ROOT")
        manifests = {name: pd.read_csv(folder / f"{name}.csv") for name in names}
        verify_manifests(exp, manifests)
        exp.lock()
        return manifests
    if folder.exists() and list(folder.glob("*.csv")):
        raise RuntimeError("Incomplete manifest creation; use a new OUTPUT_ROOT")
    meta, official = load_metadata(exp)
    paths = {}
    for p in sorted(exp.data_root.glob("images_*/images/*.png")):
        if p.name in paths:
            raise ValueError(f"Duplicate source image: {p.name}")
        paths[p.name] = p.relative_to(exp.data_root).as_posix()
    frame = meta.rename(columns={"Image Index": "image", "Patient ID": "patient_id", "Finding Labels": "finding_labels"})
    frame["relative_path"] = frame.image.map(paths)
    usable = frame[frame.finding_labels.isin(["Infiltration", "No Finding"])].copy()
    usable["label"] = (usable.finding_labels == "Infiltration").astype(int)
    columns = ["image", "patient_id", "relative_path", "label", "finding_labels"]
    pool = usable[usable.image.isin(official["train_val"])].sort_values("image")
    rng = np.random.default_rng(exp.config["seed"])
    patients = rng.permutation(sorted(pool.patient_id.unique()))
    n_val = int(np.ceil(len(patients) * exp.config["data"]["validation_patient_fraction"]))
    val_patients = set(patients[:n_val])
    def balanced(candidates, count):
        chosen = []
        for label in (0, 1):
            group = candidates[candidates.label == label].sort_values("image")
            if len(group) < count:
                raise ValueError(f"Insufficient class {label}: {len(group)} < {count}")
            chosen.append(group.iloc[rng.permutation(len(group))[:count]])
        return pd.concat(chosen)[columns].sort_values("image").reset_index(drop=True)
    manifests = {
        "train": balanced(pool[~pool.patient_id.isin(val_patients)], exp.config["data"]["train_per_class"]),
        "validation": balanced(pool[pool.patient_id.isin(val_patients)], exp.config["data"]["validation_per_class"]),
        "test": usable[usable.image.isin(official["test"])][columns].sort_values("image").reset_index(drop=True),
    }
    bbox = pd.read_csv(exp.data_root / "BBox_List_2017.csv").iloc[:, :6]
    bbox.columns = ["image", "finding", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    xai = bbox[bbox.finding == "Infiltrate"].merge(frame, on="image", validate="one_to_one")
    xai["label"] = 1
    def risk(labels):
        labels = set(labels.split("|"))
        return "pure" if labels == {"Infiltration"} else "mixed_high_risk" if labels & {"Effusion", "Atelectasis"} else "mixed_low_risk"
    xai["risk_flag"] = xai.finding_labels.map(risk)
    manifests["xai"] = xai[columns + ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "risk_flag"]].sort_values("image").reset_index(drop=True)
    manifests["background"] = balanced(manifests["train"], exp.config["xai"]["background_per_class"])
    if exp.config["data"].get("test_scope") == "official_balanced_subset":
        manifests["test"] = balanced_sample(manifests["test"], exp.config["data"]["test_per_class"], exp.config["seed"], stream=1)
    if exp.config["run_mode"] == "smoke":
        if exp.config["data"].get("test_scope") != "official_balanced_subset":
            manifests["test"] = balanced(manifests["test"], 2)
        manifests["xai"] = manifests["xai"].head(2)
    if any(f.relative_path.isna().any() for f in manifests.values()):
        raise FileNotFoundError("Some selected NIH images are missing")
    verify_manifests(exp, manifests)
    folder.mkdir(parents=True, exist_ok=True)
    for name, subset in manifests.items():
        subset.to_csv(folder / f"{name}.csv", index=False)
    write_json(marker, signature)
    exp.lock()
    return manifests
