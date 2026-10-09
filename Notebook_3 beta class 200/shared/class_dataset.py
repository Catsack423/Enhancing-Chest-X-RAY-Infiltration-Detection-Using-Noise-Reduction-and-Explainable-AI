"""Prepare 200/class once while preserving the original patient partition."""
from copy import deepcopy
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from .data import balanced_sample, load_metadata, verify_manifests
from .data_bundle import build_data_bundle
from .experiment import Experiment, digest, file_hash, write_json

NAMES = ("train", "validation", "test", "xai", "background")


def prepare_class_dataset(root, data_root):
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    config = json.loads((root / "shared/config.json").read_text(encoding="utf-8"))
    if config["data"]["train_per_class"] != 200 or config["training"]["images_per_epoch"] != 400:
        raise ValueError("This variant fixes 200 images per class and 400 images per epoch")
    reference = root / "reference_run/shared/artifacts"
    recorded = json.loads((reference / "experiment.json").read_text(encoding="utf-8"))
    if digest(recorded["config"]) != recorded["config_hash"]:
        raise ValueError("Original configuration changed")
    old_config = recorded["config"]
    if config["seed"] != old_config["seed"] or config["preprocessing"] != old_config["preprocessing"]:
        raise ValueError("Preserve the original seed and preprocessing")
    signature = json.loads((reference / "manifests/source.json").read_text(encoding="utf-8"))
    if signature["config_hash"] != recorded["config_hash"]:
        raise ValueError("Original manifest signature changed")
    for name, expected in signature["sources"].items():
        if file_hash(data_root / name) != expected:
            raise ValueError(f"Original NIH source changed: {name}")
    for name, expected in recorded["manifest_hashes"].items():
        if file_hash(reference / "manifests" / name) != expected:
            raise ValueError(f"Original manifest changed: {name}")
    frames = {name: pd.read_csv(reference / "manifests" / (name + ".csv")) for name in NAMES}
    old_train = frames["train"]
    if old_train.label.value_counts().to_dict() != {0: 160, 1: 160}:
        raise ValueError("Expected the original balanced 320-image training set")
    source_config = deepcopy(config)
    source_config["run_mode"] = "full"
    source_exp = Experiment(root, data_root, root / "source_run", source_config)
    meta, official = load_metadata(source_exp)
    pool = meta[meta["Image Index"].isin(official["train_val"]) &
                meta["Finding Labels"].isin(["No Finding", "Infiltration"])].copy()
    # Reconstruct v2's reserved validation-patient partition. Extra images must
    # not come from unused patients in this original validation partition.
    rng = np.random.default_rng(old_config["seed"])
    ordered = rng.permutation(sorted(pool["Patient ID"].unique()))
    n_val = int(np.ceil(len(ordered) * old_config["data"]["validation_patient_fraction"]))
    reserved_val = set(ordered[:n_val])
    if set(old_train.patient_id) & reserved_val:
        raise ValueError("Could not reproduce the original training-patient partition")
    if not set(frames["validation"].patient_id) <= reserved_val:
        raise ValueError("Could not reproduce the original validation-patient partition")
    candidates = pool[~pool["Patient ID"].isin(reserved_val) &
                      ~pool["Image Index"].isin(old_train.image)].rename(columns={
        "Image Index": "image", "Patient ID": "patient_id", "Finding Labels": "finding_labels"})
    candidates["label"] = (candidates.finding_labels == "Infiltration").astype(int)
    settings = config["training_data_change"]
    added = balanced_sample(candidates, 40, settings["selection_seed"], settings["selection_stream"])
    directories = sorted(data_root.glob("images_*/images"))
    relative_paths = []
    for name in added.image:
        matches = [directory / name for directory in directories if (directory / name).is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Need exactly one source for selected image {name}; found {len(matches)}")
        relative_paths.append(matches[0].relative_to(data_root).as_posix())
    added["relative_path"] = relative_paths
    added = added[list(old_train.columns)].sort_values("image").reset_index(drop=True)
    frames["train"] = pd.concat([old_train, added], ignore_index=True).sort_values("image").reset_index(drop=True)
    if len(frames["train"]) != 400 or not set(old_train.image) <= set(frames["train"].image):
        raise ValueError("Expansion must retain every old training image")
    verify_manifests(source_exp, frames)
    folder = source_exp.artifacts / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    train_path = folder / "train.csv"
    if train_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(train_path), frames["train"])
    else:
        frames["train"].to_csv(train_path, index=False)
    for name in NAMES[1:]:
        source, target = reference / "manifests" / (name + ".csv"), folder / (name + ".csv")
        if target.exists() and file_hash(target) != file_hash(source):
            raise ValueError(f"Refusing to replace a different {name} manifest")
        if not target.exists():
            shutil.copyfile(source, target)
    write_json(folder / "source.json", {"config_hash": digest(source_config), "sources": signature["sources"]})
    source_exp.lock()
    added.to_csv(source_exp.artifacts / "added_train_images.csv", index=False)
    selection = {"reference_config_hash": recorded["config_hash"],
                 "reference_manifest_hashes": recorded["manifest_hashes"],
                 "new_manifest_hashes": source_exp.provenance()["manifest_hashes"],
                 "retained_train_images": 320, "added_train_images": 80,
                 "seed": settings["selection_seed"], "stream": settings["selection_stream"],
                 "reserved_validation_patient_count": len(reserved_val),
                 "train_patients": int(frames["train"].patient_id.nunique()),
                 "counts": {name: len(frame) for name, frame in frames.items()},
                 "policy": settings["policy"], "official_split_unchanged": True,
                 "patient_disjoint": True, "image_availability_used_for_selection": False}
    write_json(source_exp.artifacts / "dataset_selection.json", selection)
    return source_exp, frames, selection


def prepare_and_pack(root, data_root):
    exp, frames, selection = prepare_class_dataset(root, data_root)
    report = build_data_bundle(exp, frames, exp.root / "nih_cxr_subset.zip")
    return {"selection": selection, "bundle_images": report["image_count"],
            "bundle_bytes": (exp.root / "nih_cxr_subset.zip").stat().st_size}
