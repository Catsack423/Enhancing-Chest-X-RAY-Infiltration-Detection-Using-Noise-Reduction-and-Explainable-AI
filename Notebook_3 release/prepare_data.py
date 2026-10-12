"""Build lightweight frozen manifests locally; on Colab verify and cache Drive data."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def build_manifests(data_root):
    from shared.experiment import file_hash, write_json
    data_root = Path(data_root).resolve()
    if str(data_root).replace("\\", "/").startswith("/content/drive/"):
        raise ValueError("Use the bundled manifests on Colab; local rebuild does not scan mounted Drive")
    config = json.loads((ROOT / "configs/release_full.json").read_text(encoding="utf-8"))
    sources = config["data"]["source_hashes"]
    for name, expected in sources.items():
        if file_hash(data_root / name) != expected:
            raise ValueError(f"NIH metadata/split changed: {name}")
    meta = pd.read_csv(data_root / "Data_Entry_2017.csv", usecols=["Image Index", "Finding Labels", "Patient ID"])
    if meta["Image Index"].duplicated().any():
        raise ValueError("Duplicate NIH image IDs")
    official = {part: set((data_root / filename).read_text().splitlines())
                for part, filename in (("train_val", "train_val_list.txt"), ("test", "test_list.txt"))}
    if official["train_val"] & official["test"]:
        raise ValueError("Official splits overlap")
    frame = meta.rename(columns={"Image Index": "image", "Finding Labels": "finding_labels", "Patient ID": "patient_id"})
    usable = frame[frame.finding_labels.isin(["No Finding", "Infiltration"])].copy()
    usable["label"] = (usable.finding_labels == "Infiltration").astype(int)
    pool = usable[usable.image.isin(official["train_val"])].sort_values("image")
    ordered = np.random.default_rng(config["dataset_seed"]).permutation(sorted(pool.patient_id.unique()))
    reserved = set(ordered[:int(np.ceil(len(ordered) * config["data"]["validation_patient_fraction"]))])
    if len(reserved) != 4815:
        raise ValueError("Original reserved validation partition changed")
    selected = {
        "train": pool[~pool.patient_id.isin(reserved)].copy(),
        "validation": pool[pool.patient_id.isin(reserved)].copy(),
        "test": usable[usable.image.isin(official["test"])].copy(),
    }
    # This inventory is produced locally once and bundled as relative paths.
    paths = {}
    for directory in sorted(data_root.glob("images_*")):
        candidates = [directory / "images"] if (directory / "images").is_dir() else [directory]
        for images in candidates:
            for p in images.glob("*.png"):
                if p.name in paths:
                    raise ValueError(f"Duplicate image path: {p.name}")
                paths[p.name] = p.relative_to(data_root).as_posix()
    references = json.loads((ROOT / "configs/legacy_reference_ids.json").read_text())
    output = ROOT / "manifests"
    output.mkdir(exist_ok=True)
    columns = ["image", "patient_id", "relative_path", "label", "finding_labels"]
    audit = {"dataset_seed": config["dataset_seed"], "reserved_validation_patients": len(reserved), "sets": {}}
    for part, data in selected.items():
        if len(data) != config["data"]["expected_counts"][part]:
            raise ValueError(f"Unexpected {part} count")
        if not set(references[part]) <= set(data.image):
            raise ValueError(f"Lost original pilot images: {part}")
        data["relative_path"] = data.image.map(paths)
        if data.relative_path.isna().any():
            raise FileNotFoundError(f"Missing selected images: {data.loc[data.relative_path.isna(), 'image'].tolist()[:20]}")
        data = data[columns].sort_values("image").reset_index(drop=True)
        data.to_csv(output / f"{part}.csv", index=False)
        audit["sets"][part] = {"images": len(data), "patients": int(data.patient_id.nunique()),
                                "class_counts": {str(k): int(v) for k, v in data.label.value_counts().items()}}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(selected[left].patient_id) & set(selected[right].patient_id):
            raise ValueError(f"Patient leakage: {left}/{right}")
    manifest_hashes = {p.name: file_hash(p) for p in sorted(output.glob("*.csv"))}
    if set(manifest_hashes) != {f"{name}.csv" for name in ("train", "validation", "test", "xai", "background")}:
        raise ValueError("Five frozen manifests are required")
    write_json(output / "source_hashes.json", {"sources": sources, "manifests": manifest_hashes,
                                               "expected_counts": config["data"]["expected_counts"], "dataset_seed": 42})
    write_json(output / "dataset_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--build-manifests", action="store_true", help="Local reproducible rebuild; no images are packed")
    parser.add_argument("--cache", action="store_true", help="Verify Drive/local sources and prepare temporary working cache")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.build_manifests:
        if args.data_root is None:
            parser.error("--build-manifests requires --data-root")
        build_manifests(args.data_root)
    else:
        from pipeline import open_run
        exp, frames = open_run(data_root=args.data_root, smoke=args.smoke, prepare_cache=args.cache)
        print({name: len(frame) for name, frame in frames.items()})
        print("Verified data:", exp.data_root, "| results:", exp.output_root)


if __name__ == "__main__":
    main()
