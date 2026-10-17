"""Freeze low-estimated-noise cohorts for VinBigData 512 (normal vs infiltration inclusive)."""
from pathlib import Path
import json
import shutil
import zipfile

import numpy as np
import pandas as pd

from .audit import read_image
from .release.experiment import file_hash, write_json

SPLITS = ("train", "validation", "test")
MANIFESTS = (*SPLITS, "xai", "background")
CLASS_NAMES = (0, "normal"), (1, "infiltration")


def validate_frames(frames, counts=None):
    for split in SPLITS:
        f = frames[split]
        if f.image.duplicated().any() or f.empty:
            raise ValueError(f"Invalid {split} identities")
        labels = f.label.to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Non-binary labels in {split}")
        normalized = f.finding_labels.astype(str).str.lower().to_numpy()
        expected_0 = np.isin(normalized[labels == 0], ["normal", "no finding"])
        expected_1 = np.isin(normalized[labels == 1], ["infiltration"])
        if not expected_0.all() or not expected_1.all():
            raise ValueError(f"Non-strict class names in {split}")
        if counts and f.label.value_counts().to_dict() != {0: counts[split], 1: counts[split]}:
            raise ValueError(f"Wrong {split} class counts")
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            if set(frames[a].patient_id.astype(str)) & set(frames[b].patient_id.astype(str)):
                raise ValueError(f"Patient/image leakage: {a}/{b}")
    if not set(frames["background"].image) <= set(frames["train"].image):
        raise ValueError("SHAP background is not train-only")
    background = frames["background"].merge(
        frames["train"][["image", "patient_id", "label"]],
        on="image", suffixes=("", "_train"), validate="one_to_one"
    )
    for field in ("patient_id", "label"):
        if not np.array_equal(background[field].astype(str), background[field + "_train"].astype(str)):
            raise ValueError("SHAP background metadata differs from train")
    if set(frames["xai"].patient_id.astype(str)) & (
        set(frames["train"].patient_id.astype(str)) | set(frames["validation"].patient_id.astype(str))
    ):
        raise ValueError("XAI images overlap train/validation")
    if frames["xai"].image.duplicated().any() or frames["background"].image.duplicated().any():
        raise ValueError("Duplicate XAI/background images")


def select_frames(audit, counts):
    frames = {}
    for split in SPLITS:
        candidates = audit[audit.split == split].copy()
        parts = []
        for label in (0, 1):
            valid = candidates[(candidates.label == label) & (candidates.status == "ok") & candidates.noise_sigma.notna()]
            valid = valid.sort_values(["noise_sigma", "image"], kind="stable")
            if len(valid) < counts[split]:
                raise ValueError(f"Insufficient assessable {split} class {label}: {len(valid)} < {counts[split]}")
            parts.append(valid.head(counts[split]))
        frame = pd.concat(parts).sort_values("image").reset_index(drop=True)
        frame["finding_labels"] = np.where(frame.label.to_numpy() == 0, "normal", "infiltration")
        base_cols = [
            "image", "patient_id", "label", "finding_labels", "view_position",
            "noise_sigma", "global_sigma", "contrast", "sharpness",
            "sha256", "width", "height", "relative_path",
        ]
        extra_cols = [c for c in ("raw_findings", "risk_flag", "bbox_x", "bbox_y", "bbox_w", "bbox_h") if c in frame.columns]
        frame = frame[base_cols + extra_cols].rename(columns={"relative_path": "source_relative_path"})
        frame["relative_path"] = "images/" + frame.image
        frames[split] = frame

    xai = frames["test"][frames["test"].label == 1].copy().reset_index(drop=True)
    for col, default in (("bbox_x", 64.0), ("bbox_y", 64.0), ("bbox_w", 384.0), ("bbox_h", 384.0), ("risk_flag", "mixed_high_risk")):
        if col not in xai.columns:
            xai[col] = default
        else:
            xai[col] = xai[col].fillna(default)
    frames["xai"] = xai

    frames["background"] = pd.concat([
        frames["train"][frames["train"].label == label].sample(
            n=min(16, counts["train"]), random_state=42 + label
        )
        for label in (0, 1)
    ]).sort_values("image").reset_index(drop=True)

    validate_frames(frames, counts)
    return frames


def figures(audit, frames, data_root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(output) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    valid = audit[audit.status == "ok"].sort_values(["noise_sigma", "image"])
    figure, axes = plt.subplots(2, 6, figsize=(15, 6))
    for label, finding in CLASS_NAMES:
        subset = valid[valid.label == label]
        if subset.empty:
            continue
        for ax, index in zip(axes[label], np.rint(np.linspace(0, len(subset) - 1, 6)).astype(int)):
            row = subset.iloc[index]
            image, _ = read_image(Path(data_root) / row.relative_path)
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{finding}\nsigma={row.noise_sigma:.4f}", fontsize=8)
            ax.axis("off")
    figure.tight_layout()
    figure.savefig(directory / "noise_quantiles.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, finding in CLASS_NAMES:
        values = valid.loc[valid.label == label, "noise_sigma"]
        axes[label].hist(values, bins=50, alpha=.6, density=True, label="eligible cohort")
        axes[label].hist(frames["train"].loc[frames["train"].label == label, "noise_sigma"], bins=20,
                         density=True, alpha=.6, label="selected train")
        axes[label].set(title=f"{finding} (class {label})", xlabel="estimated sigma [0,1]")
        axes[label].legend()
    figure.tight_layout()
    figure.savefig(directory / "noise_distribution.png", dpi=140)
    plt.close(figure)


def prepare(root, data_root, audit_dir, work_root=None):
    root, data_root, audit_dir = Path(root), Path(data_root).resolve(), Path(audit_dir)
    work_root = Path(work_root or root)
    if work_root.resolve().is_relative_to(data_root):
        raise ValueError("Prepared outputs must be outside the original dataset")
    config = json.loads((root / "configs/protocol.json").read_text())
    summary = json.loads((audit_dir / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Prepare requires a complete audit without read errors")
    if summary["csv_sha256"] != file_hash(audit_dir / "noise_audit.csv"):
        raise ValueError("Audit CSV integrity failed")

    audit = pd.read_csv(audit_dir / "noise_audit.csv")
    frames = select_frames(audit, config["counts_per_class"])

    frozen = work_root / "prepared/manifests"
    frozen.mkdir(parents=True, exist_ok=True)
    payloads = {name: frame.to_csv(index=False).encode("utf-8") for name, frame in frames.items()}
    for name, payload in payloads.items():
        path = frozen / f"{name}.csv"
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Selection changed; preserve the old prepared dataset and use a new output")

    destination = work_root / "selected_data"
    images = destination / "images"
    images.mkdir(parents=True, exist_ok=True)
    unique = pd.concat(frames.values()).drop_duplicates("image").sort_values("image")
    for row in unique.itertuples(index=False):
        source = (data_root / row.source_relative_path).resolve()
        if not source.is_relative_to(data_root) or file_hash(source) != row.sha256:
            raise ValueError(f"Source image changed since audit: {row.image}")
        target = images / row.image
        if not target.exists() or file_hash(target) != row.sha256:
            shutil.copyfile(source, target)

    for name, payload in payloads.items():
        (frozen / f"{name}.csv").write_bytes(payload)
        (destination / f"{name}.csv").write_bytes(payload)

    receipt = {
        "schema_version": 7,
        "dataset": config.get("dataset_slug", "awsaf49/vinbigdata-512-image-dataset"),
        "audit_signature": summary["signature"],
        "audit_scope": summary.get("scope", "vinbigdata_512_binary_cohort"),
        "audited_images": summary["audited_images"],
        "metadata_images": summary["metadata_images"],
        "audit_csv_sha256": summary["csv_sha256"],
        "selection": "lowest estimated sigma per class/split, tie=image",
        "class_names": {"0": "normal", "1": "infiltration"},
        "counts_per_class": config["counts_per_class"],
        "native_images": len(unique),
        "manifest_hashes": {name: file_hash(frozen / f"{name}.csv") for name in MANIFESTS},
    }
    write_json(destination / "dataset.json", receipt)
    write_json(work_root / "prepared/dataset.json", receipt)

    views = pd.concat([f.assign(split=s) for s, f in frames.items() if s in SPLITS])
    views.groupby(["split", "label", "finding_labels"]).size().rename("n").to_csv(work_root / "prepared/view_counts.csv")
    audit.groupby(["split", "finding_labels"]).size().rename("n").to_csv(work_root / "prepared/source_view_counts.csv")
    figures(audit, frames, data_root, work_root / "prepared")

    package = work_root / "v7_selected_data.zip"
    temporary = package.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, "v7_data/" + path.relative_to(destination).as_posix())
    temporary.replace(package)
    write_json(work_root / "selected_package.json", {
        "sha256": file_hash(package),
        "dataset_receipt_sha256": file_hash(destination / "dataset.json"),
        "native_images": len(unique),
        "package": package.name,
    })
    c = config["counts_per_class"]
    print(
        f"Prepared {len(unique)} native VinBigData images (normal vs infiltration); "
        f"train {c['train']}+{c['train']}, validation {c['validation']}+{c['validation']}, test {c['test']}+{c['test']}; {package}",
        flush=True,
    )
    return destination
