"""Resumable all-image noise audit for VinBigData 512 (normal vs infiltration inclusive)."""
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
SPLITS = ("train", "validation", "test")


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


def resolve_csv_path(data_root):
    root = Path(data_root).resolve()
    candidates = [
        root / "vinbigdata/train.csv",
        root / "train.csv",
        root / "Data_Entry_2017.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    found = sorted(root.rglob("train.csv"))
    if found:
        return found[0]
    raise FileNotFoundError(f"Could not find VinBigData train.csv under {root}")


def image_index(data_root):
    root = Path(data_root).resolve()
    result = {}
    patterns = (
        "vinbigdata/train/*.png", "vinbigdata/train/*.jpg",
        "train/*.png", "train/*.jpg",
        "images/*.png", "images_*/*.png", "images_*/images/*.png",
    )
    paths = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    if not paths:
        paths.update(p for p in root.rglob("*.png") if "test" not in p.parts)
    for path in sorted(paths):
        if not path.resolve().is_relative_to(root):
            raise ValueError("Image path escapes the dataset")
        if path.name not in result:
            result[path.name] = path
        if path.stem not in result:
            result[path.stem] = path
    return result


def read_image(path):
    raw = Path(path).read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image decoding failed")
    from hashlib import sha256
    return image, sha256(raw).hexdigest()


def assign_splits(records, seed=42, split_ratios=None, partition_root=None):
    if partition_root is not None:
        partition_root = Path(partition_root)
        if all((partition_root / f"{s}.csv").is_file() for s in SPLITS):
            mapping = {}
            for split in SPLITS:
                frame = pd.read_csv(partition_root / f"{split}.csv")
                for img in frame.image.astype(str):
                    mapping[img] = split
                    mapping[Path(img).stem] = split
            for r in records:
                r["split"] = mapping.get(r["image"], mapping.get(Path(r["image"]).stem, "unassigned"))
            return records

    ratios = split_ratios or {"train": 0.64, "validation": 0.16, "test": 0.20}
    by_label = {0: [], 1: []}
    for r in sorted(records, key=lambda x: x["image"]):
        if r["label"] in by_label:
            by_label[r["label"]].append(r)

    for label, group in by_label.items():
        n = len(group)
        if n == 0:
            continue
        rng = np.random.default_rng(int(seed) + int(label))
        order = rng.permutation(n)
        n_train = max(1, int(round(n * ratios["train"])))
        n_val = max(1, int(round(n * ratios["validation"])))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = max(1, (n - n_train) // 2)
        for rank, idx in enumerate(order):
            if rank < n_train:
                group[idx]["split"] = "train"
            elif rank < n_train + n_val:
                group[idx]["split"] = "validation"
            else:
                group[idx]["split"] = "test"
    return records


def extract_binary_cohort(df, seed=42, split_ratios=None, partition_root=None):
    """Build 2-class cohort from VinBigData train.csv:
    - label 0 ('normal'): normal lungs ('No finding' / class_id 14 only)
    - label 1 ('infiltration'): any image with 'Infiltration' (class_id 6), even with co-existing findings.
    """
    if "image_id" in df.columns:
        id_col = "image_id"
    elif "Image Index" in df.columns:
        id_col = "Image Index"
    elif "image" in df.columns:
        id_col = "image"
    else:
        raise ValueError(f"Unsupported metadata columns: {list(df.columns)}")

    has_class_name = "class_name" in df.columns
    has_class_id = "class_id" in df.columns
    has_finding_labels = "Finding Labels" in df.columns

    records = []
    for raw_id, group in df.groupby(id_col, sort=True):
        raw_id_str = str(raw_id)
        image_name = raw_id_str if raw_id_str.lower().endswith((".png", ".jpg")) else f"{raw_id_str}.png"
        stem = Path(image_name).stem

        findings = set()
        infiltration_rows = []
        for row in group.to_dict("records"):
            cname = str(row.get("class_name", "")).strip() if has_class_name else ""
            cid = row.get("class_id", None) if has_class_id else None
            flabels = str(row.get("Finding Labels", "")).strip() if has_finding_labels else ""

            if cname:
                for part in cname.split("|"):
                    findings.add(part.strip())
            elif flabels:
                for part in flabels.split("|"):
                    findings.add(part.strip())
            elif cid is not None and pd.notna(cid):
                cid_int = int(cid)
                if cid_int == 6:
                    findings.add("Infiltration")
                elif cid_int == 14:
                    findings.add("No finding")
                else:
                    findings.add(f"class_{cid_int}")

            is_inf = (cname.lower() == "infiltration") or (cid is not None and pd.notna(cid) and int(cid) == 6) or ("infiltration" in flabels.lower())
            if is_inf:
                infiltration_rows.append(row)

        lower_findings = {f.lower() for f in findings if f}
        has_infiltration = "infiltration" in lower_findings
        is_normal_only = lower_findings in ({"no finding"}, {"normal"})

        if not has_infiltration and not is_normal_only:
            continue

        label = 1 if has_infiltration else 0
        finding_label = "infiltration" if label == 1 else "normal"
        raw_findings_str = "|".join(sorted(findings)) if findings else finding_label
        other_diseases = lower_findings - {"infiltration", "no finding", "normal"}
        risk_flag = ("pure" if not other_diseases else "mixed_high_risk") if label == 1 else "normal"

        bbox_x, bbox_y, bbox_w, bbox_h = None, None, None, None
        orig_w, orig_h = None, None
        if label == 1:
            for r in infiltration_rows:
                if all(k in r and pd.notna(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")):
                    x0, y0, x1, y1 = float(r["x_min"]), float(r["y_min"]), float(r["x_max"]), float(r["y_max"])
                    if x1 > x0 and y1 > y0:
                        bbox_x, bbox_y, bbox_w, bbox_h = x0, y0, x1 - x0, y1 - y0
                        if "width" in r and "height" in r and pd.notna(r["width"]) and pd.notna(r["height"]):
                            orig_w, orig_h = float(r["width"]), float(r["height"])
                        break
                elif all(k in r and pd.notna(r[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h")):
                    bbox_x, bbox_y, bbox_w, bbox_h = float(r["bbox_x"]), float(r["bbox_y"]), float(r["bbox_w"]), float(r["bbox_h"])
                    break

        first_row = group.iloc[0]
        patient_id = str(first_row["patient_id"]) if "patient_id" in group.columns and pd.notna(first_row["patient_id"]) else (
            str(first_row["Patient ID"]) if "Patient ID" in group.columns and pd.notna(first_row["Patient ID"]) else stem
        )
        view_pos = str(first_row.get("View Position", first_row.get("view_position", "PA")))
        if view_pos in ("nan", "None", ""):
            view_pos = "PA"

        records.append({
            "image": image_name,
            "image_id": stem,
            "patient_id": patient_id,
            "label": int(label),
            "finding_labels": finding_label,
            "raw_findings": raw_findings_str,
            "risk_flag": risk_flag,
            "view_position": view_pos,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "orig_width": orig_w,
            "orig_height": orig_h,
        })

    return assign_splits(records, seed=seed, split_ratios=split_ratios, partition_root=partition_root)


def _scale_bbox_to_image(record, width, height):
    bx, by, bw, bh = record.get("bbox_x"), record.get("bbox_y"), record.get("bbox_w"), record.get("bbox_h")
    if bx is None or by is None or bw is None or bh is None or not np.isfinite([bx, by, bw, bh]).all() or bw <= 0 or bh <= 0:
        # Default central lung region when DICOM box was not provided
        return float(width * 0.15), float(height * 0.15), float(width * 0.70), float(height * 0.70)
    ow, oh = record.get("orig_width"), record.get("orig_height")
    if ow and oh and ow > 0 and oh > 0 and (ow != width or oh != height):
        sx, sy = width / float(ow), height / float(oh)
        bx, by, bw, bh = bx * sx, by * sy, bw * sx, bh * sy
    elif bx + bw > width or by + bh > height:
        # Heuristic scaling if coordinates are in original ~2048-3000px DICOM space
        max_x, max_y = bx + bw, by + bh
        sx = width / max(max_x * 1.05, width)
        sy = height / max(max_y * 1.05, height)
        bx, by, bw, bh = bx * sx, by * sy, bw * sx, bh * sy
    x0 = float(np.clip(bx, 0, width - 2))
    y0 = float(np.clip(by, 0, height - 2))
    x1 = float(np.clip(bx + bw, x0 + 1, width))
    y1 = float(np.clip(by + bh, y0 + 1, height))
    return x0, y0, x1 - x0, y1 - y0


def audit(data_root, output, partition_root=None, *, workers=4, limit=None, reuse_complete=False, image_ids=None, seed=42):
    data_root, output = Path(data_root).resolve(), Path(output)
    if output.resolve().is_relative_to(data_root):
        raise ValueError("Audit output must be outside the read-only dataset")
    cv2.setNumThreads(1)

    csv_path = resolve_csv_path(data_root)
    raw_df = pd.read_csv(csv_path)
    cohort = extract_binary_cohort(raw_df, seed=seed, partition_root=partition_root)
    if not cohort:
        raise ValueError(f"No eligible 'normal' or 'infiltration' images found in {csv_path}")

    if image_ids is not None:
        allowed = {str(x) for x in image_ids} | {Path(str(x)).stem for x in image_ids}
        cohort = [r for r in cohort if r["image"] in allowed or r["image_id"] in allowed]
        if not cohort:
            raise ValueError("Filtered image_ids produced an empty cohort")

    sources = [csv_path]
    if partition_root is not None and Path(partition_root).is_dir():
        sources += sorted(Path(partition_root).glob("*.csv"))
    signature = digest({
        "algorithm": AUDIT_VERSION,
        "source": file_hash(__file__),
        "inputs": {p.name: file_hash(p) for p in sources},
        "seed": int(seed),
        "cohort_images": len(cohort),
    })

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
    items = cohort[:limit]
    missing_images = [row["image"] for row in items if row["image"] not in lookup and row["image_id"] not in lookup]
    if missing_images:
        raise FileNotFoundError(
            f"Image index found {len(lookup):,} entries under {data_root}; "
            f"{len(missing_images):,} requested images are missing, e.g. {missing_images[:3]}. "
            "Check the VinBigData download path."
        )

    normal_total = sum(1 for r in cohort if r["label"] == 0)
    infiltration_total = sum(1 for r in cohort if r["label"] == 1)
    print(
        f"VinBigData binary cohort: total={len(cohort):,} (normal={normal_total:,}, infiltration={infiltration_total:,}); "
        f"auditing {len(items):,} images ({len(known):,} checkpointed) with {workers} workers",
        flush=True,
    )
    started = time.monotonic()

    def inspect(entry):
        name = entry["image"]
        record = {
            "image": name,
            "patient_id": entry["patient_id"],
            "label": int(entry["label"]),
            "finding_labels": entry["finding_labels"],
            "raw_findings": entry["raw_findings"],
            "risk_flag": entry["risk_flag"],
            "view_position": entry["view_position"],
            "split": entry["split"],
        }
        try:
            path = lookup.get(name) or lookup[entry["image_id"]]
            stat = path.stat()
            old = known.get(name)
            if old and old.get("status") in ("ok", "insufficient_patches") and old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
                return old
            image, sha = read_image(path)
            h, w = image.shape
            bx, by, bw, bh = _scale_bbox_to_image(entry, w, h) if entry["label"] == 1 else (None, None, None, None)
            record.update(
                relative_path=path.relative_to(data_root).as_posix(),
                sha256=sha,
                width=w,
                height=h,
                bbox_x=bx,
                bbox_y=by,
                bbox_w=bw,
                bbox_h=bh,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                **estimate(image),
            )
        except (KeyError, OSError, ValueError, cv2.error) as error:
            record.update(status="error", error=str(error)[:160], noise_sigma=None)
        return record

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
            done = min(start + 256, len(items))
            errors = sum(row["status"] == "error" for row in records)
            print(
                f"Audit {done:,}/{len(items):,}; batch read {len(records)-errors}/{len(records)}, "
                f"errors {errors}; elapsed {time.monotonic()-started:.1f}s",
                flush=True,
            )
            if errors == len(records):
                failed_batch = True
                print(
                    "All reads in this batch failed; stopping. Examples: "
                    + str([{key: row.get(key) for key in ("image", "error")} for row in records[:3]]),
                    flush=True,
                )
                break

    frame = pd.DataFrame([known[name] for name in sorted(known)])
    if frame.empty:
        frame = pd.DataFrame(columns=["image", "status", "noise_sigma"])
    temporary = output / "noise_audit.csv.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(output / "noise_audit.csv")
    error_count = int((frame.status == "error").sum())
    summary = {
        "signature": signature,
        "algorithm": AUDIT_VERSION,
        "dataset": "awsaf49/vinbigdata-512-image-dataset",
        "scope": "vinbigdata_512_binary_cohort",
        "metadata_images": int(raw_df.iloc[:, 0].nunique()) if not raw_df.empty else 0,
        "requested_images": len(cohort),
        "class_counts": {"normal": normal_total, "infiltration": infiltration_total},
        "audited_images": len(frame),
        "complete": len(frame) == len(cohort) and error_count == 0,
        "readable_images": int(frame.status.isin(["ok", "insufficient_patches"]).sum()),
        "statuses": {str(k): int(v) for k, v in frame.status.value_counts().items()},
        "csv_sha256": file_hash(output / "noise_audit.csv"),
        "estimate_units": "intensity normalized to [0,1]",
        "interpretation": "Estimated noise proxy; low score does not prove noise-free anatomy",
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed_batch:
        raise OSError("All image reads failed in an audit batch.")
    return frame
