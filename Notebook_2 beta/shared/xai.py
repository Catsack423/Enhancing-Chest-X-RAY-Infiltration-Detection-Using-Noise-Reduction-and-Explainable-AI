"""Matched 123-image XAI runs with per-image resumable artifacts."""
import json
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from .denoising import Preprocessor, read_gray
from .evaluation import bbox_mask, localization_metrics, summarize_xai
from .experiment import file_hash, write_json, set_seed
from .explainers import gradcam, shap_maps
from .models import load_cnn


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def evaluate_condition(exp, manifests, criterion, condition_id, device="cuda"):
    if criterion not in ("Grad-CAM", "SHAP"):
        raise ValueError("Unsupported explainer")
    exp.lock()
    folder = exp.results(criterion, condition_id)
    model_hash = file_hash(exp.results("CNN", condition_id) / "best.pt")
    provenance = exp.provenance(condition_id)
    marker = folder / "run.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        exp.check_provenance(saved["provenance"], condition_id)
        if saved["cnn_checkpoint_hash"] != model_hash:
            raise ValueError("CNN checkpoint changed since these heatmaps were created")
    else:
        write_json(marker, {"provenance": provenance, "cnn_checkpoint_hash": model_hash})
    set_seed(exp.config["seed"])
    model = load_cnn(exp, condition_id, device)
    prep = Preprocessor(exp, condition_id, device)
    explainer = None
    if criterion == "SHAP":
        import shap
        background = torch.stack([prep.tensor(r) for r in manifests["background"].to_dict("records")]).to(device)
        explainer = shap.GradientExplainer(model, background, batch_size=exp.config["xai"]["shap_batch_size"], local_smoothing=0)
        background_path = folder / "background_manifest.csv"
        manifests["background"].to_csv(background_path, index=False)
    records = []
    for i, row in enumerate(manifests["xai"].to_dict("records"), 1):
        stem = row["image"].rsplit(".", 1)[0]
        record_path = folder / "records" / f"{stem}.json"
        map_path = folder / "heatmaps" / f"{stem}.npz"
        overlay_path = folder / "overlays" / f"{stem}.png"
        if record_path.exists():
            saved = json.loads(record_path.read_text())
            if saved["image"] != row["image"] or saved["cnn_checkpoint_hash"] != model_hash:
                raise ValueError("Incompatible per-image result")
            for artifact, key in ((map_path, "heatmap_hash"), (overlay_path, "overlay_hash")):
                if file_hash(artifact) != saved[key]:
                    raise ValueError(f"Corrupted artifact: {artifact}")
            records.append(saved)
            continue
        started = perf_counter()
        raw = read_gray(exp.data_root / row["relative_path"])
        input_tensor = prep.tensor(row).unsqueeze(0).to(device)
        if criterion == "Grad-CAM":
            heatmap, probability = gradcam(model, input_tensor, raw.shape)
            arrays = {"heatmap": heatmap}
            signed = None
        else:
            arrays = shap_maps(explainer, input_tensor, raw.shape, exp.config)
            heatmap = arrays["positive_heatmap"]
            signed = arrays["signed_native"]
            with torch.no_grad():
                probability = float(model(input_tensor)[0, 0].sigmoid())
        box = tuple(row[key] for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        mask = bbox_mask(raw.shape, [box])
        metrics = localization_metrics(heatmap, mask, exp.config["xai"]["iou_thresholds"])
        save_npz(map_path, arrays)
        from .reporting import save_overlay
        save_overlay(raw, heatmap, box, overlay_path, signed=signed)
        record = {
            "condition_id": condition_id, "image": row["image"], "patient_id": row["patient_id"],
            "risk_flag": row["risk_flag"], "prob_infiltration": probability,
            "original_height": raw.shape[0], "original_width": raw.shape[1],
            "cnn_checkpoint_hash": model_hash,
            "heatmap_hash": file_hash(map_path), "overlay_hash": file_hash(overlay_path),
            "seconds": perf_counter() - started, **metrics,
        }
        write_json(record_path, record)
        records.append(record)
        print(f"{criterion} {condition_id}: {i}/{len(manifests['xai'])}", flush=True)
    records = pd.DataFrame(records)
    if len(records) != len(manifests["xai"]) or records.image.duplicated().any():
        raise ValueError("Incomplete or duplicate XAI evaluations")
    records.to_csv(folder / "per_image_metrics.csv", index=False)
    summarize_xai(records, exp.config).to_csv(folder / "summary.csv", index=False)
    write_json(folder / "complete.json", {"provenance": provenance, "cnn_checkpoint_hash": model_hash, "n_images": len(records)})
    return records


def run_xai(exp, manifests, criterion, device="cuda"):
    records = []
    for condition_id in exp.condition_ids:
        records.append(evaluate_condition(exp, manifests, criterion, condition_id, device))
        from .reporting import refresh_reports
        refresh_reports(exp)
    return pd.concat(records, ignore_index=True)
