"""Native 123-image evaluations with verified per-image resume and durable markers."""
from pathlib import Path
import json
import shutil
from time import perf_counter
import numpy as np
import pandas as pd
import torch
from .evaluation import bbox_mask, localization_metrics, summarize_xai
from .experiment import file_hash, write_json, set_seed
from .explainers import gradcam_maps, shap_maps


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def publish_file(source, destination):
    """Copy VM-staged artifacts, verify bytes at destination, then expose final name."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name+".copying")
    shutil.copyfile(source, temporary)
    if file_hash(source) != file_hash(temporary):
        raise IOError(f"Output copy failed integrity verification: {destination}")
    temporary.replace(destination)
    if file_hash(source) != file_hash(destination):
        raise IOError(f"Final output failed integrity verification: {destination}")


def _stem(image):
    if Path(str(image)).name != image or "/" in image or "\\" in image:
        raise ValueError("Image ID must be a plain filename")
    return Path(image).stem


def _paths(folder, image):
    stem = _stem(image)
    return folder/"records"/(stem+".json"), folder/"heatmaps"/(stem+".npz"), folder/"overlays"/(stem+".png")


def validate_arrays(arrays, shape, criterion):
    required = {"positive_heatmap", "signed_native", "signed_input", "signed_view_input"}
    if criterion == "SHAP":
        required.add("view_channel_attributions")
    if not required <= set(arrays):
        raise ValueError(f"Missing {criterion} array artifacts: {required-set(arrays)}")
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite explainer artifact: {key}")
    if arrays["positive_heatmap"].shape != tuple(shape) or arrays["signed_native"].shape != tuple(shape):
        raise ValueError("Explainer artifact is not in native image coordinates")
    positive = arrays["positive_heatmap"]
    if positive.min() < 0 or positive.max() > 1 or arrays["signed_input"].ndim != 2 or arrays["signed_view_input"].ndim != 3:
        raise ValueError("Invalid localization map or signed attribution shape")
    if not np.allclose(arrays["signed_view_input"].sum(0), arrays["signed_input"], atol=1e-5, rtol=1e-5):
        raise ValueError("Signed view contributions do not sum to the fused explanation")
    if criterion == "SHAP":
        channels = arrays["view_channel_attributions"]
        if channels.ndim != 4 or channels.shape[1] != 3 or not np.allclose(channels.sum(1), arrays["signed_view_input"], atol=1e-5, rtol=1e-5):
            raise ValueError("Invalid signed RGB/view SHAP artifact")


def verify_record(exp, criterion, condition_id, row, model_hash, folder=None):
    folder = folder or exp.results(criterion, condition_id)
    record_path, map_path, overlay_path = _paths(folder, row["image"])
    saved = json.loads(record_path.read_text(encoding="utf-8"))
    exp.check_provenance(saved["provenance"], condition_id)
    for key, expected in (("image", row["image"]), ("patient_id", row["patient_id"]), ("risk_flag", row["risk_flag"]),
                          ("condition_id", condition_id), ("criterion", criterion), ("cnn_checkpoint_hash", model_hash)):
        if saved.get(key) != expected:
            raise ValueError(f"Incompatible per-image result: {key}")
    for artifact, key in ((map_path, "heatmap_hash"), (overlay_path, "overlay_hash")):
        if file_hash(artifact) != saved[key]:
            raise ValueError(f"Corrupted artifact: {artifact}")
    shape = (int(saved["original_height"]), int(saved["original_width"]))
    with np.load(map_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_arrays(arrays, shape, criterion)
    box = tuple(float(row[key]) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
    actual = localization_metrics(arrays["positive_heatmap"], bbox_mask(shape, [box]), exp.config["xai"]["iou_thresholds"])
    for key, value in actual.items():
        stored = saved.get(key)
        if value is None:
            if stored is not None:
                raise ValueError(f"Corrupted XAI metric: {key}")
        elif stored is None or not np.isclose(stored, value):
            raise ValueError(f"Corrupted XAI metric: {key}")
    if not 0 <= saved["prob_infiltration"] <= 1:
        raise ValueError("Invalid saved Infiltration score")
    return saved


def evaluate_condition(exp, manifests, criterion, condition_id, device="cuda", *,
                       preprocessor_factory=None, model_loader=None, gradcam_fn=gradcam_maps,
                       shap_maps_fn=shap_maps, explainer_factory=None):
    if criterion not in ("Grad-CAM", "SHAP"):
        raise ValueError("Unsupported explainer")
    from .preprocessing import ReleasePreprocessor
    from .models import load_cnn
    from .reporting import save_overlay
    preprocessor_factory = preprocessor_factory or ReleasePreprocessor
    model_loader = model_loader or load_cnn
    expected = manifests["xai"]
    if expected.empty or expected.image.duplicated().any():
        raise ValueError("XAI manifest is empty or contains duplicate images")
    exp.lock()
    if exp.cache is None or not getattr(exp.cache, "ready", True):
        from .runtime_cache import RuntimeCache
        RuntimeCache(exp).prepare(manifests)
    folder = exp.results(criterion, condition_id)
    stage = Path(exp.staging_root)/criterion/condition_id
    model_hash = file_hash(exp.results("CNN", condition_id)/"best.pt")
    cnn_completion = json.loads((exp.results("CNN", condition_id)/"complete.json").read_text(encoding="utf-8"))
    exp.check_provenance(cnn_completion["provenance"], condition_id)
    if cnn_completion["checkpoint_hash"] != model_hash:
        raise ValueError("XAI requires the completed classification run's unchanged best checkpoint")
    provenance = exp.provenance(condition_id)
    marker = folder/"run.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        exp.check_provenance(saved["provenance"], condition_id)
        if saved["cnn_checkpoint_hash"] != model_hash or saved["criterion"] != criterion:
            raise ValueError("CNN checkpoint or criterion changed since XAI started")
    else:
        write_json(stage/"run.json", {"provenance": provenance, "cnn_checkpoint_hash": model_hash, "criterion": criterion})
        publish_file(stage/"run.json", marker)
    set_seed(exp.config["seed"])
    model = model_loader(exp, condition_id, device=device)
    prep = preprocessor_factory(exp, condition_id, device)
    prep.prepare(manifests)
    explainer = None
    if criterion == "SHAP":
        if explainer_factory is None:
            import shap
            explainer_factory = shap.GradientExplainer
        settings = exp.config["xai"]
        if settings["shap_batch_size"] > 8 or settings["shap_batch_size"] < 1:
            raise ValueError("SHAP batch size must be between 1 and 8")
        background = torch.stack([prep.tensor(r) for r in manifests["background"].to_dict("records")]).to(device)
        explainer = explainer_factory(model, background, batch_size=settings["shap_batch_size"], local_smoothing=0)
        stage.mkdir(parents=True, exist_ok=True)
        manifests["background"].to_csv(stage/"background_manifest.csv", index=False)
        publish_file(stage/"background_manifest.csv", folder/"background_manifest.csv")
    records = []
    for i, row in enumerate(expected.to_dict("records"), 1):
        record_path, map_path, overlay_path = _paths(folder, row["image"])
        if record_path.exists():
            records.append(verify_record(exp, criterion, condition_id, row, model_hash))
            continue
        started = perf_counter()
        raw = exp.cache.native(row)
        if raw.ndim != 2 or not np.isfinite(raw).all():
            raise ValueError(f"Invalid native XAI pixels: {row['image']}")
        input_tensor = prep.tensor(row).unsqueeze(0).to(device)
        if criterion == "Grad-CAM":
            arrays, probability = gradcam_fn(model, input_tensor, raw.shape)
        else:
            arrays = shap_maps_fn(explainer, input_tensor, raw.shape, exp.config)
            with torch.no_grad():
                probability = float(model(input_tensor)[0, 0].sigmoid())
        validate_arrays(arrays, raw.shape, criterion)
        heatmap = arrays["positive_heatmap"]
        box = tuple(float(row[key]) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        metrics = localization_metrics(heatmap, bbox_mask(raw.shape, [box]), exp.config["xai"]["iou_thresholds"])
        stage_record, stage_map, stage_overlay = _paths(stage, row["image"])
        save_npz(stage_map, arrays)
        save_overlay(raw, heatmap, box, stage_overlay, signed=arrays["signed_native"] if criterion == "SHAP" else None)
        record = {"condition_id": condition_id, "criterion": criterion, "image": row["image"],
                  "patient_id": row["patient_id"], "risk_flag": row["risk_flag"], "prob_infiltration": probability,
                  "original_height": raw.shape[0], "original_width": raw.shape[1], "provenance": provenance,
                  "cnn_checkpoint_hash": model_hash, "heatmap_hash": file_hash(stage_map), "overlay_hash": file_hash(stage_overlay),
                  "seconds": perf_counter()-started, "attribution_domain": "processed_spatial_views", **metrics}
        # Per-image completion record is exposed only after both durable artifacts verify.
        publish_file(stage_map, map_path)
        publish_file(stage_overlay, overlay_path)
        write_json(stage_record, record)
        publish_file(stage_record, record_path)
        records.append(verify_record(exp, criterion, condition_id, row, model_hash))
        print(f"{criterion} {condition_id}: {i}/{len(expected)}", flush=True)
    frame = pd.DataFrame([{key: value for key, value in record.items() if key != "provenance"} for record in records])
    frame.to_csv(stage/"per_image_metrics.csv", index=False)
    summarize_xai(frame, exp.config).to_csv(stage/"summary.csv", index=False)
    for name in ("per_image_metrics.csv", "summary.csv"):
        publish_file(stage/name, folder/name)
    completion = {"provenance": provenance, "criterion": criterion, "cnn_checkpoint_hash": model_hash,
                  "n_images": len(frame), "per_image_hash": file_hash(folder/"per_image_metrics.csv"),
                  "summary_hash": file_hash(folder/"summary.csv")}
    write_json(stage/"complete.json", completion)
    publish_file(stage/"complete.json", folder/"complete.json")
    return frame


def run_xai(exp, manifests, criterion, device="cuda"):
    records = []
    for condition_id in exp.condition_ids:
        records.append(evaluate_condition(exp, manifests, criterion, condition_id, device))
        from .reporting import report_all
        report_all(exp)
    return pd.concat(records, ignore_index=True)
