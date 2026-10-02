"""Classification and native-resolution localization metrics, including patient CIs."""
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def classification_metrics(truth, probabilities, threshold=0.5):
    truth = np.asarray(truth, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid classification probabilities")
    predicted = (probabilities >= threshold).astype(int)
    return {
        "n": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }


def resize_map(values, shape):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Heatmap must be finite and two-dimensional")
    return cv2.resize(values, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def positive_normalize(values):
    values = np.maximum(np.asarray(values, dtype=np.float32), 0)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite heatmap")
    maximum = float(values.max())
    return values / maximum if maximum > 0 else np.zeros_like(values)


def bbox_mask(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for x, y, width, height in boxes:
        if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
            raise ValueError("Invalid bounding box")
        left, top = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        right, bottom = min(shape[1], int(np.ceil(x + width))), min(shape[0], int(np.ceil(y + height)))
        if right <= left or bottom <= top:
            raise ValueError("Bounding box lies outside the image")
        mask[top:bottom, left:right] = True
    if not mask.any():
        raise ValueError("Missing Infiltration bounding box")
    return mask


def localization_metrics(heatmap, mask, thresholds=(0.3, 0.5)):
    if heatmap.shape != mask.shape or not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.max() > 1:
        raise ValueError("Heatmap and original-coordinate mask must align in [0,1]")
    nonempty = bool(heatmap.max() > 0)
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    result = {
        "pointing_hit": int(nonempty and mask[peak_y, peak_x]),
        "peak_x": int(peak_x) if nonempty else None,
        "peak_y": int(peak_y) if nonempty else None,
        "empty_heatmap": not nonempty,
        "energy_inside": float(heatmap[mask].sum() / heatmap.sum()) if nonempty else 0.0,
    }
    for threshold in thresholds:
        binary = (heatmap >= threshold) & (heatmap > 0)
        union = np.count_nonzero(binary | mask)
        result[f"iou_{threshold:g}"] = float(np.count_nonzero(binary & mask) / union) if union else 0.0
    return result


def patient_bootstrap(frame, metric, seed=42, replicates=2000):
    # Sorted patient order and a freshly seeded generator give identical draws
    # across conditions and across the two explainers for each risk stratum.
    grouped = frame.groupby("patient_id", sort=True)[metric].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    sums = grouped["sum"].to_numpy()[indexes].sum(axis=1)
    counts = grouped["count"].to_numpy()[indexes].sum(axis=1)
    lower, upper = np.quantile(sums / counts, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_xai(records, config):
    rows = []
    metrics = ["pointing_hit", "energy_inside"] + [f"iou_{t:g}" for t in config["xai"]["iou_thresholds"]]
    for condition, frame in records.groupby("condition_id", sort=False):
        for stratum in ("overall", "pure", "mixed_high_risk", "mixed_low_risk"):
            subset = frame if stratum == "overall" else frame[frame.risk_flag == stratum]
            for metric in metrics:
                row = {"condition_id": condition, "stratum": stratum, "metric": metric,
                       "n_images": len(subset), "n_patients": subset.patient_id.nunique(),
                       "mean": None, "ci_low": None, "ci_high": None}
                if len(subset):
                    row["mean"] = float(subset[metric].mean())
                    row["ci_low"], row["ci_high"] = patient_bootstrap(subset, metric, config["seed"], config["xai"]["bootstrap_replicates"])
                rows.append(row)
    return pd.DataFrame(rows)
