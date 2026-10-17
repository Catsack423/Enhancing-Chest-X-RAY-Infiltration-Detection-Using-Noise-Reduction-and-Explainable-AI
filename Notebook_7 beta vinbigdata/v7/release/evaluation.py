"""Measured classification and native-coordinate localization, clustered by patient."""
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall",
                          "specificity", "f1", "roc_auc", "pr_auc")
STRATA = ("overall", "pure", "mixed_high_risk", "mixed_low_risk")


def classification_metrics(truth, probabilities, threshold=0.5):
    truth, probabilities = np.asarray(truth), np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or probabilities.shape != truth.shape or not len(truth):
        raise ValueError("Classification requires matched nonempty one-dimensional arrays")
    if not np.isin(truth, [0, 1]).all() or not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid classification labels or probabilities")
    if not 0 < threshold < 1:
        raise ValueError("Classification threshold must lie strictly between zero and one")
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    both = sensitivity is not None and specificity is not None
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float((sensitivity + specificity) / 2) if both else None,
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "specificity": float(specificity) if specificity is not None else None,
            "f1": float(f1_score(truth, predicted, zero_division=0)),
            "roc_auc": float(roc_auc_score(truth, probabilities)) if both else None,
            "pr_auc": float(average_precision_score(truth, probabilities)) if both else None,
            "positive_prevalence": float(np.mean(truth)),
            "constant_score_pr_auc": float(np.mean(truth)),
            "confusion_matrix": matrix.tolist()}


def youden_threshold(truth, probabilities):
    """Validation operating point maximizing sensitivity + specificity - 1."""
    truth, probabilities = np.asarray(truth), np.asarray(probabilities, dtype=float)
    if set(np.unique(truth)) != {0, 1} or not np.isfinite(probabilities).all():
        raise ValueError("Youden threshold requires both classes and finite scores")
    fpr, tpr, thresholds = roc_curve(truth, probabilities, drop_intermediate=False)
    finite = np.isfinite(thresholds)  # roc_curve prepends an infinite "predict none" threshold
    j = tpr[finite] - fpr[finite]
    # Thresholds descend: keep the highest (most specific) one among float-level ties.
    best = thresholds[finite][int(np.flatnonzero(j >= j.max() - 1e-12)[0])]
    # classification_metrics needs an open (0,1) threshold; prediction rule stays score >= threshold.
    return float(np.clip(best, np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)))


def resize_map(values, shape):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or len(shape) != 2 or min(shape) <= 0:
        raise ValueError("Heatmap must be finite, nonempty, and two-dimensional")
    return cv2.resize(values, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)


def positive_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("Invalid heatmap")
    values = np.maximum(values, 0)
    maximum = float(values.max())
    return values / maximum if maximum else np.zeros_like(values)


def bbox_mask(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for x, y, width, height in boxes:
        if not np.isfinite([x, y, width, height]).all() or width <= 0 or height <= 0:
            raise ValueError("Invalid bounding box")
        left, top = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        right, bottom = min(shape[1], int(np.ceil(x + width))), min(shape[0], int(np.ceil(y + height)))
        if right <= left or bottom <= top:
            raise ValueError("Bounding box lies outside the native image")
        mask[top:bottom, left:right] = True
    if not mask.any():
        raise ValueError("Missing Infiltration bounding box")
    return mask


def localization_metrics(heatmap, mask, thresholds=(0.3, 0.5)):
    heatmap, mask = np.asarray(heatmap), np.asarray(mask, dtype=bool)
    if heatmap.shape != mask.shape or heatmap.ndim != 2 or not mask.any() or not np.isfinite(heatmap).all() or heatmap.min() < 0 or heatmap.max() > 1:
        raise ValueError("Heatmap and native-coordinate mask must align in [0,1]")
    nonempty = bool(heatmap.max() > 0)
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    result = {"pointing_hit": int(nonempty and mask[peak_y, peak_x]),
              "peak_x": int(peak_x) if nonempty else None, "peak_y": int(peak_y) if nonempty else None,
              "empty_heatmap": int(not nonempty),
              "energy_inside": float(heatmap[mask].sum() / heatmap.sum()) if nonempty else 0.0}
    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError("Localization thresholds must lie in (0,1]")
        binary = (heatmap >= threshold) & (heatmap > 0)
        result[f"iou_{threshold:g}"] = float(np.count_nonzero(binary & mask) / np.count_nonzero(binary | mask))
    return result


def _validate_frame(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing or frame.empty or frame[list(columns)].isna().any().any():
        raise ValueError(f"Missing or invalid measured columns: {sorted(missing)}")
    if "image" in columns and frame.image.duplicated().any():
        raise ValueError("Duplicate measured image IDs")


def patient_bootstrap(frame, metric, seed=42, replicates=2000):
    _validate_frame(frame, ["patient_id", metric])
    if replicates < 2 or not np.isfinite(frame[metric].to_numpy(dtype=float)).all():
        raise ValueError("Invalid patient bootstrap data or replicate budget")
    grouped = frame.groupby("patient_id", sort=True)[metric].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, replicates, 128):
        indexes = rng.integers(0, len(grouped), size=(min(128, replicates-start), len(grouped)))
        samples.extend(grouped["sum"].to_numpy()[indexes].sum(1) / grouped["count"].to_numpy()[indexes].sum(1))
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def _weighted_scores(frame, weights, threshold):
    """Fixed score sorting and patient multiplicities avoid re-sorting 2,000 times."""
    y = frame.label.to_numpy(dtype=int)
    p = frame.prob_infiltration.to_numpy(dtype=float)
    pred = p >= threshold
    totals = [weights @ a for a in ((y == 0) & ~pred, (y == 0) & pred, (y == 1) & ~pred, (y == 1) & pred)]
    tn, fp, fn, tp = totals
    neg, pos = tn + fp, tp + fn
    divide = lambda a, b: np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b != 0)
    recall, specificity = divide(tp, pos), divide(tn, neg)
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp, dtype=float), where=(tp+fp) != 0)
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p)-1]
    positive = np.cumsum(weights[:, order] * y[order], axis=1)[:, ends]
    negative = np.cumsum(weights[:, order] * (1-y[order]), axis=1)[:, ends]
    tpr, fpr = divide(positive, pos[:, None]), divide(negative, neg[:, None])
    # Include the (0,0) origin, combine equal-score groups, and use AP convention.
    tpr0, fpr0 = np.c_[np.zeros(len(weights)), tpr], np.c_[np.zeros(len(weights)), fpr]
    auc = np.sum(np.diff(fpr0, axis=1) * (tpr0[:, 1:] + tpr0[:, :-1]) / 2, axis=1)
    precision_curve = np.divide(positive, positive+negative, out=np.zeros_like(positive), where=(positive+negative) != 0)
    ap = np.sum(np.diff(tpr0, axis=1) * precision_curve, axis=1)
    return {"accuracy": divide(tp+tn, pos+neg), "balanced_accuracy": (recall+specificity)/2,
            "precision": precision, "recall": np.nan_to_num(recall), "specificity": specificity,
            "f1": np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp, dtype=float), where=(2*tp+fp+fn) != 0),
            "roc_auc": auc, "pr_auc": ap}


def pair_records(frame, baseline, fields=()):
    _validate_frame(frame, ["image", "patient_id", *fields])
    _validate_frame(baseline, ["image", "patient_id", *fields])
    if set(frame.image) != set(baseline.image):
        raise ValueError("Paired comparison requires identical image IDs")
    paired = frame.merge(baseline, on="image", suffixes=("", "_base"), validate="one_to_one")
    for field in ("patient_id", *fields):
        if not np.array_equal(paired[field].to_numpy(), paired[field+"_base"].to_numpy()):
            raise ValueError(f"Paired comparison metadata differs: {field}")
    return paired


def classification_bootstrap(frame, threshold=0.5, seed=42, replicates=2000, baseline=None,
                             baseline_threshold=None):
    """Each model keeps its own validation-selected threshold in paired deltas."""
    baseline_threshold = threshold if baseline_threshold is None else baseline_threshold
    _validate_frame(frame, ["image", "patient_id", "label", "prob_infiltration"])
    measured = classification_metrics(frame.label, frame.prob_infiltration, threshold)
    reference = None
    if baseline is not None:
        paired = pair_records(frame, baseline, ["label"])
        reference = baseline.set_index("image").loc[frame.image].reset_index()
        reference_metrics = classification_metrics(reference.label, reference.prob_infiltration, baseline_threshold)
    if replicates < 2:
        raise ValueError("Bootstrap requires at least two replicates")
    patients, patient_index = np.unique(frame.patient_id.to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in CLASSIFICATION_METRICS}
    for start in range(0, replicates, 64):
        draws = rng.integers(0, len(patients), size=(min(64, replicates-start), len(patients)))
        counts = np.stack([np.bincount(draw, minlength=len(patients)) for draw in draws])
        weights = counts[:, patient_index].astype(float)
        scores = _weighted_scores(frame, weights, threshold)
        if reference is not None:
            other = _weighted_scores(reference, weights, baseline_threshold)
            scores = {metric: values-other[metric] for metric, values in scores.items()}
        for metric, values in scores.items():
            samples[metric].extend(values)
    rows = []
    for metric, values in samples.items():
        finite = np.asarray(values)[np.isfinite(values)]
        estimate = measured[metric]
        if reference is not None:
            estimate = estimate-reference_metrics[metric] if estimate is not None and reference_metrics[metric] is not None else None
        interval = [None, None] if not len(finite) else [float(v) for v in np.quantile(finite, [.025, .975])]
        rows.append({"metric": metric, "mean_delta" if baseline is not None else "mean": estimate,
                     "ci_low": interval[0], "ci_high": interval[1], "n_images": len(frame),
                     "n_patients": len(patients), "n_valid": len(finite), "replicates": replicates})
    return pd.DataFrame(rows)


def summarize_xai(records, config):
    rows = []
    metrics = ["pointing_hit", "energy_inside", "empty_heatmap"] + [f"iou_{t:g}" for t in config["xai"]["iou_thresholds"]]
    _validate_frame(records, ["condition_id", "patient_id", "risk_flag", *metrics])
    if records.image.isna().any() or records.duplicated(["condition_id", "image"]).any():
        raise ValueError("Missing or duplicate condition/image XAI records")
    for condition, frame in records.groupby("condition_id", sort=False):
        for stratum in STRATA:
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
