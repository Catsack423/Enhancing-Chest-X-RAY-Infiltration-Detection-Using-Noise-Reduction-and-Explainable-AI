"""Verified measured reports. Pending conditions carry no invented scores."""
from pathlib import Path
import json
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from .evaluation import (CLASSIFICATION_METRICS, STRATA, classification_metrics,
                         classification_bootstrap, summarize_xai, pair_records, patient_bootstrap)
from .experiment import digest, file_hash, write_json


def plot_confusion(matrix, path, title, ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    matrix = np.asarray(matrix)
    totals = matrix.sum(1, keepdims=True)
    percentages = np.divide(matrix*100., totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    ax.imshow(percentages, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks([0, 1], ["Normal", "Infiltration"])
    ax.set_yticks([0, 1], ["Normal", "Infiltration"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, f"{value:,}\n({percentages[y,x]:.2f}%)", ha="center", va="center", bbox={"facecolor":"white", "alpha":.7, "edgecolor":"none"})
    if own:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


def save_overlay(raw, heatmap, box, path, signed=None):
    if raw.shape != heatmap.shape:
        raise ValueError("Overlay must preserve native image dimensions")
    gray = np.repeat(raw[..., None], 3, axis=2).astype(np.float32)
    if signed is None:
        color = cv2.cvtColor(cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        alpha = .45*heatmap[..., None]
        overlay = gray*(1-alpha)+color*alpha
    else:
        if signed.shape != raw.shape or not np.isfinite(signed).all():
            raise ValueError("Signed overlay must align with native image")
        scale = float(np.abs(signed).max())
        display = signed/scale if scale else np.zeros_like(signed)
        overlay = gray.copy()
        overlay[..., 0] += np.maximum(display, 0)*140
        overlay[..., 2] += np.maximum(-display, 0)*140
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    x, y, w, h = box
    start = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    end = min(raw.shape[1]-1, int(np.ceil(x+w))-1), min(raw.shape[0]-1, int(np.ceil(y+h))-1)
    cv2.rectangle(overlay, start, end, (0, 255, 0), 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Overlay encoding failed")
    temporary = path.with_suffix(".tmp")
    encoded.tofile(temporary)
    temporary.replace(path)


def plot_classification_curves(predictions, folder, title):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    metrics = classification_metrics(predictions.label, predictions.prob_infiltration)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(predictions.label, predictions.prob_infiltration)
        precision, recall, _ = precision_recall_curve(predictions.label, predictions.prob_infiltration)
        axes[0].plot(fpr, tpr, label=f"AUC {metrics['roc_auc']:.3f}")
        axes[1].plot(recall, precision, label=f"AP {metrics['pr_auc']:.3f}")
    axes[0].plot([0,1], [0,1], "--", color="gray")
    axes[1].axhline(metrics["positive_prevalence"], ls="--", color="gray", label="Constant score / prevalence")
    for ax in axes:
        ax.set(xlim=(0,1), ylim=(0,1))
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall (AP convention)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(folder/"roc_pr.png", dpi=150)
    plt.close(fig)
    return folder/"roc_pr.png"


def plot_training_history(frame, path, title, best_epoch=None):
    required = {"epoch", "train_eval_loss", "validation_loss"}
    if frame.empty or not required <= set(frame.columns) or not np.isfinite(frame[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("Incomplete or invalid training history")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frame.epoch, frame.train_eval_loss, "o-", label="Train eval balanced BCE")
    ax.plot(frame.epoch, frame.validation_loss, "o-", label="Validation balanced BCE")
    if best_epoch is not None:
        ax.axvline(best_epoch, ls="--", color="gray", label="Validation-selected epoch")
    ax.set(xlabel="Epoch", ylabel="Balanced BCE (eval mode)", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def _read_marker(exp, criterion, condition):
    folder = exp.results(criterion, condition)
    path = folder/"complete.json"
    if not path.exists():
        return None
    marker = json.loads(path.read_text(encoding="utf-8"))
    exp.check_provenance(marker["provenance"], condition)
    key = "checkpoint_hash" if criterion == "CNN" else "cnn_checkpoint_hash"
    if marker[key] != file_hash(exp.results("CNN", condition)/"best.pt"):
        raise ValueError("Result refers to a different CNN checkpoint")
    pairs = (("test_predictions.csv", "predictions_hash"), ("metrics.json", "metrics_hash"), ("history.csv", "history_hash")) if criterion == "CNN" else (("per_image_metrics.csv", "per_image_hash"), ("summary.csv", "summary_hash"))
    for filename, hash_key in pairs:
        if marker.get(hash_key) != file_hash(folder/filename):
            raise ValueError(f"Corrupted completed result: {folder/filename}")
    return marker


def _same_manifest(frame, expected, fields):
    if frame.image.duplicated().any() or set(frame.image) != set(expected.image):
        raise ValueError("Completed predictions do not contain exactly the expected image IDs")
    aligned = frame.set_index("image").loc[expected.image]
    for field in fields:
        if not np.array_equal(aligned[field].to_numpy(), expected[field].to_numpy()):
            raise ValueError(f"Measured records differ from manifest: {field}")


def _table(frame, columns):
    lines = ["| "+" | ".join(columns)+" |", "| "+" | ".join(["---"]*len(columns))+" |"]
    for row in frame.to_dict("records"):
        lines.append("| "+" | ".join("—" if pd.isna(row.get(c)) else f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns)+" |")
    return "\n".join(lines)


def _strength_plots(exp, cnn, xai, stage):
    groups = ("median", "clahe_dwt", "dae_clahe")
    configured = {c["id"]: c for c in exp.config["conditions"]}
    if not cnn.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for group in groups:
            rows = cnn[(cnn.status == "complete") & cnn.condition_id.map(lambda c: configured[c]["group"] == group)].copy()
            if rows.empty:
                continue
            rows["strength_level"] = rows.condition_id.map(lambda c: configured[c].get("level", 0))
            rows = rows.sort_values("strength_level")
            for ax, metric in zip(axes, ("balanced_accuracy", "f1")):
                ax.plot(rows.strength_level, rows[metric], "o-", label=group)
                ax.set(xlabel="Configured strength level", ylabel=metric, ylim=(0,1))
        for ax in axes:
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
        fig.suptitle("Controlled denoising sweep; fusion representations reported separately")
        fig.tight_layout()
        fig.savefig(stage/"strength_classification.png", dpi=150)
        plt.close(fig)
    if not xai.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for criterion, row_axes in zip(("Grad-CAM", "SHAP"), axes):
            for group in groups:
                for ax, metric in zip(row_axes, ("iou_0.3", "pointing_hit")):
                    rows = xai[(xai.explainer == criterion) & (xai.stratum == "overall") & (xai.metric == metric) & xai.condition_id.map(lambda c: configured[c]["group"] == group)].copy()
                    if rows.empty:
                        continue
                    rows["strength_level"] = rows.condition_id.map(lambda c: configured[c].get("level", 0))
                    rows = rows.sort_values("strength_level")
                    ax.plot(rows.strength_level, rows["mean"], "o-", label=group)
                    ax.set(title=f"{criterion} {metric}", xlabel="Configured strength level", ylim=(0,1))
            for ax in row_axes:
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(stage/"strength_xai.png", dpi=150)
        plt.close(fig)


def report_all(exp):
    """Refresh all completed results; a completed-but-missing artifact is an error."""
    from .xai import verify_record, publish_file
    destination = exp.output_root/"comparison"
    stage = Path(exp.staging_root)/"reports"
    stage.mkdir(parents=True, exist_ok=True)
    expected_test = pd.read_csv(exp.artifacts/"manifests/test.csv")
    expected_xai = pd.read_csv(exp.artifacts/"manifests/xai.csv")
    cnn_rows, summaries, ci_rows, paired_class, paired_xai, image_deltas = [], [], [], [], [], []
    predictions, details, statuses, evidence = {}, {}, [], {}
    replicates = exp.config.get("reporting", {}).get("classification_bootstrap_replicates", exp.config["xai"]["bootstrap_replicates"])
    for condition in exp.condition_ids:
        status = {"condition_id": condition}
        row = {"condition_id": condition, "seed": exp.config["seed"], "status": "not_run", **{k: None for k in CLASSIFICATION_METRICS}, "n": None}
        for criterion in ("CNN", "Grad-CAM", "SHAP"):
            marker = _read_marker(exp, criterion, condition)
            status[criterion] = "complete" if marker else "not_run"
            if marker is None:
                continue
            folder = exp.results(criterion, condition)
            evidence[f"{criterion}/{condition}"] = file_hash(folder/"complete.json")
            if criterion == "CNN":
                measured = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
                frame = pd.read_csv(folder/"test_predictions.csv")
                _same_manifest(frame, expected_test, ["patient_id", "label"])
                calculated = classification_metrics(frame.label, frame.prob_infiltration, exp.config["cnn"]["classification_threshold"])
                for key, value in calculated.items():
                    if key == "confusion_matrix":
                        if measured[key] != value:
                            raise ValueError("Stored confusion matrix differs from measured predictions")
                    elif value is None:
                        if measured.get(key) is not None:
                            raise ValueError(f"Stored classification metric differs: {key}")
                    elif not np.isclose(measured.get(key, np.nan), value):
                        raise ValueError(f"Stored classification metric differs: {key}")
                row.update({k: v for k,v in measured.items() if k not in ("provenance", "confusion_matrix")})
                row["status"] = "complete"
                predictions[condition] = frame
                ci = classification_bootstrap(frame, exp.config["cnn"]["classification_threshold"], exp.config["seed"], replicates)
                ci.insert(0, "condition_id", condition)
                ci_rows.append(ci)
                plot_confusion(calculated["confusion_matrix"], stage/condition/"confusion.png", condition)
                plot_classification_curves(frame, stage/condition, condition)
                plot_training_history(pd.read_csv(folder/"history.csv"), stage/condition/"training.png", condition, measured.get("best_epoch"))
            else:
                frame = pd.read_csv(folder/"per_image_metrics.csv")
                _same_manifest(frame, expected_xai, ["patient_id", "risk_flag"])
                if len(frame) != marker["n_images"]:
                    raise ValueError("Completed XAI image count differs from marker")
                for native_row in expected_xai.to_dict("records"):
                    saved = verify_record(exp, criterion, condition, native_row, marker["cnn_checkpoint_hash"])
                    csv_row = frame.set_index("image").loc[native_row["image"]]
                    for metric in ("pointing_hit", "empty_heatmap", "energy_inside", *[f"iou_{t:g}" for t in exp.config["xai"]["iou_thresholds"]]):
                        if not np.isclose(csv_row[metric], saved[metric]):
                            raise ValueError(f"Per-image table differs from completed artifact: {metric}")
                summary = summarize_xai(frame, exp.config)
                stored = pd.read_csv(folder/"summary.csv")
                pd.testing.assert_frame_equal(summary, stored, check_dtype=False, atol=1e-9, rtol=1e-9)
                summary.insert(0, "explainer", criterion)
                summaries.append(summary)
                details[(criterion, condition)] = frame
        cnn_rows.append(row)
        statuses.append(status)
    cnn = pd.DataFrame(cnn_rows)
    xai = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(columns=["explainer", "condition_id", "stratum", "metric", "mean", "ci_low", "ci_high", "n_images", "n_patients"])
    if "baseline" in predictions:
        for condition, frame in predictions.items():
            if condition == "baseline":
                continue
            paired = classification_bootstrap(frame, exp.config["cnn"]["classification_threshold"], exp.config["seed"], replicates, predictions["baseline"])
            paired.insert(0, "condition_id", condition)
            paired_class.append(paired)
    for criterion in ("Grad-CAM", "SHAP"):
        baseline = details.get((criterion, "baseline"))
        if baseline is None:
            continue
        for (explainer, condition), frame in details.items():
            if explainer != criterion or condition == "baseline":
                continue
            paired = pair_records(frame, baseline, ["risk_flag"])
            metrics = ["pointing_hit", "energy_inside", "empty_heatmap", *[f"iou_{t:g}" for t in exp.config["xai"]["iou_thresholds"]]]
            for metric in metrics:
                paired["delta_"+metric] = paired[metric]-paired[metric+"_base"]
            detailed = paired[["image", "patient_id", "risk_flag", *["delta_"+m for m in metrics]]].copy()
            detailed.insert(0, "condition_id", condition)
            detailed.insert(0, "explainer", criterion)
            image_deltas.append(detailed)
            for stratum in STRATA:
                subset = paired if stratum == "overall" else paired[paired.risk_flag == stratum]
                if subset.empty:
                    continue
                for metric in metrics:
                    low, high = patient_bootstrap(subset, "delta_"+metric, exp.config["seed"], exp.config["xai"]["bootstrap_replicates"])
                    paired_xai.append({"explainer":criterion, "condition_id":condition, "stratum":stratum, "metric":metric,
                                       "n_images":len(subset), "n_patients":subset.patient_id.nunique(), "mean_delta":float(subset["delta_"+metric].mean()), "ci_low":low, "ci_high":high})
    tables = {"classification.csv": cnn, "xai.csv": xai, "status.csv": pd.DataFrame(statuses),
              "classification_cis.csv": pd.concat(ci_rows, ignore_index=True) if ci_rows else pd.DataFrame(columns=["condition_id", "metric", "mean", "ci_low", "ci_high"]),
              "paired_classification_deltas.csv": pd.concat(paired_class, ignore_index=True) if paired_class else pd.DataFrame(columns=["condition_id", "metric", "mean_delta", "ci_low", "ci_high"]),
              "paired_xai_deltas.csv": pd.DataFrame(paired_xai, columns=["explainer", "condition_id", "stratum", "metric", "n_images", "n_patients", "mean_delta", "ci_low", "ci_high"]),
              "paired_xai_per_image.csv": pd.concat(image_deltas, ignore_index=True) if image_deltas else pd.DataFrame(columns=["explainer", "condition_id", "image", "patient_id"])}
    frequency = {c["id"] for c in exp.config["conditions"] if c["group"] in ("dwt_features", "dft_features")}
    tables["frequency_classification.csv"] = cnn[cnn.condition_id.isin(frequency | {"baseline"})]
    tables["frequency_xai.csv"] = xai[xai.condition_id.isin(frequency | {"baseline"})]
    for name, frame in tables.items():
        frame.to_csv(stage/name, index=False)
    _strength_plots(exp, cnn, xai, stage)
    note = ("# V3 release — measured results\n\n"
            f"Run mode: {exp.config['run_mode']}; training seed: {exp.config['seed']}. Smoke scores verify the pipeline only.\n\n"
            "Classification uses Infiltration-only vs No Finding-only. XAI includes all bbox cases, wrong predictions and empty maps. "
            "PR-AUC uses average precision. Patient-bootstrap intervals and paired deltas use matched patient draws within this seed; "
            "between-seed mean/SD is reported separately. No clinical significance is inferred from a delta interval.\n\n"
            "Scores from weighted BCE are not assumed to be calibrated population probabilities. "
            "SHAP explains processed spatial views; signed RGB/view arrays are retained and localization uses positive evidence only. "
            "Frequency fusion is a separate representation experiment.\n\n"+_table(cnn, ["condition_id", "status", "n", *CLASSIFICATION_METRICS])+"\n")
    (stage/"README.md").write_text(note, encoding="utf-8")
    hashes = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "complete.json":
            relative = path.relative_to(stage)
            publish_file(path, destination/relative)
            hashes[relative.as_posix()] = file_hash(destination/relative)
    write_json(stage/"complete.json", {"provenance": exp.provenance(), "source_completion_hashes":evidence, "files":hashes})
    publish_file(stage/"complete.json", destination/"complete.json")
    return tables


refresh_reports = report_all


def summarize_seeds(output_root):
    """Mean/SD of distinct seed scores; never concatenate seeds as extra patients."""
    output_root = Path(output_root)
    classifications, explanations = [], []
    signature = None
    for directory in sorted(output_root.glob("seed_*")):
        marker_path = directory/"comparison/complete.json"
        if not marker_path.exists():
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        config = marker["provenance"]["config"]
        if digest(config) != marker["provenance"]["config_hash"]:
            raise ValueError("Corrupted seed report configuration")
        for filename, expected_hash in marker["provenance"]["manifest_hashes"].items():
            if file_hash(directory/"shared/artifacts/manifests"/filename) != expected_hash:
                raise ValueError("Seed manifests changed after reporting")
        for source, expected_hash in marker["source_completion_hashes"].items():
            criterion, condition = source.split("/", 1)
            completion = directory/criterion/"results"/condition/"complete.json"
            if file_hash(completion) != expected_hash:
                raise ValueError("Source completion changed after seed reporting")
            source_marker = json.loads(completion.read_text(encoding="utf-8"))
            hash_key = "checkpoint_hash" if criterion == "CNN" else "cnn_checkpoint_hash"
            if file_hash(directory/"CNN/results"/condition/"best.pt") != source_marker[hash_key]:
                raise ValueError("Checkpoint changed after seed reporting")
        run_signature = {key:value for key,value in config.items() if key != "seed"}
        current = {"config":run_signature, "manifests":marker["provenance"]["manifest_hashes"], "versions":marker["provenance"]["versions"]}
        if signature is not None and current != signature:
            raise ValueError("Seed summaries require identical protocol, manifests and environment")
        signature = current
        if directory.name != f"seed_{config['seed']}":
            raise ValueError("Seed output folder does not match its saved provenance")
        for relative, expected_hash in marker["files"].items():
            if file_hash(directory/"comparison"/relative) != expected_hash:
                raise ValueError("Corrupted seed report; regenerate after checking source artifacts")
        cnn = pd.read_csv(directory/"comparison/classification.csv")
        classifications.append(cnn[cnn.status == "complete"].assign(seed=config["seed"]))
        explanations.append(pd.read_csv(directory/"comparison/xai.csv").assign(seed=config["seed"]))
    folder = output_root/"seed_summary"
    folder.mkdir(parents=True, exist_ok=True)
    records = pd.concat(classifications, ignore_index=True) if classifications else pd.DataFrame()
    xai = pd.concat(explanations, ignore_index=True) if explanations else pd.DataFrame()
    cnn_rows, xai_rows = [], []
    if not records.empty:
        if records.duplicated(["condition_id", "seed"]).any():
            raise ValueError("Duplicate training seed results")
        for condition, frame in records.groupby("condition_id", sort=False):
            for metric in CLASSIFICATION_METRICS:
                cnn_rows.append({"condition_id":condition, "metric":metric, "n_seeds":frame[metric].count(),
                                 "mean":frame[metric].mean(), "sd":frame[metric].std(ddof=1)})
    if not xai.empty:
        if xai.duplicated(["explainer", "condition_id", "stratum", "metric", "seed"]).any():
            raise ValueError("Duplicate XAI seed results")
        for keys, frame in xai.groupby(["explainer", "condition_id", "stratum", "metric"], sort=False):
            xai_rows.append(dict(zip(["explainer", "condition_id", "stratum", "metric"], keys), n_seeds=frame["mean"].count(), mean=frame["mean"].mean(), sd=frame["mean"].std(ddof=1)))
    cnn_summary = pd.DataFrame(cnn_rows, columns=["condition_id", "metric", "n_seeds", "mean", "sd"])
    xai_summary = pd.DataFrame(xai_rows, columns=["explainer", "condition_id", "stratum", "metric", "n_seeds", "mean", "sd"])
    cnn_summary.to_csv(folder/"classification_mean_sd.csv", index=False)
    xai_summary.to_csv(folder/"xai_mean_sd.csv", index=False)
    (folder/"README.md").write_text("Mean and sample SD across distinct training seeds. n_seeds states measured coverage; SD is unavailable for one seed. Patient-bootstrap CIs remain in each seed report. These are descriptive research scores, not claims of clinical significance.\n", encoding="utf-8")
    return cnn_summary, xai_summary
