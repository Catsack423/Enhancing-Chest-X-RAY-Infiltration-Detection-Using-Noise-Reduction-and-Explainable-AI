"""Read-only e2 inputs; write comparisons only under the e7 output folder."""
from pathlib import Path
import json

import pandas as pd

from .experiment import file_hash

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def _cnn_result(root, condition, budget):
    folder = root / "CNN/results" / condition
    marker = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    measured = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if marker["checkpoint_hash"] != file_hash(folder / "best.pt"):
        raise ValueError(f"Checkpoint hash mismatch: {folder}")
    if marker["provenance"] != measured["provenance"]:
        raise ValueError(f"Metric provenance mismatch: {folder}")
    if measured["epochs_run"] != budget or measured["provenance"]["config"]["cnn"]["epochs"] != budget:
        raise ValueError(f"Expected {budget} completed epochs: {folder}")
    if measured["condition_id"] != condition or measured["provenance"]["condition_id"] != condition:
        raise ValueError(f"Condition mismatch: {folder}")
    return measured, marker


def _check_controls(a, b):
    for key in ("manifest_hashes", "versions"):
        if a[key] != b[key]:
            raise ValueError(f"e2/e7 {key} differ; the budgets cannot be compared under identical conditions")
    for key in ("seed", "training", "data", "preprocessing", "xai", "conditions"):
        if a["config"][key] != b["config"][key]:
            raise ValueError(f"e2/e7 control changed: {key}")
    cnn_a, cnn_b = dict(a["config"]["cnn"]), dict(b["config"]["cnn"])
    for field in ("epochs", "patience"):
        cnn_a.pop(field, None)
        cnn_b.pop(field, None)
    if cnn_a != cnn_b:
        raise ValueError("e2/e7 CNN controls differ beyond epochs/patience")
    for key in ("fusion", "backbone_batchnorm", "train_eval_loss"):
        if a["config"]["anti_overfit"][key] != b["config"]["anti_overfit"][key]:
            raise ValueError(f"e2/e7 architecture changed: {key}")


def compare_with_e2(exp, e2_root):
    """No old scores or smoke substitutes if a real run is unavailable."""
    e2_root, e7_root = Path(e2_root).resolve(), exp.output_root.resolve()
    if e2_root == e7_root:
        raise ValueError("e2 and e7 outputs must be separate")
    required = ("complete.json", "metrics.json", "best.pt", "test_predictions.csv", "history.csv")
    missing = [str(root / "CNN/results" / condition / name)
               for root in (e2_root, e7_root) for condition in exp.condition_ids for name in required
               if not (root / "CNN/results" / condition / name).is_file()]
    if missing:
        return {"ready": False, "missing": missing}
    classifications, summaries, histories, missing_xai = [], [], {}, []
    for condition in exp.condition_ids:
        a, a_marker = _cnn_result(e2_root, condition, 2)
        b, b_marker = _cnn_result(e7_root, condition, 7)
        exp.check_provenance(b["provenance"], condition)
        _check_controls(a["provenance"], b["provenance"])
        prediction_frames = []
        for root in (e2_root, e7_root):
            frame = pd.read_csv(root / "CNN/results" / condition / "test_predictions.csv")
            if frame.image.duplicated().any():
                raise ValueError("Duplicate test images")
            prediction_frames.append(frame[["image", "patient_id", "label"]].sort_values("image").reset_index(drop=True))
        pd.testing.assert_frame_equal(*prediction_frames)
        for budget, root, result, marker in ((2, e2_root, a, a_marker), (7, e7_root, b, b_marker)):
            history = pd.read_csv(root / "CNN/results" / condition / "history.csv")
            if len(history) != budget or history.epoch.tolist() != list(range(1, budget + 1)):
                raise ValueError("Incomplete epoch history")
            histories[(budget, condition)] = history
            classifications.append({"epoch_budget": budget, "condition_id": condition,
                                    **{k: result[k] for k in ("n", *METRICS, "epochs_run", "best_epoch", "trainable_parameters")},
                                    "last_train_eval_loss": float(history.iloc[-1].train_eval_loss),
                                    "last_validation_loss": float(history.iloc[-1].validation_loss)})
        for criterion in ("Grad-CAM", "SHAP"):
            paths = [(root / criterion / "results" / condition) for root in (e2_root, e7_root)]
            if not all((p / "complete.json").is_file() and (p / "summary.csv").is_file() for p in paths):
                missing_xai.append(f"{criterion}/{condition}: need complete summaries in both budgets")
                continue
            for budget, folder, marker in ((2, paths[0], a_marker), (7, paths[1], b_marker)):
                x_marker = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
                if x_marker["cnn_checkpoint_hash"] != marker["checkpoint_hash"] or x_marker["provenance"] != marker["provenance"]:
                    raise ValueError(f"XAI refers to a different CNN: {folder}")
                summary = pd.read_csv(folder / "summary.csv")
                summary.insert(0, "explainer", criterion)
                summary.insert(0, "epoch_budget", budget)
                summaries.append(summary)
    classification = pd.DataFrame(classifications)
    for budget in (2, 7):
        selection = classification.epoch_budget == budget
        baseline = classification[selection].set_index("condition_id").loc["baseline", "accuracy"]
        classification.loc[selection, "delta_accuracy_vs_own_baseline"] = classification.loc[selection, "accuracy"] - baseline
    index = classification.set_index(["epoch_budget", "condition_id"])
    differences = pd.DataFrame([{"condition_id": c, **{f"delta_{m}_e7_minus_e2": index.loc[(7, c), m] - index.loc[(2, c), m] for m in METRICS}}
                                for c in exp.condition_ids])
    xai = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    destination = e7_root / "comparison/e2_vs_e7"
    destination.mkdir(parents=True, exist_ok=True)
    classification.to_csv(destination / "classification.csv", index=False)
    differences.to_csv(destination / "classification_differences.csv", index=False)
    xai.to_csv(destination / "xai.csv", index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(exp.condition_ids), figsize=(13, 4), squeeze=False)
    for ax, condition in zip(axes[0], exp.condition_ids):
        for budget, style in ((2, "o--"), (7, "s-")):
            history = histories[(budget, condition)]
            ax.plot(history.epoch, history.train_eval_loss, style, color="#1976a3", label=f"e{budget} train (eval)")
            ax.plot(history.epoch, history.validation_loss, style, color="#c55c26", label=f"e{budget} validation")
        ax.set(title=condition, xlabel="Epoch", ylabel="BCE loss", xticks=range(1, 8))
        ax.grid(alpha=0.2)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Same frozen model/data/settings: epoch budgets 2 vs 7")
    fig.tight_layout()
    curve_path = destination / "training_curves.png"
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"ready": True, "classification": classification, "differences": differences,
            "xai": xai, "missing_xai": missing_xai, "curve_path": curve_path}
