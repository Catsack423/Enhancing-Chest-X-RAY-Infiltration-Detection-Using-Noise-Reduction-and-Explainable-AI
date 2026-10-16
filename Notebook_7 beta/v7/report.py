"""Measured pilot classification, paired patient CIs and reference-proxy image quality."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from .dataset import Preprocessor
from .noise import STATES
from .release.evaluation import classification_metrics, classification_bootstrap


def image_quality(reference, processed):
    """PSNR and Wang SSIM with a 7x7 uniform window, sample covariance, range=1."""
    x, y = reference.astype(np.float64)/255, processed.astype(np.float64)/255
    if x.shape != y.shape or min(x.shape) < 7:
        raise ValueError("Image quality requires same-sized images of at least 7x7")
    mse = np.mean((x-y)**2)
    psnr = math.inf if mse == 0 else float(-10*np.log10(mse))
    ux, uy = uniform_filter(x, 7), uniform_filter(y, 7)
    vx = (uniform_filter(x*x, 7)-ux*ux)*49/48
    vy = (uniform_filter(y*y, 7)-uy*uy)*49/48
    covariance = (uniform_filter(x*y, 7)-ux*uy)*49/48
    ssim = ((2*ux*uy+.01**2)*(2*covariance+.03**2)) / ((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return {"psnr": psnr, "ssim": float(ssim[3:-3, 3:-3].mean()), "mse": float(mse),
            "reference_identical": bool(mse == 0)}


def report(run, dae, device):
    from .runner import csv
    directory = run.output / "reports"
    if run.completed(directory, marker="report_complete.json"):
        print("Reports already verified; reusing durable results.", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    names = [c["id"] for c in run.config["conditions"]]
    predictions, specs = {}, {}
    for name in names:
        folder = run.output / "CNN" / name
        predictions[name] = pd.read_csv(folder / "test_predictions.csv")
        specs[name] = json.loads((folder / "model_spec.json").read_text())
    baseline = predictions["baseline"]
    measurements, intervals, deltas = [], [], []
    for name in names:
        for state in STATES:
            frame = predictions[name][predictions[name].corruption_id == state["id"]]
            base = baseline[baseline.corruption_id == state["id"]]
            threshold = specs[name]["threshold"]
            measured = classification_metrics(frame.label, frame.prob_infiltration, threshold)
            reference = classification_metrics(frame.label, frame.prob_infiltration, .5)
            meta = {"condition_id": name, "corruption_id": state["id"], "noise_family": state["family"],
                    "noise_level": state["level"], "threshold": threshold}
            measurements.append({**meta, **{k: (json.dumps(v) if k == "confusion_matrix" else v) for k, v in measured.items()},
                **{k+"_at_0_5": (json.dumps(v) if k == "confusion_matrix" else v) for k, v in reference.items()}})
            ci = classification_bootstrap(frame, threshold, run.config["seed"], run.config["bootstrap_replicates"])
            intervals.append(ci.assign(**meta))
            if name != "baseline":
                paired = classification_bootstrap(frame, threshold, run.config["seed"], run.config["bootstrap_replicates"],
                    baseline=base, baseline_threshold=specs["baseline"]["threshold"])
                deltas.append(paired.assign(**meta))
        print(f"Reports: classification and patient CIs {name}", flush=True)
    csv(directory / "classification.csv", pd.DataFrame(measurements))
    csv(directory / "classification_cis.csv", pd.concat(intervals, ignore_index=True))
    csv(directory / "paired_classification_deltas.csv", pd.concat(deltas, ignore_index=True))
    quality = []
    for condition in run.config["conditions"]:
        prep = Preprocessor(run.data, condition, run.config, dae, device)
        fusion = condition["group"] in ("dwt_features", "dft_features")
        for state in STATES:
            for row in run.data.frames["test"].itertuples(index=False):
                record = {"condition_id": condition["id"], "corruption_id": state["id"],
                          "image": row.image, "patient_id": row.patient_id, "label": row.label,
                          "status": "representation_only" if fusion else "reference_proxy"}
                if not fusion:
                    record.update(image_quality(run.data.working[row.image], prep.image(row.image, state["id"])))
                quality.append(record)
        print(f"Reports: image quality {condition['id']}", flush=True)
    quality = pd.DataFrame(quality)
    csv(directory / "image_quality_per_image.csv", quality)
    summary = quality.groupby(["condition_id", "corruption_id", "status"], sort=False)[["psnr", "ssim", "mse"]].mean().reset_index()
    csv(directory / "image_quality.csv", summary)
    plots(run, predictions, specs, dae, device, directory)
    note = ("# v7 beta pilot — seed 42\n\n"
        "12 CNNs, one shared DAE; 320 distinct train images. Validation selects each checkpoint and one operating threshold.\n"
        "Test is evaluated independently for each of 16 fixed corruption states. Patient-bootstrap 95% CIs use 2,000 draws in full mode.\n\n"
        "Low estimated noise does not prove a noise-free image. PSNR/SSIM use the pre-corruption reference proxy, not detector ground truth.\n"
        "Fusion views are representations; their image-quality entries are intentionally not applicable. Infinite PSNR means exact identity.\n"
        "This is a low-noise selected cohort, not an estimate for the full NIH population. One seed does not measure across-training-seed variability.\n\n"
        "XAI has not been run. The original bbox reference, train-only SHAP background, model_spec.json, native geometry and load_classifier interface are retained.\n")
    if run.config["run_mode"] == "smoke":
        note = "# SMOKE VALIDATION ONLY — RANDOM WEIGHTS, NOT RESEARCH RESULTS\n\n" + note
    (directory / "README.md").write_text(note, encoding="utf-8")
    paths = [p for p in directory.rglob("*") if p.is_file() and p.name != "report_complete.json"]
    run.mark(directory, None, [p.relative_to(directory).as_posix() for p in paths], "report_complete.json")


def plots(run, predictions, specs, dae, device, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    for ax, condition in zip(axes.flat, run.config["conditions"]):
        history = pd.read_csv(run.output / f"CNN/{condition['id']}/history.csv")
        ax.plot(history.epoch, history.train_loss, label="train")
        ax.plot(history.epoch, history.validation_loss, label="validation, 16 states")
        ax.set_title(condition["id"], fontsize=8)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "learning_curves.png", dpi=130)
    plt.close(fig)
    for name, frame in predictions.items():
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for ax, state in zip(axes.flat, STATES):
            selected = frame[frame.corruption_id == state["id"]]
            matrix = np.asarray(classification_metrics(selected.label, selected.prob_infiltration, specs[name]["threshold"])["confusion_matrix"])
            ax.imshow(matrix, cmap="Blues")
            for (y, x), value in np.ndenumerate(matrix):
                ax.text(x, y, str(value), ha="center", va="center")
            ax.set(title=state["id"], xticks=[0, 1], yticks=[0, 1], xlabel="predicted", ylabel="true")
            ax.title.set_fontsize(7)
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(figures / f"confusion_{name}.png", dpi=110)
        plt.close(fig)
    for label in (0, 1):
        row = run.data.frames["test"].query("label == @label").sort_values("image").iloc[0]
        for state in [s for s in STATES if s["level"] == 2]:
            fig, axes = plt.subplots(3, 4, figsize=(11, 8))
            samples = [("reference proxy", run.data.working[row.image]), (state["id"], run.data.noisy(row.image, state["id"])[0])]
            for condition in run.config["conditions"]:
                if condition["group"] not in ("dwt_features", "dft_features"):
                    prep = Preprocessor(run.data, condition, run.config, dae, device)
                    samples.append((condition["id"], prep.image(row.image, state["id"])))
            for ax, (title, image) in zip(axes.flat, samples):
                ax.imshow(image, cmap="gray", vmin=0, vmax=255)
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            fig.suptitle(f"class {label}: {row.image}")
            fig.tight_layout()
            fig.savefig(figures / f"example_class{label}_{state['family']}.png", dpi=110)
            plt.close(fig)
