"""Plot saved CNN histories only; no model imports, training, or run locks."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONDITIONS = ("baseline", "dwt_L1_L4_fusion", "dft_low_mid_high_fusion")


def plot_overfit(run_root, condition_ids=None, display_plots=True):
    run_root = Path(run_root)
    if condition_ids is None:
        condition_ids = [p.parent.name for p in sorted((run_root / "CNN/results").glob("*/history.csv"))]
    frames, missing = {}, []
    for condition in condition_ids:
        history = run_root / "CNN/results" / condition / "history.csv"
        if not history.is_file():
            missing.append(condition)
            continue
        frame = pd.read_csv(history).sort_values("epoch")
        if frame.empty or not {"epoch", "train_loss", "validation_loss"} <= set(frame):
            raise ValueError(f"Invalid or empty history: {history}")
        if frame.epoch.duplicated().any():
            raise ValueError(f"Duplicate epochs: {history}")
        for name in ("epoch", "train_loss", "validation_loss", "train_eval_loss"):
            if name in frame:
                frame[name] = pd.to_numeric(frame[name], errors="raise")
                if not np.isfinite(frame[name]).all():
                    raise ValueError(f"Nonfinite {name}: {history}")
        frames[condition] = frame
    if not frames:
        raise FileNotFoundError(f"No CNN/results/<condition>/history.csv under {run_root}. Set RUN_ROOT to your saved run.")

    def draw(ax, condition, frame):
        if "train_eval_loss" in frame:
            ax.plot(frame.epoch, frame.train_eval_loss, "o-", label="Train (eval mode)", color="#2563eb")
            ax.plot(frame.epoch, frame.train_loss, "--", alpha=0.6, label="Train (dropout on)", color="#6b7280")
        else:
            ax.plot(frame.epoch, frame.train_loss, "o-", label="Train (during updates)", color="#2563eb")
        ax.plot(frame.epoch, frame.validation_loss, "o-", label="Validation", color="#ea580c")
        best = frame.loc[frame.validation_loss.idxmin(), "epoch"]
        ax.axvline(best, linestyle=":", alpha=0.65, color="#15803d", label=f"Best validation: epoch {best:g}")
        ax.set(title=condition, xlabel="Epoch", ylabel="BCE loss", xticks=frame.epoch)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    import math
    columns = min(3, len(frames))
    rows = math.ceil(len(frames) / columns)
    combined, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows), squeeze=False)
    paths, summary = {}, []
    for ax, (condition, frame) in zip(axes.flat, frames.items()):
        draw(ax, condition, frame)
        fig, single = plt.subplots(figsize=(7, 4.5))
        draw(single, condition, frame)
        fig.tight_layout()
        destination = run_root / "CNN/results" / condition / "training_curves.png"
        fig.savefig(destination, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths[condition] = destination
        best = frame.loc[frame.validation_loss.idxmin()]
        last = frame.iloc[-1]
        row = {"condition_id": condition, "epochs_recorded": len(frame),
               "best_validation_epoch": int(best.epoch), "best_validation_loss": float(best.validation_loss),
               "last_epoch": int(last.epoch), "last_validation_loss": float(last.validation_loss)}
        if "train_eval_loss" in frame:
            row.update(last_train_eval_loss=float(last.train_eval_loss),
                       last_validation_minus_train_eval_loss=float(last.validation_loss - last.train_eval_loss))
        summary.append(row)
    for ax in axes.flat[len(frames):]:
        ax.set_visible(False)
    combined.suptitle("Check overfit: train / validation loss")
    combined.tight_layout()
    output = run_root / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    combined_path = output / "training_curves.png"
    combined.savefig(combined_path, dpi=160, bbox_inches="tight")
    plt.close(combined)
    table = pd.DataFrame(summary)
    table.to_csv(output / "overfit_summary.csv", index=False)
    if missing:
        print("No saved history yet:", ", ".join(missing))
    print("Combined plot:", combined_path)
    for condition, path in paths.items():
        print(condition, "plot:", path)
    if display_plots:
        from IPython.display import display, Image
        display(Image(filename=str(combined_path)))
        display(table)
    return {"combined": combined_path, "per_condition": paths, "summary": table, "missing": missing}
