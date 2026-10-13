"""Build gap_experiments.ipynb from the cell sources below."""
import json
from pathlib import Path

CELLS = [
    ("markdown", """# v5 beta — why does the reference notebook score so much higher?

The reference notebook (`Seminae_source_code (1).ipynb`, VinBigData 512px) reports ROC-AUC ~0.92
with 100+100 training images; release v4 reports ~0.67 on NIH. These cells replay the reference
protocol on NIH and switch one difference at a time. See README.md for the hypotheses."""),
    ("code", """import os, sys, json
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_ROOT = Path("/content/cxr_v3_originals")   # extracted by the v3/v4 setup cell
    WORK = Path("/content/drive/MyDrive/nih -xray/Notebook_5_beta")
else:
    WORK = Path.cwd()
    DATA_ROOT = WORK.parent / "data" / "versions" / "3"
sys.path.insert(0, str(WORK))
import pandas as pd
import gap_experiments as gx
RESULTS = WORK / "results"
RESULTS.mkdir(exist_ok=True)
entries = gx.load_index(DATA_ROOT)
cache = gx.ImageCache(Path("/content/gap_cache") if IN_COLAB else WORK / ".cache")
len(entries)"""),
    ("markdown", """## E0 — would v4 look like 0.92 on a 50+50 test draw?
Re-score existing v4 full-test predictions on 1,000 reference-sized balanced draws."""),
    ("code", """from sklearn.metrics import roc_auc_score
rows = []
for path in sorted((WORK.parent / "comparison" / "epochs").glob("*/epoch_7/test_predictions.csv")):
    frame = pd.read_csv(path)
    draws = gx.resample_predictions(frame)
    rows.append({"condition": path.parts[-3],
                 "full_test_auc": roc_auc_score(frame.label, frame.prob_infiltration),
                 "draw_auc_p2.5": draws.roc_auc.quantile(0.025),
                 "draw_auc_median": draws.roc_auc.median(),
                 "draw_auc_p97.5": draws.roc_auc.quantile(0.975),
                 "draw_auc_max": draws.roc_auc.max()})
pd.DataFrame(rows).round(3)"""),
    ("markdown", """## E1–E5 — replay and ablate the reference protocol on NIH
Each `run` is patient-disjoint and appends one row to `results/runs.jsonl`."""),
    ("code", """MATRIX = [
    ("E1_reference_only",   dict(positive="only"), [42, 43, 44, 45, 46]),
    ("E2_reference_any",    dict(positive="any"),  [42, 43, 44, 45, 46]),
    ("E3_shuffled_labels",  dict(positive="only", shuffle_labels=True), [42, 43, 44]),
    ("E4_layer4_only",      dict(positive="only", trainable="layer4"), [42, 43, 44]),
    ("E4_head_only",        dict(positive="only", trainable="head"), [42, 43, 44]),
    ("E5_full_ft_2000",     dict(positive="only", train_per_class=2000, val_per_class=200,
                                 test_per_class=None), [42]),
    ("E5_full_ft_2000_any", dict(positive="any", train_per_class=2000, val_per_class=200,
                                 test_per_class=None), [42]),
]
done = set()
if (RESULTS / "runs.jsonl").exists():
    done = {(r["tag"], r["seed"]) for r in map(json.loads, (RESULTS / "runs.jsonl").open())}
for tag, kwargs, seeds in MATRIX:
    for seed in seeds:
        if (tag, seed) in done:
            continue
        print(tag, seed)
        result, _ = gx.run(entries, cache, seed=seed, num_workers=2 if IN_COLAB else 0, **kwargs)
        result["tag"] = tag
        with (RESULTS / "runs.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\\n")"""),
    ("code", """runs = pd.read_json(RESULTS / "runs.jsonl", lines=True)
summary = runs.groupby("tag", sort=False).agg(
    n_seeds=("seed", "size"), test_n=("n", "first"),
    auc_mean=("roc_auc", "mean"), auc_sd=("roc_auc", "std"),
    auc_min=("roc_auc", "min"), auc_max=("roc_auc", "max"),
    acc_mean=("accuracy", "mean"), f1_mean=("f1", "mean"))
summary.round(3)"""),
]


def main():
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
                "metadata": {"accelerator": "GPU",
                             "kernelspec": {"name": "python3", "display_name": "Python 3"},
                             "language_info": {"name": "python"}}}
    out = Path(__file__).with_name("gap_experiments.ipynb")
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
