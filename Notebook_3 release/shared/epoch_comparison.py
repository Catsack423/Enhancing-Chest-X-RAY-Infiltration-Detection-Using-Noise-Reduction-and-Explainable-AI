"""Optional descriptive snapshot comparisons; never choose a model from test scores."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .evaluation import classification_metrics
from .experiment import file_hash, write_json
from .xai import publish_file


def compare_epochs(exp, manifests, condition_ids=None, epochs=(2, 7, 10), device="cuda"):
    from .preprocessing import ReleasePreprocessor
    from .models import load_cnn
    from .reporting import plot_classification_curves, plot_confusion
    exp.lock()
    if exp.cache is None or not getattr(exp.cache, "ready", True):
        from .runtime_cache import RuntimeCache
        RuntimeCache(exp).prepare(manifests)
    selected = list(condition_ids or exp.condition_ids)
    stage = Path(exp.staging_root)/"epoch_comparison"
    destination = exp.output_root/"comparison/epochs"
    stage.mkdir(parents=True, exist_ok=True)
    rows, evidence = [], {}
    for condition in selected:
        exp.condition(condition)
        prep = ReleasePreprocessor(exp, condition, device)
        prep.prepare(manifests)
        for epoch in epochs:
            folder = exp.results("CNN", condition)
            candidates = [folder/f"epoch_{epoch:03}.pt", folder/f"epoch_{epoch}.pt", folder/f"snapshot_epoch_{epoch}.pt"]
            checkpoint = next((p for p in candidates if p.is_file()), None)
            row = {"condition_id":condition, "epoch":epoch, "status":"not_available", "seed":exp.config["seed"],
                   "stage":"head_warmup" if epoch <= exp.config["cnn"].get("warmup_epochs", 3) else exp.config["cnn"]["trainable_layers"]}
            if checkpoint is None:
                rows.append(row)
                continue
            saved = exp.load_checkpoint(checkpoint, condition)
            if saved.get("epoch") != epoch:
                raise ValueError("Snapshot epoch does not match requested epoch")
            model = load_cnn(exp, condition, device=device, checkpoint_name=checkpoint.name)
            predicted = []
            records = manifests["test"].to_dict("records")
            size = exp.config["cnn"]["batch_size"]
            for start in range(0, len(records), size):
                batch = torch.stack([prep.tensor(r) for r in records[start:start+size]]).to(device)
                with torch.no_grad():
                    predicted.extend(model(batch).sigmoid().flatten().cpu().numpy())
            scores = classification_metrics(manifests["test"].label, predicted, exp.config["cnn"]["classification_threshold"])
            row.update({key:value for key,value in scores.items() if key != "confusion_matrix"})
            row["status"] = "complete"
            row["checkpoint_hash"] = file_hash(checkpoint)
            evidence[f"{condition}/epoch_{epoch}"] = row["checkpoint_hash"]
            results = manifests["test"][["image", "patient_id", "label"]].copy()
            results["prob_infiltration"] = predicted
            target = stage/condition/f"epoch_{epoch}"
            target.mkdir(parents=True, exist_ok=True)
            results.to_csv(target/"test_predictions.csv", index=False)
            plot_confusion(scores["confusion_matrix"], target/"confusion.png", f"{condition} epoch {epoch}")
            plot_classification_curves(results, target, f"{condition} epoch {epoch}")
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(stage/"classification_snapshots.csv", index=False)
    (stage/"README.md").write_text("Snapshots are descriptive and do not replace validation-selected best.pt. Epoch 2 vs 7 crosses head warm-up and layer4 fine-tuning, so it does not isolate epoch count. Missing snapshots stay unavailable. Data and model protocol differ from older releases; this does not isolate the effect of adding data.\n", encoding="utf-8")
    hashes = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "complete.json":
            relative = path.relative_to(stage)
            publish_file(path, destination/relative)
            hashes[relative.as_posix()] = file_hash(destination/relative)
    write_json(stage/"complete.json", {"provenance":exp.provenance(), "checkpoint_hashes":evidence, "files":hashes})
    publish_file(stage/"complete.json", destination/"complete.json")
    return frame
