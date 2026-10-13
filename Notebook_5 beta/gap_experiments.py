"""Diagnose the gap between our NIH results and a small VinBigData notebook.

The reference notebook (VinBigData, 100+100 train, 50+50 test, full fine-tune,
7 epochs) reports ROC-AUC ~0.92 while release v4 reports ~0.67 on NIH. This
module replays that reference protocol on NIH so each difference can be
switched on or off in isolation:

- positive label: Infiltration-only (ours) or any Infiltration (reference)
- trainable layers: all (reference) or the v4 head/layer4 protocol
- train size and test scope: balanced small subset or the full official test

Every run is patient-disjoint: train/validation come from train_val_list.txt,
test from test_list.txt, and validation patients never appear in training.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_index(data_root):
    data_root = Path(data_root)
    entries = pd.read_csv(data_root / "Data_Entry_2017.csv")
    entries = entries.rename(columns={"Image Index": "image", "Finding Labels": "labels",
                                      "Patient ID": "patient_id"})
    paths = {p.name: p for p in data_root.glob("images_*/images/*.png")}
    entries["path"] = entries["image"].map(paths)
    missing = entries["path"].isna().sum()
    if missing:
        raise FileNotFoundError(f"{missing} images listed in Data_Entry_2017.csv are missing")
    official = set((data_root / "test_list.txt").read_text().split())
    entries["official_split"] = np.where(entries["image"].isin(official), "test", "train_val")
    return entries[["image", "labels", "patient_id", "path", "official_split"]]


def label_pool(entries, positive):
    negatives = entries[entries["labels"] == "No Finding"].assign(target=0)
    if positive == "only":
        positives = entries[entries["labels"] == "Infiltration"]
    elif positive == "any":
        positives = entries[entries["labels"].str.split("|").apply(lambda x: "Infiltration" in x)]
    else:
        raise ValueError(f"Unknown positive definition: {positive}")
    return pd.concat([negatives, positives.assign(target=1)], ignore_index=True)


def sample_balanced(pool, per_class, rng, exclude_patients=()):
    pool = pool[~pool["patient_id"].isin(set(exclude_patients))]
    parts = []
    for target in (0, 1):
        candidates = pool[pool["target"] == target]
        if len(candidates) < per_class:
            raise ValueError(f"Only {len(candidates)} images for class {target}")
        parts.append(candidates.sample(n=per_class, random_state=rng.integers(2**31)))
    return pd.concat(parts).sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)


def build_splits(entries, positive, train_per_class, val_per_class, test_per_class, seed):
    """test_per_class=None keeps the whole filtered official test set."""
    rng = np.random.default_rng(seed)
    pool = label_pool(entries, positive)
    train_val = pool[pool["official_split"] == "train_val"]
    test_pool = pool[pool["official_split"] == "test"]
    train = sample_balanced(train_val, train_per_class, rng)
    val = sample_balanced(train_val, val_per_class, rng, exclude_patients=train["patient_id"])
    test = (test_pool.reset_index(drop=True) if test_per_class is None
            else sample_balanced(test_pool, test_per_class, rng))
    assert not set(train["patient_id"]) & set(val["patient_id"])
    assert not (set(train["patient_id"]) | set(val["patient_id"])) & set(test["patient_id"])
    return train, val, test


class ImageCache:
    """Grayscale uint8 images resized once to size x size and stored as .npy."""

    def __init__(self, cache_dir, size=224):
        self.dir = Path(cache_dir) / f"gray_{size}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.size = size

    def get(self, row):
        path = self.dir / f"{Path(row['image']).stem}.npy"
        if path.exists():
            return np.load(path)
        with Image.open(row["path"]) as image:
            array = np.array(image.convert("L").resize((self.size, self.size),
                                                        Image.BILINEAR), dtype=np.uint8)
        np.save(path, array)
        return array


class CXRDataset(Dataset):
    def __init__(self, frame, cache, augment=False, shuffle_labels_seed=None):
        self.frame = frame.reset_index(drop=True)
        self.cache = cache
        self.augment = augment
        self.targets = self.frame["target"].to_numpy()
        if shuffle_labels_seed is not None:
            self.targets = np.random.default_rng(shuffle_labels_seed).permutation(self.targets)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        gray = torch.from_numpy(self.cache.get(self.frame.loc[index])).float().div(255)
        image = gray.unsqueeze(0).expand(3, -1, -1)
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
        return (image - MEAN) / STD, int(self.targets[index])


def build_model(trainable):
    from torchvision.models import ResNet50_Weights, resnet50
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if trainable == "all":
        return model
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("fc.") or
                                 (trainable == "layer4" and name.startswith("layer4.")))
    return model


def set_frozen_bn(model, trainable):
    """v4 keeps every BatchNorm in eval mode; the reference trains them all."""
    if trainable != "all":
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores, targets = [], []
    for images, labels in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images.to(device, non_blocking=True))
        scores.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
        targets.append(labels)
    return torch.cat(targets).numpy(), torch.cat(scores).numpy()


def metrics(targets, scores, threshold=0.5, bootstrap=1000, seed=0):
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 precision_score, recall_score, roc_auc_score)
    predictions = (scores >= threshold).astype(int)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(bootstrap):
        draw = rng.integers(0, len(targets), len(targets))
        if targets[draw].min() != targets[draw].max():
            aucs.append(roc_auc_score(targets[draw], scores[draw]))
    return {
        "n": int(len(targets)), "prevalence": float(targets.mean()),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc_score(targets, scores),
        "roc_auc_ci_low": float(np.percentile(aucs, 2.5)),
        "roc_auc_ci_high": float(np.percentile(aucs, 97.5)),
        "pr_auc": average_precision_score(targets, scores),
    }


def run(entries, cache, *, positive="only", trainable="all", train_per_class=100,
        val_per_class=20, test_per_class=50, epochs=7, lr=1e-4, batch_size=32, seed=42,
        shuffle_labels=False, num_workers=0, device=None, verbose=True):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val, test = build_splits(entries, positive, train_per_class, val_per_class,
                                    test_per_class, seed)
    loader = lambda frame, **kw: DataLoader(
        CXRDataset(frame, cache, **kw), batch_size=batch_size, shuffle=kw.get("augment", False),
        num_workers=num_workers, pin_memory=device.type == "cuda")
    train_loader = loader(train, augment=True,
                          shuffle_labels_seed=seed if shuffle_labels else None)
    val_loader, test_loader = loader(val), loader(test)

    model = build_model(trainable).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss()
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        set_frozen_bn(model, trainable)
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        val_targets, val_scores = predict(model, val_loader, device)
        val_acc = float(((val_scores >= 0.5) == val_targets).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / seen,
                        "train_acc": correct / seen, "val_acc": val_acc})
        if verbose:
            print(f"  epoch {epoch}/{epochs} loss {total_loss / seen:.4f} "
                  f"train_acc {correct / seen:.3f} val_acc {val_acc:.3f}")
    targets, scores = predict(model, test_loader, device)
    result = {"positive": positive, "trainable": trainable, "train_per_class": train_per_class,
              "test_per_class": test_per_class or "full", "epochs": epochs, "seed": seed,
              "shuffle_labels": shuffle_labels, "seconds": round(time.time() - started, 1),
              **metrics(targets, scores, seed=seed), "history": history}
    return result, test.assign(score=scores)


def resample_predictions(frame, per_class=50, draws=1000, seed=0):
    """Score existing full-test predictions on many reference-sized balanced draws."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(draws):
        parts = [frame[frame["label"] == t].sample(n=per_class, random_state=rng.integers(2**31))
                 for t in (0, 1)]
        rows.append(_point_metrics(pd.concat(parts)))
    return pd.DataFrame(rows)


def _point_metrics(draw):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    targets, scores = draw["label"].to_numpy(), draw["prob_infiltration"].to_numpy()
    return {"roc_auc": roc_auc_score(targets, scores),
            "accuracy@0.5": accuracy_score(targets, scores >= 0.5),
            "f1@0.5": f1_score(targets, scores >= 0.5, zero_division=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1]
                                                    / "data" / "versions" / "3"))
    parser.add_argument("--cache-dir", default=str(Path(__file__).resolve().parent / ".cache"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"
                                              / "runs.jsonl"))
    parser.add_argument("--positive", choices=["only", "any"], default="only")
    parser.add_argument("--trainable", choices=["all", "layer4", "head"], default="all")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=50,
                        help="0 keeps the whole filtered official test set")
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    entries = load_index(args.data_root)
    cache = ImageCache(args.cache_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        print(f"[{args.tag or 'run'}] positive={args.positive} trainable={args.trainable} "
              f"train={args.train_per_class}/class seed={seed}")
        result, _ = run(entries, cache, positive=args.positive, trainable=args.trainable,
                        train_per_class=args.train_per_class, val_per_class=args.val_per_class,
                        test_per_class=args.test_per_class or None, epochs=args.epochs,
                        lr=args.lr, batch_size=args.batch_size, seed=seed,
                        shuffle_labels=args.shuffle_labels, num_workers=args.num_workers)
        result["tag"] = args.tag
        print(f"  -> AUC {result['roc_auc']:.3f} [{result['roc_auc_ci_low']:.3f}, "
              f"{result['roc_auc_ci_high']:.3f}] acc {result['accuracy']:.3f} "
              f"f1 {result['f1']:.3f} ({result['seconds']}s)")
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
