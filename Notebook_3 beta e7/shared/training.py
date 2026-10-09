"""DAE and binary CNN training with provenance checks and epoch resumption."""
from time import perf_counter
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .dae_model import DAE
from .denoising import Preprocessor, read_gray, working_image, synthetic_noise
from .evaluation import classification_metrics
from .experiment import set_seed, save_torch, write_json, file_hash, digest
from .data import balanced_sample
from .models import BinaryResNet50, load_cnn


class ImageDataset(Dataset):
    def __init__(self, frame, preprocessor):
        self.rows = frame.to_dict("records")
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return self.preprocessor.tensor(row), torch.tensor([float(row["label"])])


class DenoiseDataset(Dataset):
    def __init__(self, exp, frame, epoch=0):
        self.exp, self.rows, self.epoch = exp, frame.to_dict("records"), epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        raw = read_gray(self.exp.data_root / row["relative_path"])
        clean = working_image(raw, self.exp.config).astype(np.float32) / 255.0
        rng = np.random.default_rng(np.random.SeedSequence([self.exp.config["seed"], self.epoch, index]))
        noisy = synthetic_noise(clean, self.exp.config["dae"], rng)
        return torch.from_numpy(noisy[None]), torch.from_numpy(clean[None])


def loader(dataset, batch_size, seed, shuffle=False, workers=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def epoch_train_frame(exp, frame, epoch):
    count = exp.config["training"]["images_per_epoch"]
    if count < 2 or count % 2:
        raise ValueError("images_per_epoch must be a positive even total")
    selected = balanced_sample(frame, count // 2, exp.config["seed"], stream=1000 + epoch)
    path = exp.artifacts / "epoch_manifests" / f"epoch_{epoch:03d}.csv"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), selected)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(path, index=False)
    return selected


def loss_epoch(model, batches, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(x)
            n += len(x)
    return total / n


def checkpoint_state(exp, model, optimizer, epoch, best_loss, history, condition_id=None, **extra):
    return {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "best_loss": best_loss, "history": history,
        "provenance": exp.provenance(condition_id), **extra,
    }


def train_dae(exp, manifests, device="cuda"):
    exp.lock()
    device = torch.device(device)
    settings = exp.config["dae"]
    folder = exp.artifacts / "dae_checkpoint"
    marker = folder / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        exp.check_provenance(saved["provenance"])
        exp.load_checkpoint(folder / "best.pt")
        if file_hash(folder / "best.pt") != saved["checkpoint_hash"]:
            raise ValueError("DAE checkpoint changed")
        return folder / "best.pt"
    set_seed(exp.config["seed"])
    model = DAE(base_channels=settings["base_channels"], residual=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    def criterion(prediction, clean):
        return nn.functional.mse_loss(prediction, clean) + settings["l1_weight"] * nn.functional.l1_loss(prediction, clean)
    epoch, best, history = 0, float("inf"), []
    if (folder / "last.pt").exists():
        checkpoint = exp.load_checkpoint(folder / "last.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch, best, history = checkpoint["epoch"], checkpoint["best_loss"], checkpoint["history"]
    validation = loader(DenoiseDataset(exp, manifests["validation"], epoch=0), settings["batch_size"], exp.config["seed"])
    for epoch in range(epoch + 1, settings["epochs"] + 1):
        set_seed(exp.config["seed"] + epoch)
        epoch_frame = epoch_train_frame(exp, manifests["train"], epoch)
        batches = loader(DenoiseDataset(exp, epoch_frame, epoch), settings["batch_size"], exp.config["seed"] + epoch, True)
        started = perf_counter()
        train_loss = loss_epoch(model, batches, criterion, device, optimizer)
        val_loss = loss_epoch(model, validation, criterion, device)
        history.append({"epoch": epoch, "train_images": len(epoch_frame), "train_ids_hash": digest(epoch_frame.image.tolist()), "train_loss": train_loss, "validation_loss": val_loss, "seconds": perf_counter() - started})
        improved = val_loss < best
        best = min(best, val_loss)
        state = checkpoint_state(exp, model, optimizer, epoch, best, history)
        if improved:
            save_torch(folder / "best.pt", state)
        save_torch(folder / "last.pt", state)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"DAE epoch {epoch}: images={len(epoch_frame)}, train={train_loss:.5f}, validation={val_loss:.5f}", flush=True)
    checkpoint = exp.load_checkpoint(folder / "best.pt")
    if checkpoint["epoch"] < 1:
        raise ValueError("DAE did not train")
    write_json(marker, {"provenance": exp.provenance(), "checkpoint_hash": file_hash(folder / "best.pt")})
    return folder / "best.pt"


def predict(model, batches, device):
    model.eval()
    values = []
    with torch.no_grad():
        for x, _ in batches:
            values.extend(model(x.to(device)).sigmoid().flatten().cpu().tolist())
    return np.asarray(values)


def cnn_optimizer(model, settings):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("CNN has no trainable parameters")
    if settings.get("optimizer", "Adam") == "AdamW":
        return torch.optim.AdamW(trainable, lr=settings["learning_rate"],
                                 weight_decay=settings["weight_decay"])
    if settings.get("optimizer", "Adam") != "Adam":
        raise ValueError("Unsupported CNN optimizer")
    return torch.optim.Adam(trainable, lr=settings["learning_rate"])


def train_condition(exp, manifests, condition_id, device="cuda", *,
                    model_factory=BinaryResNet50, preprocessor_factory=Preprocessor):
    exp.lock()
    device = torch.device(device)
    folder = exp.results("CNN", condition_id)
    marker = folder / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        exp.check_provenance(saved["provenance"], condition_id)
        exp.load_checkpoint(folder / "best.pt", condition_id)
        if file_hash(folder / "best.pt") != saved["checkpoint_hash"]:
            raise ValueError("CNN checkpoint changed")
        for name in ("metrics.json", "test_predictions.csv", "history.csv", "confusion_matrix.png"):
            if not (folder / name).is_file():
                raise FileNotFoundError(folder / name)
        return json.loads((folder / "metrics.json").read_text())
    settings = exp.config["cnn"]
    prep = preprocessor_factory(exp, condition_id, device)
    prep.prepare([manifests[n] for n in ("train", "validation", "test")])
    val_loader = loader(ImageDataset(manifests["validation"], prep), settings["batch_size"], exp.config["seed"])
    test_loader = loader(ImageDataset(manifests["test"], prep), settings["batch_size"], exp.config["seed"])
    set_seed(exp.config["seed"])
    model = model_factory(pretrained=True, weights_name=settings["weights"]).to(device)
    optimizer = cnn_optimizer(model, settings)
    epoch, best, history, stale = 0, float("inf"), [], 0
    if (folder / "last.pt").exists():
        saved = exp.load_checkpoint(folder / "last.pt", condition_id)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        epoch, best, history, stale = saved["epoch"], saved["best_loss"], saved["history"], saved["stale"]
    for epoch in range(epoch + 1, settings["epochs"] + 1):
        if stale >= settings["patience"]:
            break
        set_seed(exp.config["seed"] + epoch)
        epoch_frame = epoch_train_frame(exp, manifests["train"], epoch)
        batches = loader(ImageDataset(epoch_frame, prep), settings["batch_size"], exp.config["seed"] + epoch, True, settings["num_workers"])
        started = perf_counter()
        train_loss = loss_epoch(model, batches, nn.BCEWithLogitsLoss(), device, optimizer)
        train_eval_loss = None
        if settings.get("record_train_eval_loss", False):
            eval_batches = loader(ImageDataset(epoch_frame, prep), settings["batch_size"], exp.config["seed"])
            train_eval_loss = loss_epoch(model, eval_batches, nn.BCEWithLogitsLoss(), device)
        val_loss = loss_epoch(model, val_loader, nn.BCEWithLogitsLoss(), device)
        history.append({"epoch": epoch, "train_images": len(epoch_frame), "train_ids_hash": digest(epoch_frame.image.tolist()), "train_loss": train_loss, "validation_loss": val_loss, "seconds": perf_counter() - started})
        if train_eval_loss is not None:
            history[-1]["train_eval_loss"] = train_eval_loss
            history[-1]["validation_minus_train_eval_loss"] = val_loss - train_eval_loss
        improved = val_loss < best
        stale = 0 if improved else stale + 1
        best = min(best, val_loss)
        state = checkpoint_state(exp, model, optimizer, epoch, best, history, condition_id, stale=stale)
        if improved:
            save_torch(folder / "best.pt", state)
        save_torch(folder / "last.pt", state)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        print(f"{condition_id} epoch {epoch}: images={len(epoch_frame)}, train={train_loss:.5f}, validation={val_loss:.5f}", flush=True)
    best_model = load_cnn(exp, condition_id, device, model_factory=model_factory)
    probabilities = predict(best_model, test_loader, device)
    sample = next(iter(test_loader))[0].to(device)
    reloaded = load_cnn(exp, condition_id, device, model_factory=model_factory)
    with torch.no_grad():
        torch.testing.assert_close(best_model(sample), reloaded(sample), rtol=0, atol=1e-6)
    result = classification_metrics(manifests["test"].label, probabilities, settings["classification_threshold"])
    result.update(condition_id=condition_id, epochs_run=len(history), best_epoch=exp.load_checkpoint(folder / "best.pt", condition_id)["epoch"], training_seconds=sum(r["seconds"] for r in history), provenance=exp.provenance(condition_id))
    result["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    predictions = manifests["test"].copy()
    predictions["prob_infiltration"] = probabilities
    predictions["pred_label"] = (probabilities >= settings["classification_threshold"]).astype(int)
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    write_json(folder / "metrics.json", result)
    from .reporting import plot_confusion
    plot_confusion(result["confusion_matrix"], folder / "confusion_matrix.png", condition_id)
    write_json(marker, {"provenance": exp.provenance(condition_id), "checkpoint_hash": file_hash(folder / "best.pt")})
    return result


def run_cnn(exp, manifests, device="cuda"):
    train_dae(exp, manifests, device)
    results = []
    for condition_id in exp.condition_ids:
        results.append(train_condition(exp, manifests, condition_id, device))
        from .reporting import refresh_reports
        refresh_reports(exp)
    return pd.DataFrame([{k: v for k, v in row.items() if k not in ("provenance", "confusion_matrix")} for row in results])
