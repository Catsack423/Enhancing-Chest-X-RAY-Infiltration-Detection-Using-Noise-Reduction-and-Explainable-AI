"""Full-manifest training with exact accumulation and durable epoch resumption."""
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dae_model import DAE
from .experiment import deterministic_backend, digest, file_hash, save_torch, set_seed, write_json
from .models import ReleaseResNet50, load_cnn


def random_affine(views, rng, settings):
    """One small rotation/shift/zoom shared by every view; intensities are untouched.

    Geometry only: brightness, contrast or noise augmentation would interact with
    the denoising and CLAHE conditions under study.
    """
    degrees = float(settings.get("max_rotation_degrees", 0.0))
    shift = float(settings.get("max_translation_fraction", 0.0))
    low, high = settings.get("scale_range", [1.0, 1.0])
    if degrees < 0 or not 0 <= shift < 0.5 or not 0 < low <= high:
        raise ValueError("Invalid geometric augmentation settings")
    angle = np.deg2rad(rng.uniform(-degrees, degrees))
    scale = rng.uniform(low, high)
    tx, ty = 2 * rng.uniform(-shift, shift, size=2)  # affine_grid spans [-1, 1]
    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=views.dtype).expand(len(views), 2, 3)
    grid = nn.functional.affine_grid(theta, list(views.shape), align_corners=False)
    return nn.functional.grid_sample(views, grid, mode="bilinear", padding_mode=settings.get("padding", "reflection"),
                                     align_corners=False)


class ImageDataset(Dataset):
    def __init__(self, frame, preprocessor, augmentation=None, seed=0, epoch=0):
        self.rows = frame.to_dict("records")
        self.preprocessor = preprocessor
        self.augmentation, self.seed, self.epoch = augmentation, int(seed), int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tensor = self.preprocessor.tensor(row)
        if self.augmentation:
            # Seeded per image like DenoiseDataset, so worker count and resume keep the same draws.
            identity = int(digest(row["image"])[:8], 16)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, identity]))
            tensor = random_affine(tensor, rng, self.augmentation)
        return tensor, torch.tensor([float(row["label"])])


def synthetic_noise(clean, settings, rng):
    """The inherited Gaussian/Poisson corruption; raw NIH is the target proxy."""
    noisy = clean + rng.normal(0, settings.get("gaussian_sigma", 0.05), clean.shape).astype(np.float32)
    scale = settings.get("poisson_scale", 30.0)
    poisson = rng.poisson(np.maximum(noisy * scale, 0)).astype(np.float32) / scale
    return np.clip(0.5 * noisy + 0.5 * poisson, 0, 1).astype(np.float32)


class DenoiseDataset(Dataset):
    def __init__(self, exp, frame, epoch=0):
        self.exp, self.rows, self.epoch = exp, frame.to_dict("records"), epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        clean = self.exp.cache.working(self.rows[index]).astype(np.float32) / 255.0
        # Stable image identity keeps class-stratum validation corruption equal
        # to that image's corruption in the complete validation frame.
        identity = int(digest(self.rows[index]["image"])[:8], 16)
        rng = np.random.default_rng(np.random.SeedSequence([self.exp.config["seed"], self.epoch, identity]))
        noisy = synthetic_noise(clean, self.exp.config["dae"], rng)
        return torch.from_numpy(noisy[None]), torch.from_numpy(clean[None])


def loader(dataset, batch_size, seed, shuffle=False, workers=0):
    if batch_size < 1 or len(dataset) < 1:
        raise ValueError("A loader requires a nonempty dataset and positive batch size")
    generator = torch.Generator().manual_seed(int(seed))
    # Items are seeded per image, so worker count changes speed but not the data.
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, generator=generator,
                      num_workers=int(workers), pin_memory=torch.cuda.is_available(), drop_last=False,
                      persistent_workers=int(workers) > 0)


def capture_rng_state():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": {
        "kind": numpy[0], "keys": numpy[1].tolist(), "position": int(numpy[2]),
        "has_gauss": int(numpy[3]), "cached_gaussian": float(numpy[4])},
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy["kind"], np.asarray(numpy["keys"], dtype=np.uint32),
                         numpy["position"], numpy["has_gauss"], numpy["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG topology changed; resume on the recorded device setup")
        torch.cuda.set_rng_state_all(state["cuda"])


def _autocast(device, amp):
    device = torch.device(device)
    return torch.autocast(device_type="cuda", dtype=torch.float16) if amp and device.type == "cuda" else nullcontext()


def _scaler(device, amp):
    return torch.amp.GradScaler("cuda", enabled=bool(amp and torch.device(device).type == "cuda"))


def _finite_loss(loss):
    if not torch.isfinite(loss).all():
        raise ValueError("Nonfinite loss; checkpoint was not updated")


def optimization_epoch(model, batches, optimizer, pos_weight, device, *, accumulation_steps=4,
                       scaler=None, amp=False, return_stats=False):
    """Mean weighted BCE over SOURCE images, including the partial last window.

    Backpropagate sums per microbatch, then divide accumulated gradients by the
    actual source-image count in that optimizer step.  A final 12-image batch
    therefore uses 12 rather than the nominal effective size of 16.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    scaler = scaler or _scaler(device, amp)
    weight = torch.tensor([float(pos_weight)], device=device)
    optimizer.zero_grad(set_to_none=True)
    total, seen, window_images, window_batches = 0.0, 0, 0, 0
    successful_steps, skipped_steps = 0, 0
    for index, (x, y) in enumerate(batches):
        x, y = x.to(device), y.to(device)
        with _autocast(device, amp):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight, reduction="sum")
        _finite_loss(loss)
        scaler.scale(loss).backward()
        total += float(loss.detach())
        seen += len(x)
        window_images += len(x)
        window_batches += 1
        if window_batches == accumulation_steps or index + 1 == len(batches):
            scaler.unscale_(optimizer)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.div_(window_images)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler reduces its scale after finding nonfinite gradients
            # and skipping optimizer.step. Stable/growing scale means a real
            # update. Disabled scalers always execute the step normally.
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                skipped_steps += 1
            else:
                successful_steps += 1
            optimizer.zero_grad(set_to_none=True)
            window_images, window_batches = 0, 0
    if not seen:
        raise ValueError("Empty training epoch")
    if not successful_steps:
        raise RuntimeError("No optimizer updates occurred in this epoch; AMP skipped every step because gradients overflowed")
    stats = {"optimizer_steps": successful_steps, "skipped_amp_steps": skipped_steps,
             "attempted_optimizer_steps": successful_steps + skipped_steps, "amp_scale": float(scaler.get_scale())}
    return (total / seen, stats) if return_stats else total / seen


def balanced_loss_epoch(model, batches, device):
    """0.5 * negative mean BCE + 0.5 * positive mean BCE in eval mode."""
    model.eval()
    sums, counts = np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.int64)
    with torch.no_grad():
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            losses = nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none").flatten()
            _finite_loss(losses)
            labels = y.flatten()
            for label in (0, 1):
                selected = labels == label
                sums[label] += float(losses[selected].sum())
                counts[label] += int(selected.sum())
    if (counts == 0).any():
        raise ValueError("Balanced BCE requires both classes in the complete manifest")
    return float(0.5 * (sums / counts).sum())


def predict(model, batches, device):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for x, _ in batches:
            values = model(x.to(device)).sigmoid().flatten()
            _finite_loss(values)
            probabilities.extend(values.cpu().tolist())
    return np.asarray(probabilities, dtype=np.float64)


def cnn_optimizer(model, settings):
    # Register late parameters even during warmup so the group layout and
    # optimizer state stay identical across a boundary resume.
    groups = [{"params": list(model.head.parameters()), "lr": settings.get("head_learning_rate", settings.get("learning_rate", 1e-4)), "name": "head"}]
    if model.trainable_layers == "layer4_after_warmup":
        groups.append({"params": list(model.backbone.layer4.parameters()), "lr": settings.get("layer4_learning_rate", 1e-5), "name": "layer4"})
    return torch.optim.AdamW(groups, weight_decay=settings.get("weight_decay", 1e-4))


def checkpoint_state(exp, model, optimizer, scaler, epoch, best_loss, history,
                     condition_id=None, **extra):
    return {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "rng_state": capture_rng_state(),
            "epoch": int(epoch), "best_loss": float(best_loss), "history": history,
            "provenance": exp.provenance(condition_id), **extra}


def _write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _resume_checkpoint(exp, folder, condition_id=None):
    last = folder / "last.pt"
    best = folder / "best.pt"
    if not last.exists() and not best.exists():
        return None
    candidates = [exp.load_checkpoint(path, condition_id) for path in (last, best) if path.exists()]
    # best is written before last. If the VM disconnected between those atomic
    # writes, the newer best checkpoint is itself a valid resumable epoch.
    checkpoint = max(candidates, key=lambda state: state["epoch"])
    if not best.is_file():
        raise ValueError("Resume checkpoint exists without best.pt")
    return checkpoint


def _repair_epoch_writes(exp, folder, checkpoint, condition_id=None, snapshots=()):
    """Finish a checkpoint transaction interrupted after best.pt or last.pt."""
    last = folder / "last.pt"
    if not last.exists() or exp.load_checkpoint(last, condition_id)["epoch"] < checkpoint["epoch"]:
        save_torch(last, checkpoint)
    snapshot = folder / f"epoch_{checkpoint['epoch']:03d}.pt"
    if checkpoint["epoch"] in snapshots and not snapshot.exists():
        save_torch(snapshot, checkpoint)
    _write_csv(folder / "history.csv", pd.DataFrame(checkpoint["history"]))


def _restore_training(checkpoint, model, optimizer, scaler, *, warmup_epochs=None):
    epoch = int(checkpoint["epoch"])
    if warmup_epochs is not None:
        expected = model.stage_for_epoch(epoch + 1, warmup_epochs)
        if checkpoint.get("next_stage") != expected:
            raise ValueError("Resume stage differs from recorded training protocol")
        if checkpoint.get("stage") != model.stage_for_epoch(epoch, warmup_epochs):
            raise ValueError("Checkpoint has an invalid training stage")
        model.set_stage(epoch, warmup_epochs)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    history = checkpoint["history"]
    if [r["epoch"] for r in history] != list(range(1, epoch + 1)):
        raise ValueError("Checkpoint history must contain every completed epoch")
    return epoch, float(checkpoint["best_loss"]), history


def _early_stop(history, settings, warmup_epochs):
    """True once validation loss has not improved for `patience` epochs, counted after warm-up.

    The best epoch is the first minimum, matching the strict improvement rule for best.pt.
    Counting from the later of best epoch and warm-up gives layer4 at least `patience` epochs.
    """
    patience = settings.get("early_stopping_patience")
    if not patience or not history:
        return False
    best_epoch = int(np.argmin([row["validation_loss"] for row in history])) + 1
    return len(history) - max(best_epoch, int(warmup_epochs)) >= int(patience)


def _completed(exp, folder, condition_id=None):
    marker = folder / "complete.json"
    if not marker.is_file():
        return None
    saved = json.loads(marker.read_text(encoding="utf-8"))
    exp.check_provenance(saved["provenance"], condition_id)
    exp.load_checkpoint(folder / "best.pt", condition_id)
    for name, key in (("best.pt", "checkpoint_hash"), ("history.csv", "history_hash"),
                      ("test_predictions.csv", "predictions_hash"), ("metrics.json", "metrics_hash"),
                      ("validation_predictions.csv", "validation_predictions_hash")):
        if condition_id is None and name in ("test_predictions.csv", "metrics.json", "validation_predictions.csv"):
            continue
        if file_hash(folder / name) != saved[key]:
            raise ValueError(f"Completed artifact changed: {folder / name}")
    return saved


def _reconstruction_epoch(model, batches, criterion, device, optimizer=None, scaler=None, amp=False):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for noisy, clean in batches:
            noisy, clean = noisy.to(device), clean.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with _autocast(device, amp):
                loss = criterion(model(noisy), clean)
            _finite_loss(loss)
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total += float(loss.detach()) * len(noisy)
            count += len(noisy)
    if not count:
        raise ValueError("Empty DAE epoch")
    return total / count


def _dae_class_reconstruction(model, exp, frame, settings, device, criterion):
    result = {}
    for label in (0, 1):
        subset = frame[frame.label == label]
        if len(subset):
            batches = loader(DenoiseDataset(exp, subset), settings["batch_size"], exp.config["seed"],
                             workers=settings.get("num_workers", 0))
            result[str(label)] = _reconstruction_epoch(model, batches, criterion, device)
    return result


def _training_backend(model, settings, device):
    """Apply the configured speed/determinism trade-off for one training run."""
    if settings.get("channels_last", False):
        model = model.to(device, memory_format=torch.channels_last)
    deterministic_backend(settings.get("deterministic", True))
    return model


def _ensure_cache(exp, manifests):
    if exp.cache is None:
        from .runtime_cache import RuntimeCache
        exp.cache = RuntimeCache(exp)
    if not getattr(exp.cache, "ready", False):
        exp.cache.prepare(manifests)


def train_dae(exp, manifests, device="cuda"):
    exp.lock()
    device = torch.device(device)
    settings = exp.config["dae"]
    folder = exp.artifacts / "dae_checkpoint"
    if _completed(exp, folder) is not None:
        return folder / "best.pt"
    _ensure_cache(exp, manifests)
    set_seed(exp.config["seed"])
    model = _training_backend(DAE(base_channels=settings.get("base_channels", 32), residual=True).to(device), settings, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.get("learning_rate", 1e-3))
    amp = settings.get("amp", exp.config["cnn"].get("amp", True))
    scaler = _scaler(device, amp)
    def criterion(prediction, clean):
        return nn.functional.mse_loss(prediction, clean) + settings.get("l1_weight", 0.2) * nn.functional.l1_loss(prediction, clean)
    epoch, best, history = 0, float("inf"), []
    checkpoint = _resume_checkpoint(exp, folder)
    if checkpoint:
        epoch, best, history = _restore_training(checkpoint, model, optimizer, scaler)
        _repair_epoch_writes(exp, folder, checkpoint)
    workers = settings.get("num_workers", 0)
    validation = loader(DenoiseDataset(exp, manifests["validation"]), settings["batch_size"], exp.config["seed"], workers=workers)
    try:
        for epoch in range(epoch + 1, settings["epochs"] + 1):
            batches = loader(DenoiseDataset(exp, manifests["train"], epoch), settings["batch_size"], exp.config["seed"] + epoch, True, workers)
            started = perf_counter()
            train_loss = _reconstruction_epoch(model, batches, criterion, device, optimizer, scaler, amp)
            val_loss = _reconstruction_epoch(model, validation, criterion, device)
            by_class = _dae_class_reconstruction(model, exp, manifests["validation"], settings, device, criterion)
            history.append({"epoch": epoch, "train_images": len(manifests["train"]), "train_ids_hash": digest(manifests["train"].image.tolist()),
                            "train_loss": train_loss, "validation_loss": val_loss, "validation_normal_loss": by_class.get("0"),
                            "validation_infiltration_loss": by_class.get("1"), "seconds": perf_counter() - started})
            improved, best = val_loss < best, min(best, val_loss)
            state = checkpoint_state(exp, model, optimizer, scaler, epoch, best, history)
            if improved:
                save_torch(folder / "best.pt", state)
            save_torch(folder / "last.pt", state)
            _write_csv(folder / "history.csv", pd.DataFrame(history))
            print(f"DAE epoch {epoch}/{settings['epochs']}: all {len(manifests['train'])} images, train={train_loss:.5f}, validation={val_loss:.5f}", flush=True)
    finally:
        # Denoising inference, evaluation and XAI stay bitwise deterministic.
        deterministic_backend(True)
    best_checkpoint = exp.load_checkpoint(folder / "best.pt")
    last_checkpoint = exp.load_checkpoint(folder / "last.pt")
    if last_checkpoint["epoch"] != settings["epochs"]:
        raise ValueError("DAE completion requires the entire configured epoch budget")
    _write_csv(folder / "history.csv", pd.DataFrame(last_checkpoint["history"]))
    write_json(folder / "complete.json", {"provenance": exp.provenance(), "checkpoint_hash": file_hash(folder / "best.pt"),
                                          "history_hash": file_hash(folder / "history.csv"), "best_epoch": best_checkpoint["epoch"], "epochs_run": settings["epochs"]})
    return folder / "best.pt"


def train_condition(exp, manifests, condition_id, device="cuda", *,
                    model_factory=ReleaseResNet50, preprocessor_factory=None):
    exp.lock()
    device = torch.device(device)
    folder = exp.results("CNN", condition_id)
    if _completed(exp, folder, condition_id) is not None:
        return json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    if preprocessor_factory is None:
        from .preprocessing import ReleasePreprocessor
        preprocessor_factory = ReleasePreprocessor
    _ensure_cache(exp, manifests)
    settings = exp.config["cnn"]
    if settings.get("pretrained", True) is False and exp.config.get("run_mode") != "smoke":
        raise ValueError("Full release training requires pretrained ImageNet weights")
    deterministic_backend(True)  # DAE denoising cache must not inherit a fast training backend.
    prep = preprocessor_factory(exp, condition_id, device)
    prep.prepare([manifests[name] for name in ("train", "validation", "test")])
    train = manifests["train"]
    counts = train.label.value_counts()
    if set(counts.index) != {0, 1}:
        raise ValueError("Training manifest must contain the two binary classes")
    pos_weight = float(counts.loc[0] / counts.loc[1])
    batch_size, seed = settings["batch_size"], exp.config["seed"]
    validation = loader(ImageDataset(manifests["validation"], prep), batch_size, seed)
    train_eval = loader(ImageDataset(train, prep), batch_size, seed)
    checkpoint = _resume_checkpoint(exp, folder, condition_id)
    set_seed(seed)
    model = model_factory(pretrained=settings.get("pretrained", True) if checkpoint is None else False, weights_name=settings.get("weights", "IMAGENET1K_V2"),
                          dropout=settings.get("head_dropout", 0.3), trainable_layers=settings.get("trainable_layers", "layer4_after_warmup")).to(device)
    model = _training_backend(model, settings, device)
    optimizer = cnn_optimizer(model, settings)
    amp = settings.get("amp", True)
    scaler = _scaler(device, amp)
    warmup = settings.get("warmup_epochs", 3)
    epoch, best, history = 0, float("inf"), []
    if checkpoint:
        if checkpoint.get("pos_weight") != pos_weight:
            raise ValueError("Train class counts changed on resume")
        epoch, best, history = _restore_training(checkpoint, model, optimizer, scaler, warmup_epochs=warmup)
        _repair_epoch_writes(exp, folder, checkpoint, condition_id, settings.get("snapshot_epochs", [2, 7, 10]))
    for epoch in range(epoch + 1, settings["epochs"] + 1):
        if _early_stop(history, settings, warmup):
            print(f"{condition_id}: early stop after epoch {epoch - 1}; no validation improvement for "
                  f"{settings['early_stopping_patience']} epochs", flush=True)
            break
        stage = model.set_stage(epoch, warmup)
        batches = loader(ImageDataset(train, prep, settings.get("augmentation"), seed, epoch), batch_size, seed + epoch, True,
                         settings.get("num_workers", 0))
        started = perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_loss, optimizer_stats = optimization_epoch(model, batches, optimizer, pos_weight, device, accumulation_steps=settings.get("accumulation_steps", 4), scaler=scaler, amp=amp, return_stats=True)
        train_eval_loss = balanced_loss_epoch(model, train_eval, device)
        val_loss = balanced_loss_epoch(model, validation, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        history.append({"epoch": epoch, "stage": stage, "train_images": len(train), "train_ids_hash": digest(train.image.tolist()),
                        "optimization_loss": train_loss, "train_loss": train_loss, "train_eval_loss": train_eval_loss,
                        "validation_loss": val_loss, "validation_minus_train_eval_loss": val_loss - train_eval_loss,
                        "pos_weight": pos_weight, "effective_batch": batch_size * settings.get("accumulation_steps", 4),
                        "seconds": perf_counter() - started, "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                        **optimizer_stats})
        improved, best = val_loss < best, min(best, val_loss)
        state = checkpoint_state(exp, model, optimizer, scaler, epoch, best, history, condition_id,
                                 stage=stage, next_stage=model.stage_for_epoch(epoch + 1, warmup), pos_weight=pos_weight)
        if improved:
            save_torch(folder / "best.pt", state)
        save_torch(folder / "last.pt", state)
        if epoch in settings.get("snapshot_epochs", [2, 7, 10]):
            save_torch(folder / f"epoch_{epoch:03d}.pt", state)
        _write_csv(folder / "history.csv", pd.DataFrame(history))
        print(f"{condition_id} epoch {epoch}/{settings['epochs']} ({stage}): all {len(train)} images, balanced train={train_eval_loss:.5f}, validation={val_loss:.5f}", flush=True)
    return evaluate_classifier(exp, manifests, condition_id, device, model_factory=model_factory, preprocessor_factory=preprocessor_factory)


def evaluate_classifier(exp, manifests, condition_id, device="cuda", *,
                        model_factory=ReleaseResNet50, preprocessor_factory=None):
    """Evaluate the validation-selected model; test never chooses a checkpoint."""
    exp.lock()
    deterministic_backend(True)
    folder, settings = exp.results("CNN", condition_id), exp.config["cnn"]
    if _completed(exp, folder, condition_id) is not None:
        return json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    last = exp.load_checkpoint(folder / "last.pt", condition_id)
    if last["epoch"] != settings["epochs"] and not _early_stop(last["history"], settings, settings.get("warmup_epochs", 3)):
        raise ValueError("Classification completion requires the epoch budget or a met early-stopping rule")
    checkpoint = exp.load_checkpoint(folder / "best.pt", condition_id)
    if preprocessor_factory is None:
        from .preprocessing import ReleasePreprocessor
        preprocessor_factory = ReleasePreprocessor
    _ensure_cache(exp, manifests)
    prep = preprocessor_factory(exp, condition_id, device)
    prep.prepare([manifests["validation"], manifests["test"]])
    model = load_cnn(exp, condition_id, device, model_factory=model_factory)
    from .evaluation import classification_metrics, classification_bootstrap, youden_threshold
    outputs = {}
    for split in ("validation", "test"):
        probabilities = predict(model, loader(ImageDataset(manifests[split], prep), settings["batch_size"], exp.config["seed"]), device)
        if len(probabilities) != len(manifests[split]):
            raise ValueError(f"Every {split} image must have one prediction")
        outputs[split] = manifests[split].copy()
        outputs[split]["prob_infiltration"] = probabilities
    reference = settings.get("classification_threshold", 0.5)
    policy = settings.get("threshold_policy", "fixed")
    validation, predictions = outputs["validation"], outputs["test"]
    if policy == "validation_youden":
        # Chosen on validation only; test never sets the operating point.
        threshold = youden_threshold(validation.label.to_numpy(), validation.prob_infiltration.to_numpy())
    elif policy == "fixed":
        threshold = reference
    else:
        raise ValueError(f"Unknown threshold policy: {policy}")
    for frame in outputs.values():
        frame["pred_label"] = (frame.prob_infiltration >= threshold).astype(int)
    probabilities = predictions.prob_infiltration.to_numpy()
    result = classification_metrics(predictions.label.to_numpy(), probabilities, threshold)
    history = pd.DataFrame(last["history"])
    result.update(threshold=threshold, threshold_policy=policy, reference_threshold=reference,
                  reference_threshold_metrics=classification_metrics(predictions.label.to_numpy(), probabilities, reference),
                  validation_metrics=classification_metrics(validation.label.to_numpy(), validation.prob_infiltration.to_numpy(), threshold),
                  stopped_early=bool(last["epoch"] < settings["epochs"]))
    result.update(condition_id=condition_id, epochs_run=int(last["epoch"]), best_epoch=int(checkpoint["epoch"]),
                  training_seconds=float(history.seconds.sum()), pos_weight=float(last["pos_weight"]),
                  peak_vram_bytes=int(history.peak_vram_bytes.max()), total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  provenance=exp.provenance(condition_id), checkpoint_hash=file_hash(folder / "best.pt"))
    _write_csv(folder / "test_predictions.csv", predictions)
    _write_csv(folder / "validation_predictions.csv", validation)
    _write_csv(folder / "history.csv", history)
    _write_csv(folder / "classification_bootstrap.csv", classification_bootstrap(predictions, threshold=threshold,
               seed=exp.config["seed"], replicates=exp.config.get("reporting", {}).get("classification_bootstrap_replicates", exp.config["xai"].get("bootstrap_replicates", 2000))))
    write_json(folder / "metrics.json", result)
    from .reporting import plot_confusion, plot_classification_curves, plot_training_history
    plot_confusion(result["confusion_matrix"], folder / "confusion_matrix.png", condition_id)
    plot_classification_curves(predictions, folder, condition_id)
    plot_training_history(history, folder / "training_history.png", condition_id, best_epoch=checkpoint["epoch"])
    write_json(folder / "complete.json", {"provenance": exp.provenance(condition_id), "checkpoint_hash": file_hash(folder / "best.pt"),
               "predictions_hash": file_hash(folder / "test_predictions.csv"), "metrics_hash": file_hash(folder / "metrics.json"),
               "validation_predictions_hash": file_hash(folder / "validation_predictions.csv"),
               "history_hash": file_hash(folder / "history.csv"), "best_epoch": int(checkpoint["epoch"]), "epochs_run": int(last["epoch"])})
    return result


def run_cnn(exp, manifests, device="cuda"):
    train_dae(exp, manifests, device)
    results = [train_condition(exp, manifests, condition_id, device) for condition_id in exp.condition_ids]
    return pd.DataFrame([{key: value for key, value in row.items() if key not in ("provenance", "confusion_matrix")} for row in results])
