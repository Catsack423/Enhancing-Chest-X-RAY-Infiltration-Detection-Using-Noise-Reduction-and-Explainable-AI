"""Unmodified v4.3 helper definitions; full runner replaced by v7."""
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
