"""Twelve validation-selected CNNs; one shared DAE; durable epoch checkpoints."""
from copy import deepcopy
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .dataset import InputData, Preprocessor, ClassificationDataset, ReconstructionDataset, expanded
from .noise import STATES, training_state
from .release.dae_model import DAE
from .release.models import ReleaseResNet50
from .release.experiment import digest, file_hash, write_json, save_torch, set_seed, versions, deterministic_backend
from .release.training import (loader, optimization_epoch, balanced_loss_epoch, predict, cnn_optimizer,
    capture_rng_state, _restore_training, _scaler, _autocast, _finite_loss, _early_stop, _reconstruction_epoch)
from .release.evaluation import youden_threshold


def csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


class Run:
    def __init__(self, root, data_dir, output, smoke=False):
        self.root, self.output = Path(root), Path(output)
        self.config = json.loads((self.root / "configs/protocol.json").read_text())
        # TF32's batch-dependent convolution rounding breaks single-image XAI parity.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.config["numeric_policy"] = {"tf32_matmul": False, "tf32_cudnn": False,
                                          "evaluation": "float32", "training_amp": self.config["cnn"]["amp"]}
        # Effective DAE batch is unchanged; microbatches make it fit an 8GB GPU.
        self.config["dae"]["micro_batch_size"] = 8
        if smoke:
            self.config.update(run_mode="smoke", counts_per_class={s: 2 for s in ("train", "validation", "test")}, bootstrap_replicates=20)
            self.config["preprocessing"].update(working_size=64, cnn_size=64)
            self.config["dae"].update(epochs=1, batch_size=4, micro_batch_size=2, base_channels=4, amp=False)
            self.config["cnn"].update(epochs=4, batch_size=4, accumulation_steps=1, effective_batch_size=4,
                                      pretrained=False, early_stopping_patience=None, amp=False)
            self.config["numeric_policy"]["training_amp"] = False
        self.config["noise_states"] = deepcopy(STATES)
        self.data = InputData(data_dir, self.config)
        core_versions = {k: v for k, v in versions().items() if k != "shap"}
        from importlib.metadata import version
        core_versions["scipy"] = version("scipy")
        source = {p.relative_to(self.root).as_posix(): file_hash(p) for p in sorted((self.root / "v7").rglob("*.py"))}
        source["run.py"] = file_hash(self.root / "run.py")
        self.provenance = {"config": self.config, "config_hash": digest(self.config), "source_hashes": source,
                           "dataset": self.data.receipt, "versions": core_versions}
        self.signature = digest(self.provenance)
        self.output.mkdir(parents=True, exist_ok=True)
        lock = self.output / "experiment.json"
        if lock.exists() and json.loads(lock.read_text())["signature"] != self.signature:
            raise ValueError("Config, code, package versions, manifests or source pixels changed; use a new output directory")
        write_json(lock, {"signature": self.signature, **self.provenance})
        (self.output / "xai_ready/manifests").mkdir(parents=True, exist_ok=True)
        for name in ("xai", "background"):
            csv(self.output / f"xai_ready/manifests/{name}.csv", self.data.frames[name])
        write_json(self.output / "xai_ready/input_geometry.json", self.data.geometry)

    def checkpoint(self, path, condition=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["signature"] != self.signature or state.get("condition_id") != condition:
            raise ValueError("Checkpoint provenance differs")
        return state

    def completed(self, folder, condition=None, marker="training_complete.json"):
        path = folder / marker
        if not path.exists():
            return False
        receipt = json.loads(path.read_text())
        if receipt["signature"] != self.signature or receipt.get("condition_id") != condition:
            raise ValueError("Completion provenance differs")
        for name, expected in receipt["files"].items():
            if file_hash(folder / name) != expected:
                raise ValueError(f"Completed artifact changed: {folder / name}")
        return True

    def mark(self, folder, condition, names, marker="training_complete.json"):
        write_json(folder / marker, {"signature": self.signature, "condition_id": condition,
                                    "files": {name: file_hash(folder / name) for name in names}})


def resume(run, folder, condition=None):
    candidates = [run.checkpoint(folder / name, condition) for name in ("last.pt", "best.pt") if (folder / name).exists()]
    if not candidates:
        return None
    if not (folder / "best.pt").exists():
        raise ValueError("Resume is missing best.pt")
    state = max(candidates, key=lambda value: value["epoch"])
    # Complete a transaction interrupted between best.pt, last.pt and history.csv.
    if not (folder / "last.pt").exists() or run.checkpoint(folder / "last.pt", condition)["epoch"] < state["epoch"]:
        save_torch(folder / "last.pt", state)
    csv(folder / "history.csv", pd.DataFrame(state["history"]))
    return state


def save_epoch(run, folder, condition, model, optimizer, scaler, epoch, best, history, improved, **extra):
    state = {"signature": run.signature, "condition_id": condition, "provenance": run.provenance,
             "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "scaler_state_dict": scaler.state_dict(), "rng_state": capture_rng_state(),
             "epoch": epoch, "best_loss": best, "history": history, **extra}
    if improved:
        save_torch(folder / "best.pt", state)
    save_torch(folder / "last.pt", state)
    csv(folder / "history.csv", pd.DataFrame(history))


def record_training_noise(run, folder, epoch):
    records = []
    for row in run.data.frames["train"].itertuples(index=False):
        state = training_state(run.config["seed"], row.image, epoch)
        _, stats = run.data.noisy(row.image, state, epoch)
        records.append({"image": row.image, "patient_id": row.patient_id, **stats})
    csv(folder / f"corruptions/epoch_{epoch:03d}.csv", pd.DataFrame(records))


def train_dae(run, device):
    settings = run.config["dae"]
    folder = run.output / "DAE"
    if not run.completed(folder):
        set_seed(run.config["seed"], deterministic=False)
        model = DAE(base_channels=settings["base_channels"], residual=True).to(device)
        if settings.get("channels_last"):
            model.to(memory_format=torch.channels_last)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        scaler = _scaler(device, settings["amp"])
        epoch, best, history = 0, float("inf"), []
        checkpoint = resume(run, folder)
        if checkpoint:
            epoch, best, history = _restore_training(checkpoint, model, optimizer, scaler)
        micro = settings["micro_batch_size"]
        accumulation = max(1, settings["batch_size"] // micro)
        if settings["batch_size"] % micro:
            raise ValueError("DAE effective batch must be divisible by microbatch")
        criterion = lambda y, x: nn.functional.mse_loss(y, x) + settings["l1_weight"] * nn.functional.l1_loss(y, x)
        validation = loader(ReconstructionDataset(expanded(run.data.frames["validation"]), run.data), micro, run.config["seed"])
        for epoch in range(epoch + 1, settings["epochs"] + 1):
            started = time.monotonic()
            model.train()
            batches = loader(ReconstructionDataset(run.data.frames["train"], run.data, epoch), micro,
                             run.config["seed"] + epoch, True)
            optimizer.zero_grad(set_to_none=True)
            total, seen, window, updates = 0., 0, 0, 0
            for index, (noisy, clean) in enumerate(batches):
                noisy, clean = noisy.to(device), clean.to(device)
                with _autocast(device, settings["amp"]):
                    loss = criterion(model(noisy), clean)
                _finite_loss(loss)
                scaler.scale(loss * len(noisy)).backward()
                total += float(loss.detach()) * len(noisy)
                seen += len(noisy)
                window += len(noisy)
                if (index+1) % accumulation == 0 or index+1 == len(batches):
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(window)
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updates += int(not scaler.is_enabled() or scaler.get_scale() >= before)
                    optimizer.zero_grad(set_to_none=True)
                    window = 0
            if not updates:
                raise RuntimeError("DAE made no optimizer updates")
            val = _reconstruction_epoch(model, validation, criterion, device)
            history.append({"epoch": epoch, "train_loss": total/seen, "validation_loss": val,
                            "train_images": seen, "optimizer_steps": updates, "seconds": time.monotonic()-started})
            improved = val < best
            best = min(best, val)
            save_epoch(run, folder, None, model, optimizer, scaler, epoch, best, history, improved)
            record_training_noise(run, folder, epoch)
            print(f"DAE {epoch}/{settings['epochs']}: train={total/seen:.5f}, validation={val:.5f}", flush=True)
        run.mark(folder, None, ("best.pt", "last.pt", "history.csv"))
        del optimizer, scaler, model
    deterministic_backend(True)
    state = run.checkpoint(folder / "best.pt")
    model = DAE(base_channels=settings["base_channels"], residual=True).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


def load_classifier(run, condition_id, device="cpu"):
    """Public loader for later Grad-CAM/SHAP; never downloads ImageNet weights."""
    settings = run.config["cnn"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = run.checkpoint(run.output / f"CNN/{condition_id}/best.pt", condition_id)
    model = ReleaseResNet50(pretrained=False, weights_name=settings["weights"],
        dropout=settings["head_dropout"], trainable_layers=settings["trainable_layers"])
    model.load_state_dict(state["model_state_dict"])
    model.set_stage(state["epoch"], settings["warmup_epochs"])
    return model.to(device).eval()


def train_classifier(run, condition, dae, device):
    name, settings = condition["id"], run.config["cnn"]
    folder = run.output / "CNN" / name
    prep = Preprocessor(run.data, condition, run.config, dae, device)
    validation = expanded(run.data.frames["validation"])
    valid_loader = loader(ClassificationDataset(validation, prep), settings["batch_size"], run.config["seed"])
    if not run.completed(folder, name):
        set_seed(run.config["seed"])
        state = resume(run, folder, name)
        model = ReleaseResNet50(pretrained=settings["pretrained"] if state is None else False,
            weights_name=settings["weights"], dropout=settings["head_dropout"],
            trainable_layers=settings["trainable_layers"]).to(device)
        if settings.get("channels_last"):
            model.to(memory_format=torch.channels_last)
        optimizer = cnn_optimizer(model, settings)
        scaler = _scaler(device, settings["amp"])
        epoch, best, history = 0, float("inf"), []
        if state:
            epoch, best, history = _restore_training(state, model, optimizer, scaler, warmup_epochs=settings["warmup_epochs"])
        for epoch in range(epoch + 1, settings["epochs"] + 1):
            if _early_stop(history, settings, settings["warmup_epochs"]):
                break
            deterministic_backend(settings["deterministic"])
            stage = model.set_stage(epoch, settings["warmup_epochs"])
            started = time.monotonic()
            batches = loader(ClassificationDataset(run.data.frames["train"], prep, epoch), settings["batch_size"],
                             run.config["seed"] + epoch, True)
            loss, stats = optimization_epoch(model, batches, optimizer, 1.0, device,
                accumulation_steps=settings["accumulation_steps"], scaler=scaler, amp=settings["amp"], return_stats=True)
            deterministic_backend(True)
            val = balanced_loss_epoch(model, valid_loader, device)
            history.append({"epoch": epoch, "stage": stage, "train_loss": loss, "validation_loss": val,
                            "train_images": len(run.data.frames["train"]), "pos_weight": 1.0,
                            "seconds": time.monotonic()-started, **stats})
            improved = val < best
            best = min(best, val)
            save_epoch(run, folder, name, model, optimizer, scaler, epoch, best, history, improved,
                       stage=stage, next_stage=model.stage_for_epoch(epoch+1, settings["warmup_epochs"]), pos_weight=1.0)
            record_training_noise(run, folder, epoch)
            print(f"{name} {epoch}/{settings['epochs']} {stage}: train={loss:.5f}, validation={val:.5f}", flush=True)
        run.mark(folder, name, ("best.pt", "last.pt", "history.csv"))
        del optimizer, scaler, model
    if not run.completed(folder, name, "evaluation_complete.json"):
        model = load_classifier(run, name, device)
        def measured_predictions(batches):
            values = []
            with torch.no_grad():
                for inputs, _ in batches:
                    values.append(model(inputs.to(device)).flatten().cpu())
            logits = torch.cat(values)
            return logits.sigmoid().numpy().astype(float), logits.numpy().astype(float)
        probabilities, validation_logits = measured_predictions(valid_loader)
        validation["prob_infiltration"] = probabilities
        validation["logit_infiltration"] = validation_logits
        threshold = youden_threshold(validation.label, probabilities)
        csv(folder / "validation_predictions.csv", validation)
        test = expanded(run.data.frames["test"])
        probabilities, logits = measured_predictions(loader(ClassificationDataset(test, prep), settings["batch_size"], run.config["seed"]))
        test["prob_infiltration"], test["logit_infiltration"] = probabilities, logits
        for frame, filename in ((validation, "validation_predictions.csv"), (test, "test_predictions.csv")):
            stats = [run.data.noisy(row.image, row.corruption_id)[1] for row in frame.itertuples(index=False)]
            for field in ("noise_mean", "noise_std", "noise_rmse", "clipping_fraction"):
                frame[field] = [r[field] for r in stats]
            frame["predicted_label"] = (frame.prob_infiltration >= threshold).astype(int)
            csv(folder / filename, frame)
        write_json(folder / "model_spec.json", {"condition": condition, "threshold": threshold,
            "threshold_policy": "one Youden threshold on all 16 validation states; fixed for every test state",
            "view_order": prep.view_order, "input_shape": ["N", len(prep.view_order), 3,
                run.config["preprocessing"]["cnn_size"], run.config["preprocessing"]["cnn_size"]],
            "output_shape": ["N", 1], "target": "infiltration_logit", "target_layer": "backbone.layer4",
            "preprocessing": run.config["preprocessing"], "dae_checkpoint_sha256": file_hash(run.output / "DAE/best.pt"),
            "checkpoint_sha256": file_hash(folder / "best.pt"), "signature": run.signature})
        run.mark(folder, name, ("best.pt", "model_spec.json", "validation_predictions.csv", "test_predictions.csv"), "evaluation_complete.json")
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(root, data_dir, output, *, device="cuda", smoke=False):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime in Colab, or explicitly use --device cpu")
    torch.set_num_threads(4)
    cv = __import__("cv2")
    cv.setNumThreads(1)
    experiment = Run(root, data_dir, output, smoke)
    print(f"v7 {'SMOKE (random weights, not research)' if smoke else 'BETA PILOT seed 42'}: 12 CNN, one DAE; {output}", flush=True)
    dae = train_dae(experiment, device)
    for condition in experiment.config["conditions"]:
        train_classifier(experiment, condition, dae, device)
    from .report import report
    report(experiment, dae, device)
    return experiment
