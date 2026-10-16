"""Small synthetic GPU validation. Never labels its output as research."""
from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
import torch

from .prepare import MANIFESTS
from .release.experiment import file_hash, write_json


def fixture(directory):
    directory = Path(directory)
    (directory / "images").mkdir(parents=True, exist_ok=True)
    frames = {}
    rng = np.random.default_rng(121)
    for split_index, split in enumerate(("train", "validation", "test")):
        rows = []
        for i in range(4):
            identity = split_index * 10 + i
            name = f"{identity:08d}_000.png"
            image = np.clip(110 + 25*np.sin(np.linspace(0, 10, 256))[None, :] +
                            rng.normal(0, 10, (256, 256)) + (i % 2) * 8, 0, 255).astype(np.uint8)
            path = directory / "images" / name
            cv2.imwrite(str(path), image)
            rows.append({"image": name, "patient_id": identity, "label": i % 2,
                "finding_labels": "Infiltration" if i % 2 else "No Finding", "view_position": "PA",
                "relative_path": "images/" + name, "sha256": file_hash(path), "width": 256, "height": 256})
        frames[split] = pd.DataFrame(rows)
    frames["background"] = frames["train"].copy()
    frames["xai"] = frames["test"].query("label == 1").copy()
    frames["xai"] = frames["xai"].assign(bbox_x=80., bbox_y=80., bbox_w=80., bbox_h=80., risk_flag="pure")
    for name, frame in frames.items():
        path = directory / f"{name}.csv"
        payload = frame.to_csv(index=False).encode()
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Smoke fixture changed")
        path.write_bytes(payload)
    write_json(directory / "dataset.json", {"schema_version": 7, "smoke": True,
        "manifest_hashes": {name: file_hash(directory / f"{name}.csv") for name in MANIFESTS}})
    return directory


def smoke(root, output, device="cuda"):
    from .runner import run, load_classifier
    from .dataset import Preprocessor
    output = Path(output)
    data = fixture(output / "data")
    experiment = run(root, data, output / "run", device=device, smoke=True)
    probes = []
    for condition in experiment.config["conditions"]:
        if condition["id"] not in ("baseline", "dwt_L1_L4_fusion", "dft_low_mid_high_fusion"):
            continue
        row = experiment.data.frames["test"].iloc[0]
        prep = Preprocessor(experiment.data, condition, experiment.config)
        x = prep.tensor(row.image, "poisson_gaussian_N2").unsqueeze(0).to(device).requires_grad_(True)
        model = load_classifier(experiment, condition["id"], device)
        logits = model(x)
        logits.sum().backward()
        if logits.shape != (1, 1) or x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().sum() == 0:
            raise AssertionError("Future XAI input gradient failed")
        p = pd.read_csv(experiment.output / f"CNN/{condition['id']}/test_predictions.csv")
        saved = p[(p.image == row.image) & (p.corruption_id == "poisson_gaussian_N2")].prob_infiltration.iloc[0]
        np.testing.assert_allclose(logits.sigmoid().item(), saved, rtol=1e-5, atol=1e-6)
        probes.append({"condition_id": condition["id"], "input_shape": list(x.shape), "gradient": True, "reload_predictions_match": True})
        del model, x, logits
    before = {p.relative_to(experiment.output).as_posix(): file_hash(p) for p in experiment.output.rglob("*") if p.is_file()}
    run(root, data, output / "run", device=device, smoke=True)
    after = {p.relative_to(experiment.output).as_posix(): file_hash(p) for p in experiment.output.rglob("*") if p.is_file()}
    if before != after:
        raise AssertionError("Resume changed completed artifact bytes")
    write_json(output / "validation.json", {"research_results": False, "device": str(device), "conditions": 12,
        "corruption_states": 16, "dae_epochs": 1, "cnn_epochs": 4, "reload_and_gradient_probes": probes,
        "completed_resume_bytes_unchanged": True})
    print(json.dumps({"gpu_smoke_passed": True, "research_results": False, "validation": str(output / "validation.json")}), flush=True)
