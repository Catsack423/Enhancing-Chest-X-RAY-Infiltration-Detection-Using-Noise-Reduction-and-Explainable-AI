"""Read-only opening of a saved CNN experiment for later Grad-CAM/SHAP scripts.

Adding a new explainer script does not invalidate the recorded training code.
Changing any recorded training/preprocessing source file still does.
"""
from pathlib import Path
from importlib.metadata import version
import json
import torch

from v7.dataset import InputData
from v7.runner import Run, load_classifier
from v7.release.experiment import digest, file_hash, versions


def open_experiment(root, data_dir, output):
    root, output = Path(root), Path(output)
    saved = json.loads((output / "experiment.json").read_text())
    provenance = {key: value for key, value in saved.items() if key != "signature"}
    if digest(provenance) != saved["signature"]:
        raise ValueError("Saved experiment record integrity failed")
    for name, expected in saved["source_hashes"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Recorded training/preprocessing code changed: {name}")
    current = {key: value for key, value in versions().items() if key != "shap"}
    current["scipy"] = version("scipy")
    if current != saved["versions"]:
        raise ValueError("Restore the recorded core package versions before using the saved CNN")
    data = InputData(data_dir, saved["config"])
    if data.receipt != saved["dataset"]:
        raise ValueError("Saved CNN requires its original selected dataset")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = Run.__new__(Run)
    experiment.root, experiment.output = root, output
    experiment.config, experiment.data = saved["config"], data
    experiment.signature, experiment.provenance = saved["signature"], provenance
    return experiment
