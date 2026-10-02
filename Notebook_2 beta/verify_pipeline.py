"""Execute a real-data smoke experiment, separate from full research results."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if (ROOT / ".runtime").exists():
    sys.path.insert(0, str(ROOT / ".runtime"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".runtime/torch"))

import torch
from shared.experiment import Experiment, write_json
from shared.data import prepare_manifests
from shared.training import run_cnn
from shared.xai import run_xai
from shared.reporting import refresh_reports


def main():
    torch.set_num_threads(2)
    exp = Experiment.open(ROOT, smoke=True)
    manifests = prepare_manifests(exp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_cnn(exp, manifests, device)
    for criterion in ("Grad-CAM", "SHAP"):
        records = run_xai(exp, manifests, criterion, device)
        if len(records) != 10 * len(manifests["xai"]):
            raise AssertionError("Missing smoke XAI records")
    status = refresh_reports(exp)
    if not (status[["CNN", "Grad-CAM", "SHAP"]] == "complete").all().all():
        raise AssertionError("Smoke pipeline is incomplete")
    write_json(exp.output_root / "verification.json", {"mode": "smoke", "device": device, "conditions": 10,
               "xai_records_per_explainer": 20, "full_experiment_run": False,
               "checks": ["CNN checkpoint round-trip", "classification predictions", "native-resolution Grad-CAM", "signed and positive SHAP", "patient split exclusion"]})
    print("SMOKE PASS: 10 CNN checkpoints, 20 Grad-CAM records, 20 SHAP records. Full Colab research run remains pending.")


if __name__ == "__main__":
    main()
