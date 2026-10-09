"""Reproduce the 200/class manifests and portable image bundle; never train."""
import json
from pathlib import Path

from shared.class_dataset import prepare_and_pack
from shared.experiment import Experiment

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    result = prepare_and_pack(ROOT, Experiment.open(ROOT).data_root)
    print(json.dumps(result, indent=2))
