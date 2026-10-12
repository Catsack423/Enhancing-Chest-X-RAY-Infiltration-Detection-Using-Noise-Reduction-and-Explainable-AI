"""Verify the actual Colab upload, bootstrap, selected IDs, and native XAI files."""
from pathlib import Path
import json
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if (ROOT / ".runtime").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime"))

import numpy as np
import pandas as pd

from build_notebooks import SETUP
from shared.data_bundle import colab_data_from_package
from shared.data import prepare_manifests
from shared.denoising import read_gray, working_image, cnn_tensor
from shared.experiment import Experiment, file_hash, write_json, digest


def main():
    config = json.loads((ROOT / "shared/config.json").read_text())
    work = ROOT / ".runtime/bundle_validation" / digest(config)[:16]
    drive = work / "drive"
    my_drive = drive / "MyDrive"
    my_drive.mkdir(parents=True, exist_ok=True)
    # Test exactly the notebook bootstrap with a fake Drive mount on this machine.
    prefix = SETUP.split("\nif IN_COLAB:\n    import subprocess", 1)[0]
    prefix = prefix.replace("/content/drive", drive.as_posix())
    prefix = prefix.replace('Path("' + my_drive.as_posix() + '/Notebook_2_colab.zip")',
                            'Path(' + repr((ROOT / "Notebook_2_colab.zip").as_posix()) + ')')
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda *args: None)
    google.colab = colab
    namespace = {}
    with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
        exec(compile(prefix, "colab_bootstrap", "exec"), namespace)
    copied_root = namespace["NOTEBOOK_2_DIR"]
    assert (copied_root / "shared/data_bundle.py").is_file()
    assert not (copied_root / "nih_cxr_subset.zip").exists()
    data = colab_data_from_package(ROOT / "Notebook_2_colab.zip", copied_root, work / "cache")
    original = Experiment.open(ROOT)
    source = prepare_manifests(original)
    extracted = Experiment.open(copied_root, data_root=data, output_root=work / "full_results")
    portable = prepare_manifests(extracted)
    for name in source:
        pd.testing.assert_frame_equal(source[name], portable[name])
    smoke_source = Experiment.open(ROOT, output_root=ROOT / ".runtime/bundle_manifests" / digest(original.config)[:16], smoke=True)
    smoke_target = Experiment.open(copied_root, data_root=data, output_root=work / "smoke_results", smoke=True)
    smoke_frames = prepare_manifests(smoke_source)
    smoke_portable = prepare_manifests(smoke_target)
    for name in smoke_frames:
        pd.testing.assert_frame_equal(smoke_frames[name], smoke_portable[name])
    for name in portable["xai"].relative_path:
        assert file_hash(data / name) == file_hash(original.data_root / name)
        assert read_gray(data / name).shape == (1024, 1024)
    for name in portable["train"].relative_path.head(10):
        a = working_image(read_gray(original.data_root / name), original.config)
        b = working_image(read_gray(data / name), extracted.config)
        np.testing.assert_array_equal(cnn_tensor(a, original.config), cnn_tensor(b, extracted.config))
    assert colab_data_from_package(ROOT / "Notebook_2_colab.zip", copied_root, work / "cache") == data
    report = {"status": "passed", "bootstrap": "simulated Colab mount; real ZIP extraction",
              "full_manifest_ids_equal": True, "smoke_manifest_ids_equal": True,
              "native_xai_byte_identical": 123,
              "all_bundle_file_hashes_verified": True,
              "counts": {name: len(frame) for name, frame in portable.items()},
              "colab_gpu_execution": "not_run"}
    write_json(work / "verification.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
