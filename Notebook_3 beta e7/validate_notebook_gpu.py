"""Execute delivered cells with isolated tiny manifests, seven epochs and CUDA."""
import contextlib
import io
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT.parent / "Notebook_2 release/.runtime/torch"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT))

import nbformat
import torch

SMOKE = '''from copy import deepcopy
from shared.experiment import Experiment, write_json
full_exp, full_manifests = exp, manifests
config = deepcopy(exp.config)
config["run_mode"] = "smoke"
config["data"].update(train_per_class=2, validation_per_class=2, test_per_class=2, expected_test_images=4)
config["training"]["images_per_epoch"] = 4
config["cnn"].update(batch_size=2)  # still exactly 7 epochs
config["xai"].update(background_per_class=1, shap_nsamples=4, bootstrap_replicates=20, visual_examples=1)
config["validation_note"] = "GPU SMOKE ONLY: not research results; different sample sizes"
def first_per_class(frame, n):
    return frame.sort_values("image").groupby("label", group_keys=False).head(n).sort_values("image").reset_index(drop=True)
manifests = {"train": first_per_class(full_manifests["background"], 2),
             "validation": first_per_class(full_manifests["validation"], 2),
             "test": first_per_class(full_manifests["test"], 2),
             "xai": full_manifests["xai"].head(2).copy(),
             "background": first_per_class(full_manifests["background"], 1)}
exp = Experiment(full_exp.root, full_exp.data_root, VALIDATION_DIR / "smoke_run", config)
folder = exp.artifacts / "manifests"
folder.mkdir(parents=True, exist_ok=True)
for name, frame in manifests.items():
    assert set(frame.image) <= set(full_manifests[name].image)
    path = folder / (name + ".csv")
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), frame.reset_index(drop=True))
    else:
        frame.to_csv(path, index=False)
exp.lock()
print("GPU SMOKE ONLY: 4 train/4 val/4 test, 2 XAI/2 background, 7 epochs, SHAP nsamples=4")
'''


class Tee(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

    def write(self, value):
        self.stream.write(value)
        self.stream.flush()
        return super().write(value)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    if not torch.cuda.is_available():
        raise RuntimeError("Use the project .gpu/Scripts/python.exe CUDA environment")
    torch.set_num_threads(4)
    destination = ROOT / ".runtime/gpu_smoke"
    destination.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    notebook = nbformat.read(ROOT / "frequency_head_only_e7.ipynb", as_version=4)
    notebook.cells[0].source = "# GPU SMOKE ONLY — not research results\n\n" + notebook.cells[0].source
    for cell in notebook.cells:
        if cell.cell_type == "code" and "OUTPUT_ROOT = None" in cell.source:
            cell.source = cell.source.replace("OUTPUT_ROOT = None", "OUTPUT_ROOT = VALIDATION_DIR / 'full_preflight'")
    setup = next(i for i, c in enumerate(notebook.cells) if c.cell_type == "code")
    notebook.cells.insert(setup + 1, nbformat.v4.new_code_cell(SMOKE))
    namespace = {"__name__": "__main__", "VALIDATION_DIR": destination}
    started, executed = time.perf_counter(), 0
    try:
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if "export_path =" in cell.source:
                cell.source = "# Skip export during validation; no checkpoint duplication."
                continue
            executed += 1
            print(f"CELL {i} ({executed}) on {torch.cuda.get_device_name(0)}", flush=True)
            output = Tee(sys.stdout)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(cell.source, f"v3_beta:cell_{i}", "exec"), namespace)
            cell.execution_count = executed
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())]
            nbformat.write(notebook, destination / "executed_smoke.ipynb")
        assert (namespace["status"][["CNN", "Grad-CAM", "SHAP"]] == "complete").all().all()
        assert (namespace["cnn_table"].epochs_run == 7).all()
        assert (namespace["cnn_table"].trainable_parameters == 2049).all()
        # Use the previous real e2 smoke outputs only for comparison validation;
        # never fall back to these paths in the delivered notebook.
        from shared.epoch_comparison import compare_with_e2
        old_smoke = ROOT.parent / "Notebook_3 beta/.runtime/gpu_smoke/smoke_run"
        comparison_checked = False
        if old_smoke.is_dir():
            comparison = compare_with_e2(namespace["exp"], old_smoke)
            assert comparison["ready"] and len(comparison["classification"]) == 6
            assert len(comparison["differences"]) == 3 and not comparison["missing_xai"]
            comparison_checked = True
        from shared.experiment import write_json
        report = {"gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                  "seconds": time.perf_counter() - started, "executed_code_cells": executed,
                  "status": namespace["status"].to_dict("records"), "epochs": 7,
                  "research_results": False, "full_training_run": False,
                  "e2_vs_e7_comparison_checked": comparison_checked}
        write_json(destination / "smoke_validation.json", report)
        print("GPU notebook smoke passed:", report, flush=True)
    except Exception:
        (destination / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        nbformat.write(notebook, destination / "executed_smoke.ipynb")
        raise


if __name__ == "__main__":
    main()
