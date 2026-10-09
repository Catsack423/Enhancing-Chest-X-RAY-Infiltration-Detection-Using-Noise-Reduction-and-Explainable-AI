"""Execute the delivered notebook cells on RTX/CUDA with isolated smoke overrides.

Run only after telling the user duration/scope. No main-run training is started.
Outputs under .runtime/frequency_gpu_validation; original notebook stays clean.
"""
from pathlib import Path
import contextlib
import io
import os
import time
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".runtime/torch"))
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
config["cnn"].update(epochs=1, batch_size=2)
config["xai"].update(background_per_class=1, shap_nsamples=4, bootstrap_replicates=20, visual_examples=1)
config["validation_note"] = "GPU smoke only; never report these as research results"
def first_per_class(frame, n):
    return frame.sort_values("image").groupby("label", group_keys=False).head(n).sort_values("image").reset_index(drop=True)
manifests = {
    "train": first_per_class(full_manifests["background"], 2),
    "validation": first_per_class(full_manifests["validation"], 2),
    "test": first_per_class(full_manifests["test"], 2),
    "xai": full_manifests["xai"].head(2).copy(),
    "background": first_per_class(full_manifests["background"], 1),
}
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
print("GPU SMOKE: 4 train, 4 validation, 4 test, 2 XAI, 2 background; 1 epoch; SHAP nsamples=4")
'''


class Tee(io.StringIO):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
    def write(self, value):
        self.stream.write(value)
        self.stream.flush()
        return super().write(value)


def benchmark_settings(namespace, destination):
    """Two training steps and one full-budget SHAP image/method, no saved weights."""
    import gc
    import shap
    from shared.frequency import (FeatureFusionResNet50, FrequencyPreprocessor,
                                  frequency_gradcam, frequency_shap_maps)
    from shared.models import load_cnn
    from shared.experiment import write_json
    full, frames, smoke = namespace["full_exp"], namespace["full_manifests"], namespace["exp"]
    measurements = []
    for condition in full.condition_ids:
        print("BENCHMARK original batch/settings:", condition, flush=True)
        prep = FrequencyPreprocessor(full, condition)
        model = load_cnn(smoke, condition, "cuda", model_factory=FeatureFusionResNet50)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                     lr=full.config["cnn"]["learning_rate"])
        rows = frames["train"].head(full.config["cnn"]["batch_size"]).to_dict("records")
        started = time.perf_counter()
        x = torch.stack([prep.tensor(row) for row in rows]).cuda()
        y = torch.tensor([[float(row["label"])] for row in rows], device="cuda")
        torch.cuda.synchronize()
        preprocessing = time.perf_counter() - started
        step_times = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
        training_peak = torch.cuda.max_memory_allocated() / 1024**3
        del optimizer, x, y, loss
        model.zero_grad(set_to_none=True)
        model.eval()
        gc.collect()
        torch.cuda.empty_cache()
        row = frames["xai"].iloc[0].to_dict()
        x = prep.tensor(row).unsqueeze(0).cuda()
        torch.cuda.synchronize()
        started = time.perf_counter()
        frequency_gradcam(model, x, (1024, 1024))
        torch.cuda.synchronize()
        cam_seconds = time.perf_counter() - started
        background = torch.stack([prep.tensor(r) for r in frames["background"].to_dict("records")]).cuda()
        explainer = shap.GradientExplainer(model, background,
                                         batch_size=full.config["xai"]["shap_batch_size"], local_smoothing=0)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        frequency_shap_maps(explainer, x, (1024, 1024), full.config)
        torch.cuda.synchronize()
        shap_seconds = time.perf_counter() - started
        record = {"condition_id": condition, "source_images_per_batch": len(rows),
                  "preprocessing_batch_seconds": preprocessing, "train_step_seconds": step_times,
                  "train_peak_allocated_GiB": training_peak,
                  "gradcam_one_image_seconds": cam_seconds,
                  "shap_one_image_seconds": shap_seconds,
                  "shap_nsamples": full.config["xai"]["shap_nsamples"], "background_images": len(background),
                  "shap_peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3}
        measurements.append(record)
        print(record, flush=True)
        write_json(destination / "benchmark.json", measurements)
        del model, prep, x, background, explainer
        gc.collect()
        torch.cuda.empty_cache()
    return measurements


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the project .gpu/Scripts/python.exe")
    torch.set_num_threads(4)
    destination = ROOT / ".runtime/frequency_gpu_validation"
    destination.mkdir(parents=True, exist_ok=True)
    # Download is kept in the project cache; hash verification is torchvision's default.
    from torchvision.models import ResNet50_Weights
    ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True)
    os.chdir(ROOT)
    notebook = nbformat.read(ROOT / "Frequency/frequency_comparison.ipynb", as_version=4)
    notebook.cells[0].source = "# GPU SMOKE ONLY — measured pipeline checks, not research results\n\n" + notebook.cells[0].source
    # Keep the actual setup cell, but route even its preflight artifacts to scratch.
    for cell in notebook.cells:
        if cell.cell_type == "code" and "OUTPUT_ROOT = None" in cell.source:
            cell.source = cell.source.replace("OUTPUT_ROOT = None", "OUTPUT_ROOT = VALIDATION_DIR / 'full_preflight'")
    setup_index = next(i for i, cell in enumerate(notebook.cells) if cell.cell_type == "code")
    notebook.cells.insert(setup_index + 1, nbformat.v4.new_code_cell(SMOKE))
    namespace = {"__name__": "__main__", "VALIDATION_DIR": destination}
    started = time.perf_counter()
    executed = 0
    try:
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            if "export_path =" in cell.source:
                cell.source = "# Export skipped during GPU smoke to avoid unnecessary checkpoint duplication."
                continue
            executed += 1
            print(f"\nCELL {i} ({executed}) on {torch.cuda.get_device_name(0)}", flush=True)
            output = Tee(sys.stdout)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(cell.source, f"frequency_comparison.ipynb:cell_{i}", "exec"), namespace)
            cell.execution_count = executed
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())]
            nbformat.write(notebook, destination / "executed_smoke.ipynb")
        status = namespace["status"]
        assert (status[["CNN", "Grad-CAM", "SHAP"]] == "complete").all().all()
        from shared.experiment import write_json
        write_json(destination / "smoke_validation.json", {
            "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
            "seconds": time.perf_counter() - started, "executed_code_cells": executed,
            "status": status.to_dict("records"), "research_results": False,
        })
        print("GPU notebook smoke passed in", time.perf_counter() - started, "seconds", flush=True)
        benchmark_settings(namespace, destination)
    except Exception:
        (destination / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        nbformat.write(notebook, destination / "executed_smoke.ipynb")
        raise


if __name__ == "__main__":
    main()
