"""Exercise all release stages on isolated tiny frozen IDs, never claim study scores."""
from pathlib import Path
import argparse
import json
from time import perf_counter

ROOT = Path(__file__).resolve().parent


def validate(data_root=None, resume_only=False):
    import torch
    from pipeline import open_run, run_stages
    from shared.experiment import file_hash, write_json
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA; run unit tests separately on CPU")
    validation_root = ROOT / ".runtime/gpu_validation"
    exp, manifests = open_run(data_root=data_root, output_root=validation_root / "runs",
                              cache_root=validation_root / "cache", seed=42, smoke=True)
    if resume_only and not (exp.output_root / "comparison/complete.json").exists():
        raise ValueError("No completed smoke run to resume")
    started = perf_counter()
    if not resume_only:
        run_stages(exp, manifests, device="cuda")
    markers = {}
    for condition in exp.condition_ids:
        folder = exp.results("CNN", condition)
        for name in ("best.pt", "last.pt", "history.csv", "test_predictions.csv", "complete.json"):
            path = folder / name
            markers[str(path.relative_to(exp.output_root))] = file_hash(path)
        history = __import__("pandas").read_csv(folder / "history.csv")
        assert history.epoch.tolist() == [1, 2, 3, 4]
        assert history.stage.tolist() == ["head_only"] * 3 + ["layer4_head"]
        assert set(history.train_images) == {4}
        for criterion in ("Grad-CAM", "SHAP"):
            frame = __import__("pandas").read_csv(exp.results(criterion, condition) / "per_image_metrics.csv")
            assert len(frame) == 2 and set(frame.image) == set(manifests["xai"].image)
    # Independent reopen simulates resuming a notebook; a completed run must not train again.
    resumed, same = open_run(data_root=data_root, output_root=validation_root / "runs",
                             cache_root=validation_root / "cache", seed=42, smoke=True)
    run_stages(resumed, same, device="cuda")
    for relative, expected in markers.items():
        assert file_hash(resumed.output_root / relative) == expected, f"Resume rewrote {relative}"
    report = {"status": "PASS", "scope": "tiny GPU integration; random smoke initialization, not research accuracy",
              "gpu": torch.cuda.get_device_name(0), "conditions": exp.condition_ids,
              "counts": {k: len(v) for k, v in manifests.items()}, "cnn_epochs": 4,
              "warmup_transition_verified": True, "classification_gradcam_shap_reports_verified": True,
              "reopen_resume_without_retraining_verified": True, "seconds": perf_counter() - started}
    write_json(validation_root / "validation.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume-only", action="store_true")
    args = parser.parse_args()
    validate(args.data_root, args.resume_only)
