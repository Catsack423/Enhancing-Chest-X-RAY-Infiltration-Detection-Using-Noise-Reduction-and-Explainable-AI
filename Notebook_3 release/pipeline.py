"""Explicit release stages; importing this module never trains or reads Drive."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parent
STAGES = ("dae", "cnn", "gradcam", "shap", "reports")


def open_run(data_root=None, output_root=None, cache_root=None, seed=42, smoke=False,
             profile="release_controlled", model_protocol=None, prepare_cache=True):
    from shared.experiment import Experiment
    from shared.data import prepare_manifests
    exp = Experiment.open(ROOT, data_root, output_root, smoke, seed, cache_root, profile, model_protocol)
    frames = prepare_manifests(exp)
    if prepare_cache:
        from shared.runtime_cache import RuntimeCache
        exp.cache = RuntimeCache(exp).prepare(frames)
        exp.lock()
    return exp, frames


def run_stages(exp, manifests, stages=STAGES, condition_ids=None, device=None):
    import torch
    stages = tuple(stages)
    if set(stages) - set(STAGES):
        raise ValueError("Unknown pipeline stage")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if exp.config["run_mode"] == "full" and device == "cpu" and set(stages) & {"dae", "cnn", "gradcam", "shap"}:
        raise RuntimeError("Select a GPU runtime in Colab before running the full experiment")
    selected = list(condition_ids) if condition_ids is not None else exp.condition_ids
    if not selected or set(selected) - set(exp.condition_ids):
        raise ValueError("Select valid release conditions")
    if (exp.cache is None or not getattr(exp.cache, "ready", False)) and set(stages) & {"dae", "cnn", "gradcam", "shap"}:
        raise RuntimeError("Prepare the temporary runtime cache first")
    from shared.training import train_dae, train_condition, evaluate_classifier
    from shared.xai import evaluate_condition
    from shared.reporting import report_all
    needs_dae = any(exp.condition(c)["group"] == "dae_clahe" for c in selected)
    if "dae" in stages and needs_dae:
        train_dae(exp, manifests, device=device)
    if "cnn" in stages:
        if needs_dae:
            train_dae(exp, manifests, device=device)
        for condition in selected:
            print(f"Seed {exp.config['seed']} | CNN | {condition}", flush=True)
            train_condition(exp, manifests, condition, device=device)
            evaluate_classifier(exp, manifests, condition, device=device)
            from check_overfit import plot_overfit
            plot_overfit(exp.output_root, display_plots=False)
    for stage, criterion in (("gradcam", "Grad-CAM"), ("shap", "SHAP")):
        if stage in stages:
            for condition in selected:
                print(f"Seed {exp.config['seed']} | {criterion} | {condition}", flush=True)
                evaluate_condition(exp, manifests, criterion, condition, device=device)
    if "reports" in stages:
        report_all(exp)
    return exp.output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--stage", choices=("prepare", "all", *STAGES), default="prepare")
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--profile", choices=("release_controlled", "legacy_v2"), default="release_controlled")
    parser.add_argument("--model-protocol", choices=("layer4_after_warmup", "head_only_legacy"))
    args = parser.parse_args()
    settings = json.loads((ROOT / "configs/release_full.json").read_text())
    seeds = settings["training_seeds"] if args.all_seeds else [args.seed]
    exp = None
    for seed in seeds:
        exp, manifests = open_run(args.data_root, args.output_root, args.cache_root, seed,
                                  args.smoke, args.profile, args.model_protocol,
                                  prepare_cache=args.stage != "reports")
        print("Verified counts:", {k: len(v) for k, v in manifests.items()}, flush=True)
        if args.stage != "prepare":
            run_stages(exp, manifests, STAGES if args.stage == "all" else [args.stage], args.conditions, args.device)
    if args.all_seeds and args.stage in ("all", "reports"):
        from shared.reporting import summarize_seeds
        summarize_seeds(exp.base_output_root)


if __name__ == "__main__":
    main()
