"""Export one measured seed; original dataset and runtime cache are excluded."""
from pathlib import Path
import argparse
import zipfile


def export_results(run_root, destination=None, include_checkpoints=False):
    run_root = Path(run_root).resolve()
    if not (run_root / "shared/artifacts/experiment.json").is_file():
        raise FileNotFoundError("Select an existing measured seed directory")
    target = Path(destination or run_root.parent / f"{run_root.name}_results.zip").resolve()
    if target.is_relative_to(run_root):
        raise ValueError("Export destination must be outside the run directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(run_root.rglob("*")):
            if not p.is_file() or p.suffix in (".tmp", ".zip"):
                continue
            if not include_checkpoints and p.suffix in (".pt", ".pth", ".ckpt"):
                continue
            archive.write(p, run_root.name + "/" + p.relative_to(run_root).as_posix())
    temp.replace(target)
    print("Exported", target, "| checkpoints:", include_checkpoints)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    export_results(args.run_root, args.destination, args.include_checkpoints)
