"""Build the one-file Colab upload from selected local NIH images."""
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if (ROOT / ".runtime").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime"))

from shared.experiment import Experiment, digest
from shared.data import prepare_manifests
from shared.data_bundle import build_data_bundle
from build_notebooks import build


def main():
    exp = Experiment.open(ROOT)
    manifests = prepare_manifests(exp)
    smoke = Experiment.open(ROOT, output_root=ROOT / ".runtime/bundle_manifests" / digest(exp.config)[:16], smoke=True)
    smoke_manifests = prepare_manifests(smoke)
    bundle = ROOT / "nih_cxr_subset.zip"
    report = build_data_bundle(exp, manifests, bundle, smoke_manifests)
    build()
    destination = ROOT / "Notebook_2_colab.zip"
    temporary = destination.with_suffix(".tmp.zip")
    excluded = {".runtime", "smoke_runs", "runs", "__pycache__", ".ipynb_checkpoints"}
    sources = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in excluded for part in relative.parts):
            continue
        if relative.parts[:2] == ("shared", "artifacts") or "results" in relative.parts:
            continue
        if path.suffix == ".zip" or path.name == "nih_cxr_subset.json":
            continue
        sources.append(path)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in sources:
            archive.write(path, "Notebook_2/" + path.relative_to(ROOT).as_posix())
        archive.write(bundle, "Notebook_2/nih_cxr_subset.zip", compress_type=zipfile.ZIP_STORED)
    temporary.replace(destination)
    print(f"Ready: {destination} | {report['image_count']:,} unique images | {destination.stat().st_size / 1024**2:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
