"""Drive durability and VM-local training data; no copy of the full NIH corpus."""
from pathlib import Path
import json
import shutil
import zipfile

from .release.experiment import file_hash

DRIVE_DATA_ROOT = "/content/drive/MyDrive/nih -xray/data/3"
DRIVE_WORK_ROOT = "/content/drive/MyDrive/nih -xray/Notebook_7_beta"


def safe_extract(package, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        for entry in archive.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError("Archive path escapes its staging directory")
        archive.extractall(destination)


def selected_bundle(work):
    work = Path(work)
    receipt = work / "selected_package.json"
    package = work / "v7_selected_data.zip"
    if not receipt.exists() or not package.exists():
        return None
    saved = json.loads(receipt.read_text())
    if file_hash(package) != saved["sha256"]:
        raise ValueError("Selected package changed; do not mix it with the frozen run")
    return saved


def prepare_on_drive(root, data_root, work, workers=4):
    """The same notebook can restart without re-auditing or reselecting data."""
    root, data_root, work = Path(root), Path(data_root), Path(work)
    if selected_bundle(work):
        print("Reusing the verified class160 selected-image package on Drive.", flush=True)
        return
    from .audit import audit
    from .prepare import prepare
    audit(data_root, work / "audit", root / "partition_reference", workers=workers, reuse_complete=True)
    prepare(root, data_root, work / "audit", work)


def stage_selected(work, cache):
    work, cache = Path(work), Path(cache)
    receipt = selected_bundle(work)
    if receipt is None:
        raise FileNotFoundError("Run the audit/selection stage first")
    stage = cache / receipt["sha256"][:16]
    target = stage / "v7_data"
    if (target / "dataset.json").exists():
        if file_hash(target / "dataset.json") == receipt["dataset_receipt_sha256"]:
            print("Reusing the VM-local selected images; training will verify all pixel hashes.", flush=True)
            return target
        raise ValueError("Staged dataset receipt changed")
    stage.mkdir(parents=True, exist_ok=True)
    local_zip = stage / "selected.zip"
    shutil.copyfile(work / "v7_selected_data.zip", local_zip)
    if file_hash(local_zip) != receipt["sha256"]:
        raise ValueError("Drive-to-VM package copy failed integrity verification")
    safe_extract(local_zip, stage)
    if file_hash(target / "dataset.json") != receipt["dataset_receipt_sha256"]:
        raise ValueError("Extracted dataset receipt differs")
    return target
