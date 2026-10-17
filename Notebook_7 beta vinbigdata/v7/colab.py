"""Colab Drive durability and kagglehub staging for awsaf49/vinbigdata-512-image-dataset."""
from pathlib import Path
import json
import shutil
import tempfile
import zipfile

from .audit import resolve_csv_path, image_index
from .release.experiment import file_hash

DEFAULT_DATASET_SLUG = "awsaf49/vinbigdata-512-image-dataset"
DRIVE_WORK_ROOT = "/content/drive/MyDrive/nih -xray/Notebook_7_beta_vinbigdata"
AUDIT_DIRECTORY = "audit_vinbigdata_512"


def mount_on_fresh_path(mount, content_root="/content"):
    """Avoid populated mountpoints left behind after a failed Colab remount."""
    destination = Path(tempfile.mkdtemp(prefix="vin_drive_v7_", dir=content_root))
    print("Mounting Google Drive for durable checkpoints and outputs...", flush=True)
    mount(str(destination))
    return destination


def download_vinbigdata(slug=DEFAULT_DATASET_SLUG, work=None):
    """Download awsaf49/vinbigdata-512-image-dataset via kagglehub unless selected package already exists."""
    if work is not None and selected_bundle(work):
        print("Found verified v7_selected_data.zip on Drive; skipping kagglehub download.", flush=True)
        return Path("/content/vinbigdata_skipped_already_prepared")
    import kagglehub
    print(f"Downloading dataset via kagglehub: {slug} ...", flush=True)
    path = Path(kagglehub.dataset_download(slug))
    print(f"Dataset ready at: {path}", flush=True)
    return ensure_source(path)


def ensure_source(data_root):
    """Verify that VinBigData train.csv and PNG images exist at data_root."""
    root = Path(data_root)
    csv_path = resolve_csv_path(root)
    images = image_index(root)
    if not images:
        raise FileNotFoundError(f"Found {csv_path}, but no training PNG/JPG images under {root}")
    print(f"Verified VinBigData source: CSV={csv_path}, indexed images={len(images) // 2:,}", flush=True)
    return root


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


def prepare_on_drive(root, data_root, work, workers=4, *, image_ids=None):
    """The same notebook can restart without re-auditing or reselecting data."""
    root, data_root, work = Path(root), Path(data_root), Path(work)
    if selected_bundle(work):
        print("Reusing the verified class160 VinBigData selected-image package on Drive.", flush=True)
        return
    ensure_source(data_root)
    config = json.loads((root / "configs/protocol.json").read_text())
    from .audit import audit
    from .prepare import prepare
    audit(data_root, work / AUDIT_DIRECTORY, workers=workers, reuse_complete=True, image_ids=image_ids, seed=config.get("seed", 42))
    prepare(root, data_root, work / AUDIT_DIRECTORY, work)


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
