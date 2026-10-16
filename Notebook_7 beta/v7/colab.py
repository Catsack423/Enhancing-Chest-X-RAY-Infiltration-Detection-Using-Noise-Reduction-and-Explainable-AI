"""Drive durability and VM-local training data; no copy of the full NIH corpus."""
from pathlib import Path
import json
import importlib.util
import shutil
import tempfile
import zipfile

from .release.experiment import file_hash

DRIVE_DATA_ROOT = "/content/drive/MyDrive/nih -xray/data/3"
DRIVE_WORK_ROOT = "/content/drive/MyDrive/nih -xray/Notebook_7_beta"
AUDIT_DIRECTORY = "audit_v3_zip_70012"


def mount_on_fresh_path(mount, content_root="/content"):
    """Avoid populated mountpoints left behind after a failed Colab remount."""
    destination = Path(tempfile.mkdtemp(prefix="nih_drive_v7_", dir=content_root))
    print("Authorize Google Drive using faceyuslay@gmail.com only.", flush=True)
    mount(str(destination))
    return destination


def ensure_source(data_root, remount=None):
    """A mounted MyDrive is insufficient: the NIH dataset must be visible there."""
    root = Path(data_root)
    required = ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt")
    missing = [name for name in required if not (root / name).is_file()]
    if missing and remount is not None:
        print("The NIH source is not visible in this Drive mount. Remount and select faceyuslay@gmail.com (the dataset/G: account).", flush=True)
        remount()
        missing = [name for name in required if not (root / name).is_file()]
    if missing:
        parent = root.parent.parent
        try:
            visible = sorted(p.name for p in parent.iterdir())[:25] if parent.is_dir() else []
        except OSError:
            visible = []
        raise FileNotFoundError(
            f"NIH data is not accessible at {root}; missing {missing}. "
            f"Visible entries under {parent}: {visible}. "
            "Mount faceyuslay@gmail.com (the account on G:) and check for duplicate nih -xray folders, "
            "or set DATA_ROOT to an accessible existing dataset. A newly created Notebook_7_beta folder "
            "does not mean the dataset is present. No audit or cohort selection has started.")
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
        print("Reusing the verified class160 selected-image package on Drive.", flush=True)
        return
    ensure_source(data_root)
    from .audit import audit
    from .prepare import prepare
    # Preserve the old all-error audit; do not reuse its incorrect complete flag.
    audit(data_root, work / AUDIT_DIRECTORY, root / "partition_reference", workers=workers, reuse_complete=True, image_ids=image_ids)
    prepare(root, data_root, work / AUDIT_DIRECTORY, work)


def prepare_from_v3_zip(root, archive_root, data_root, work, workers=4):
    """Reuse the v3 ZIP loader, then audit originals on the Colab VM only."""
    root, archive_root, data_root, work = map(Path, (root, archive_root, data_root, work))
    if selected_bundle(work):
        print("Reusing the verified selected-image package; no ZIP staging or audit needed.", flush=True)
        return
    for parent in archive_root.resolve().parents:
        if parent.name == "MyDrive" and data_root.resolve().is_relative_to(parent.parent):
            raise ValueError("DATA_ROOT must be on VM disk, outside mounted Drive")
    catalog_path = archive_root / "dataset_zip_index.json"
    print(f"Reading v3 ZIP catalog: {catalog_path}", flush=True)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    config = json.loads((root / "configs/protocol.json").read_text())
    files = [name for part in catalog["archives"] for name in part["files"] if name.lower().endswith(".png")]
    ids = [Path(name).name for name in files]
    if (len(ids) != len(set(ids)) or len(ids) != catalog["images"] or
            len(ids) != config["expected_audit_images"] or config.get("audit_scope") != "v3_zip_originals"):
        raise ValueError("v3 ZIP catalog count/identity does not match the configured audit scope")
    import pandas as pd
    required = set()
    for name in ("train", "validation", "test", "xai"):
        required.update(pd.read_csv(root / f"partition_reference/{name}.csv").image)
    if not required <= set(ids):
        raise ValueError("ZIP catalog does not cover every frozen partition/XAI image")
    print(f"ZIP scope: {len(ids):,} originals; all frozen partition/XAI IDs covered; using v3 loader", flush=True)
    legacy = root / "v3_zip_reference/stage_dataset.py"
    specification = importlib.util.spec_from_file_location("v7_v3_zip_stager", legacy)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.stage_dataset(root / "v3_zip_reference", archive_root, data_root)
    from .release.experiment import write_json
    write_json(work / "zip_source.json", {"scope": "v3_zip_originals", "images": len(ids),
        "catalog_sha256": file_hash(catalog_path), "v3_loader_sha256": file_hash(legacy),
        "archives": {part["name"]: part["sha256"] for part in catalog["archives"]}})
    prepare_on_drive(root, data_root, work, workers=workers, image_ids=ids)


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
