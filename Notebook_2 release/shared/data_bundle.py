"""Portable selected-image data with identical working pixels and native XAI."""
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
import shutil
import zipfile

import cv2
import numpy as np

from .denoising import read_gray, working_image
from .experiment import digest, file_hash, write_json

SOURCE_FILES = ("Data_Entry_2017.csv", "BBox_List_2017.csv", "train_val_list.txt", "test_list.txt")


def build_data_bundle(exp, manifests, destination, smoke_manifests=None):
    """Select by manifests, never by scanning/uploading every NIH image."""
    frames = list(manifests.values()) + list((smoke_manifests or {}).values())
    selected = {r.relative_path for frame in frames for r in frame.itertuples()}
    native = set(manifests["xai"].relative_path)
    selected.update(native)
    config = json.loads((exp.root / "shared/config.json").read_text(encoding="utf-8"))
    report = {"schema_version": 1, "config_hash": digest(config),
              "image_count": len(selected), "native_xai_images": len(native),
              "working_size": config["preprocessing"]["working_size"],
              "counts": {name: len(frame) for name, frame in manifests.items()},
              "image_policy": "native XAI; lossless PNG of INTER_AREA working pixels for classification",
              "files": {}}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name in SOURCE_FILES:
            content = (exp.data_root / name).read_bytes()
            archive.writestr(name, content, compress_type=zipfile.ZIP_DEFLATED)
            report["files"][name] = {"bytes": len(content), "sha256": sha256(content).hexdigest()}
        for i, name in enumerate(sorted(selected), 1):
            source = exp.data_root / name
            raw = read_gray(source)
            working = working_image(raw, config)
            if name in native:
                content = source.read_bytes()
            else:
                ok, encoded = cv2.imencode(".png", working)
                if not ok:
                    raise RuntimeError(f"PNG encoding failed: {name}")
                content = encoded.tobytes()
            restored = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if restored is None or not np.array_equal(working_image(restored, config), working):
                raise ValueError(f"Working pixels changed: {name}")
            if name in native and not np.array_equal(restored, raw):
                raise ValueError(f"Native XAI pixels changed: {name}")
            archive.writestr(name, content)
            report["files"][name] = {"bytes": len(content), "sha256": sha256(content).hexdigest(),
                                     "height": restored.shape[0], "width": restored.shape[1]}
            if i % 1000 == 0 or i == len(selected):
                print(f"Packed and verified {i:,}/{len(selected):,} images", flush=True)
        archive.writestr("bundle.json", json.dumps(report, sort_keys=True).encode(),
                         compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(destination)
    write_json(destination.with_suffix(".json"), report)
    return report


def _checked_entries(archive, report):
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(report["files"]) | {"bundle.json"}:
        raise ValueError("Unexpected or duplicate data bundle members")
    for name, info in report["files"].items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe data bundle path")
        if archive.getinfo(name).file_size != info["bytes"]:
            raise ValueError(f"Data bundle size changed: {name}")
    if not set(SOURCE_FILES) <= set(report["files"]):
        raise ValueError("Data bundle metadata is missing")


def extract_data_bundle(archive_path, config, cache_root):
    """Hash-check each extracted file and reuse a fully verified runtime cache."""
    with zipfile.ZipFile(archive_path) as archive:
        report = json.loads(archive.read("bundle.json"))
        if report["config_hash"] != digest(config):
            raise ValueError("Data bundle config changed; rebuild Notebook_2_colab.zip on the local machine")
        _checked_entries(archive, report)
        target = Path(cache_root).resolve() / digest(report)[:20]
        marker = target / ".verified.json"
        if marker.is_file() and json.loads(marker.read_text()) == {"bundle_hash": digest(report)}:
            if all((target / name).is_file() and (target / name).stat().st_size == info["bytes"]
                   and file_hash(target / name) == info["sha256"]
                   for name, info in report["files"].items()):
                return target
        target.mkdir(parents=True, exist_ok=True)
        for i, (name, info) in enumerate(report["files"].items(), 1):
            path = (target / name).resolve()
            if not path.is_relative_to(target):
                raise ValueError("Extraction target leaves the runtime cache")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            h = sha256()
            with archive.open(name) as source, temporary.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    h.update(chunk)
                    output.write(chunk)
            if h.hexdigest() != info["sha256"]:
                raise ValueError(f"Corrupt bundled data: {name}")
            temporary.replace(path)
            if i % 2000 == 0:
                print(f"Extracted and verified {i:,} files", flush=True)
        write_json(marker, {"bundle_hash": digest(report)})
    return target


def colab_data_from_package(package_path, notebook_root, cache_root="/content/nih_cxr_subset"):
    """Copy just the nested data ZIP from Drive, then read images on runtime disk."""
    cache = Path(cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(notebook_root) / "shared/config.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(package_path) as package:
        name = "Notebook_2/nih_cxr_subset.zip"
        info = package.getinfo(name)
        local = cache / f"data_{info.CRC:08x}_{info.file_size}.zip"
        if not local.exists() or local.stat().st_size != info.file_size:
            temporary = local.with_suffix(".tmp")
            with package.open(name) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(local)
    return extract_data_bundle(local, config, cache / "extracted")
