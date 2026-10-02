# Export this run's measured results and checkpoints; no retraining required.
from pathlib import Path
from datetime import datetime, timezone
import tempfile
import zipfile
import os

source = Path(exp.output_root).resolve()
if not source.is_dir():
    raise FileNotFoundError(source)
# Regenerate summaries before exporting the existing measured results.
refresh_reports(exp)
stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
export_dir = Path("/content") if IN_COLAB else Path(tempfile.gettempdir())
archive = export_dir / f"{source.name}_results_{stamp}.zip"
partial = archive.with_suffix(".zip.partial")
count = 0
try:
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as zipped:
        for path in sorted(source.rglob("*")):
            relative = path.relative_to(source)
            # Recomputable image caches are omitted; retain all result/provenance files.
            if any(part in {"cache", "__pycache__"} for part in relative.parts):
                continue
            if path.is_file():
                zipped.write(path, (Path(source.name) / relative).as_posix())
                count += 1
    if not count:
        raise RuntimeError("No result files found for export")
    os.replace(partial, archive)
except Exception:
    partial.unlink(missing_ok=True)
    raise
print(f"Exported {count} files: {archive} ({archive.stat().st_size / 1024**2:.1f} MiB)")
print("Contains existing CNN / Grad-CAM / SHAP results, checkpoints and shared artifacts.")
if IN_COLAB:
    from google.colab import files
    files.download(str(archive))
    print("If the browser blocks the download, download this ZIP from Colab's Files panel.")
