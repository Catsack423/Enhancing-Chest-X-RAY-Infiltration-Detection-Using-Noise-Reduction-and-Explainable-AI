"""Build one self-contained Colab notebook and a source-only recovery package."""
from pathlib import Path
import ast
import base64
import hashlib
import io
import json
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def build():
    paths = []
    for folder in ("v7", "configs", "partition_reference", "v3_zip_reference", "tests", "validation"):
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".csv"))
    paths.extend(ROOT / name for name in ("run.py", "progress.py", "xai_interface.py", "build_notebook.py", "requirements.txt", "VENDOR_MANIFEST.json", "README.md", "VALIDATION.md") if (ROOT / name).exists())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            entry = zipfile.ZipInfo("Notebook_7_beta/" + path.relative_to(ROOT).as_posix(), date_time=(2026, 10, 16, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    source = buffer.getvalue()
    (ROOT / "v7_source.zip").write_bytes(source)
    encoded = base64.b64encode(source).decode()
    sha = hashlib.sha256(source).hexdigest()
    intro = """# v7 beta — class160: noise audit → noise simulation → CNN

ใช้ ZIP v3 เดิมจาก `MyDrive/nih -xray/data/v3_selected_zip/` **70,012 ภาพ**
คัดลอก ZIP ทีละส่วน แตกและวิเคราะห์บน Colab VM แล้วคัดภาพ estimated noise ต่ำ
ชุด ZIP นี้เป็นส่วนหนึ่งของ NIH 112,120 ภาพ; ไม่รายงานว่า audit ครบ NIH ทั้งหมด
Train **160+160**, validation **40+40**, test **50+50** ตาม patient partition เดิม

ทดลอง **12 วิธีเดิม ไม่มี strength L4**, noise 5 ชนิด × 3 ระดับ + ภาพก่อนเติม, seed 42
ฝึก **1 DAE + 12 CNN**; เก็บ checkpoint และโครงสำหรับ XAI แต่ **ยังไม่รัน Grad-CAM/SHAP**

1. เลือก **Runtime → Change runtime type → GPU**
2. กด **Run all** แล้วอนุญาต mount Google Drive ด้วย **faceyuslay@gmail.com เท่านั้น**
3. ผลและข้อมูลใช้ซ้ำอยู่ `MyDrive/nih -xray/Notebook_7_beta/`

ถ้า runtime หลุด เปิด notebook นี้แล้ว Run all ใหม่ จะใช้ audit chunks / ชุดภาพที่คัด / checkpoint บน Drive ต่อ
ไม่ต้องอัปโหลด ZIP ภาพใหม่ ใช้ loader v3 เดิมและ notebook มี source code ฝังไว้แล้ว
มี log ทันทีและสถานะทุก 10 วินาที ทั้งโหลด ZIP / แตกไฟล์ / วิเคราะห์ noise

ภาพที่คัดเป็น **low estimated noise reference** ไม่อ้างว่าเป็นภาพไร้ noise จริง
รอบนี้เป็น beta pilot หนึ่ง seed ไม่ใช่ผลสรุปหลาย training seeds
"""
    setup = f'''#@title ติดตั้งและเชื่อมต่อ Google Drive
from pathlib import Path
import base64, hashlib, subprocess, sys, zipfile, io
import importlib.util
from google.colab import drive
dependencies = {{"numpy": "numpy", "pandas": "pandas", "PIL": "Pillow", "cv2": "opencv-python-headless", "pywt": "PyWavelets", "scipy": "scipy", "sklearn": "scikit-learn", "matplotlib": "matplotlib", "nbformat": "nbformat"}}
missing = [package for module, package in dependencies.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
SOURCE_SHA256 = {sha!r}
SOURCE_B64 = {encoded!r}
payload = base64.b64decode(SOURCE_B64)
if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
    raise ValueError("Embedded source integrity failed")
SOURCE_STAGE = Path("/content/v7_source_") .with_name("v7_source_" + SOURCE_SHA256[:16])
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        if not (SOURCE_STAGE / entry.filename).resolve().is_relative_to(SOURCE_STAGE.resolve()):
            raise ValueError("Source path escapes staging")
    archive.extractall(SOURCE_STAGE)
ROOT = SOURCE_STAGE / "Notebook_7_beta"
loaded = sys.modules.get("v7")
if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(ROOT.resolve()):
    raise RuntimeError("Notebook source changed; restart the Colab session before loading this version")
sys.path.insert(0, str(ROOT))
from v7.colab import mount_on_fresh_path
MOUNT = mount_on_fresh_path(drive.mount)
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU selected")
'''
    configure = '''import os
ZIP_ROOT = MOUNT / "MyDrive/nih -xray/data/v3_selected_zip"
DATA_ROOT = Path("/content/cxr_v7_originals")  # แตก ZIP ลง VM; ไม่อ่าน PNG ทีละภาพบน Drive
WORK = MOUNT / "MyDrive/nih -xray/Notebook_7_beta"  # ผลและข้อมูลใช้ซ้ำ
if not (ZIP_ROOT / "dataset_zip_index.json").is_file():
    raise FileNotFoundError(f"Missing v3 ZIP catalog: {ZIP_ROOT}; authorize faceyuslay@gmail.com and check folder access")
WORK.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(WORK / "cache/torch")  # reuse ImageNet weights across runtimes
print("Read-only ZIP source:", ZIP_ROOT)
print("VM-local originals:", DATA_ROOT)
print("Durable outputs:", WORK)
'''
    prepare = '''from progress import prepare_with_logs
prepare_with_logs(ROOT, DATA_ROOT, WORK, workers=4, zip_root=ZIP_ROOT)
'''
    stage = '''from v7.colab import stage_selected
LOCAL_DATA = stage_selected(WORK, Path("/content/v7_cache"))
print("VM-local training images:", LOCAL_DATA)
'''
    train = '''from v7.runner import run
if not torch.cuda.is_available():
    raise RuntimeError("Runtime > Change runtime type > GPU แล้ว Run all ใหม่")
EXPERIMENT = run(ROOT, LOCAL_DATA, WORK / "runs/v7_beta/seed_42", device="cuda")
'''
    results = '''import pandas as pd
REPORTS = WORK / "runs/v7_beta/seed_42/reports"
table = pd.read_csv(REPORTS / "classification.csv")
display(table[["condition_id", "corruption_id", "roc_auc", "pr_auc", "balanced_accuracy", "f1", "accuracy"]])
print("Reports:", REPORTS)
print("Checkpoints / XAI loader metadata:", REPORTS.parent)
'''
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(intro), nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell("## 1. ตำแหน่งข้อมูลและผลบน Drive"), nbf.v4.new_code_cell(configure),
        nbf.v4.new_markdown_cell("## 2. โหลด ZIP v3 → แตกบน Colab → วิเคราะห์ noise 70,012 ภาพ\nใช้ loader v3 เดิม มี log ทุก 10 วินาที; checkpoint และชุดที่คัดบันทึกลง Drive"), nbf.v4.new_code_cell(prepare),
        nbf.v4.new_markdown_cell("## 3. โหลดชุดภาพที่คัดจาก Drive เข้า Colab VM"), nbf.v4.new_code_cell(stage),
        nbf.v4.new_markdown_cell("## 4. ฝึก DAE และ CNN 12 วิธี — เก็บ checkpoint ทุก epoch บน Drive"), nbf.v4.new_code_cell(train),
        nbf.v4.new_markdown_cell("## 5. ผล CNN และคุณภาพภาพ"), nbf.v4.new_code_cell(results)])
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "run_v7_colab_v3_zip.ipynb"}, "accelerator": "GPU"}
    notebook.cells[1].metadata["cellView"] = "form"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nbf.validate(notebook)
    output = ROOT / "run_v7_colab.ipynb"
    nbf.write(notebook, output)
    print(f"{output}: embedded source {len(source)/1024:.0f} KiB; {len(paths)} files; source SHA256 {sha}")
    return output


if __name__ == "__main__":
    build()
