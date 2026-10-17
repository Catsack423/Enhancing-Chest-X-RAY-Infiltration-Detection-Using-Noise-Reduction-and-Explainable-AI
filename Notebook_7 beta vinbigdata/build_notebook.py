"""Build one self-contained Colab notebook for VinBigData 512 and a source-only recovery package."""
from pathlib import Path
import ast
import base64
import hashlib
import io
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def build():
    paths = []
    for folder in ("v7", "configs", "tests"):
        folder_path = ROOT / folder
        if folder_path.exists():
            paths.extend(p for p in folder_path.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".csv"))
    paths.extend(
        ROOT / name
        for name in ("run.py", "progress.py", "xai_interface.py", "build_notebook.py", "requirements.txt", "VENDOR_MANIFEST.json", "README.md")
        if (ROOT / name).exists()
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(paths)):
            entry = zipfile.ZipInfo("Notebook_7_beta_vinbigdata/" + path.relative_to(ROOT).as_posix(), date_time=(2026, 10, 17, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, path.read_bytes())
    source = buffer.getvalue()
    (ROOT / "v7_source.zip").write_bytes(source)
    encoded = base64.b64encode(source).decode()
    sha = hashlib.sha256(source).hexdigest()

    intro = """# v7 beta (VinBigData 512) — class160: noise audit → noise simulation → CNN

ใช้ชุดข้อมูล **`awsaf49/vinbigdata-512-image-dataset`** ดึงผ่าน `kagglehub` ลงบน Colab VM โดยตรง
แบ่งเป็น **2 คลาส**:
- **`normal` (Class 0)**: ภาพปอดปกติ (`No finding` / `class_id == 14` ไม่มีภาวะโรคอื่นปน; ~10,606 ภาพ)
- **`infiltration` (Class 1)**: ภาพปอดที่มีภาวะ `Infiltration` (`class_name == 'Infiltration'` หรือ `class_id == 6`) แม้จะมีโรคอื่นร่วมด้วยในภาพเดียวกัน (~613 ภาพ)

คัดภาพ estimated noise ต่ำตามสัดส่วนผู้สมัคร (seed 42):
- **Train**: `160 normal + 160 infiltration` (**320 ภาพ**)
- **Validation**: `40 normal + 40 infiltration` (**80 ภาพ**)
- **Test**: `50 normal + 50 infiltration` (**100 ภาพ**)

ทดลอง **12 วิธีลดและสกัดความถี่ Noise**, จำลอง noise 5 ชนิด × 3 ระดับ + ภาพก่อนเติม (รวม 16 สภาวะ), seed 42
ฝึก **1 DAE + 12 CNN (ResNet50)**; เก็บ checkpoint ทุก epoch และข้อมูลกรอบ Bounding Box สำหรับ XAI บน Google Drive

1. เลือก **Runtime → Change runtime type → GPU**
2. กด **Run all** แล้วอนุญาต mount Google Drive
3. ผลลัพธ์และข้อมูลที่คัดแล้วจะถูกบันทึกไว้ที่ `MyDrive/nih -xray/Notebook_7_beta_vinbigdata/` (หาก Colab หลุด สามารถกด Run all ใหม่เพื่อรันต่อจาก checkpoint เดิมได้ทันทีโดยไม่ต้องโหลดหรือคัดภาพใหม่)
"""

    setup = f'''#@title ติดตั้งแพ็กเกจ แตกซอร์สโค้ด และเชื่อมต่อ Google Drive
from pathlib import Path
import base64, hashlib, subprocess, sys, zipfile, io
import importlib.util
from google.colab import drive
dependencies = {{
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "cv2": "opencv-python-headless",
    "pywt": "PyWavelets",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "nbformat": "nbformat",
    "kagglehub": "kagglehub",
}}
missing = [package for module, package in dependencies.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
SOURCE_SHA256 = {sha!r}
SOURCE_B64 = {encoded!r}
payload = base64.b64decode(SOURCE_B64)
if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
    raise ValueError("Embedded source integrity failed")
SOURCE_STAGE = Path("/content/v7_vinbigdata_source_").with_name("v7_vinbigdata_source_" + SOURCE_SHA256[:16])
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for entry in archive.infolist():
        if not (SOURCE_STAGE / entry.filename).resolve().is_relative_to(SOURCE_STAGE.resolve()):
            raise ValueError("Source path escapes staging")
    archive.extractall(SOURCE_STAGE)
ROOT = SOURCE_STAGE / "Notebook_7_beta_vinbigdata"
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
from v7.colab import download_vinbigdata
WORK = MOUNT / "MyDrive/nih -xray/Notebook_7_beta_vinbigdata"  # โฟลเดอร์เก็บผลและ checkpoint ถาวรบน Drive
WORK.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_HOME"] = str(WORK / "cache/torch")  # reuse ImageNet weights across runtimes
DATA_ROOT = download_vinbigdata("awsaf49/vinbigdata-512-image-dataset", work=WORK)
print("VinBigData VM source:", DATA_ROOT)
print("Durable Drive outputs:", WORK)
'''

    prepare = '''from progress import prepare_with_logs
prepare_with_logs(ROOT, DATA_ROOT, WORK, workers=4)
'''

    stage = '''from v7.colab import stage_selected
LOCAL_DATA = stage_selected(WORK, Path("/content/v7_vinbigdata_cache"))
print("VM-local training images:", LOCAL_DATA)
'''

    train = '''from v7.runner import run
if not torch.cuda.is_available():
    raise RuntimeError("Runtime > Change runtime type > GPU แล้ว Run all ใหม่")
EXPERIMENT = run(ROOT, LOCAL_DATA, WORK / "runs/v7_beta_vinbigdata/seed_42", device="cuda")
'''

    results = '''import pandas as pd
REPORTS = WORK / "runs/v7_beta_vinbigdata/seed_42/reports"
table = pd.read_csv(REPORTS / "classification.csv")
display(table[["condition_id", "corruption_id", "roc_auc", "pr_auc", "balanced_accuracy", "f1", "accuracy"]])
print("Reports:", REPORTS)
print("Checkpoints / XAI loader metadata:", REPORTS.parent)
'''

    notebook = nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell(intro),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_markdown_cell("## 1. ดาวน์โหลด `awsaf49/vinbigdata-512-image-dataset` และตั้งค่าโฟลเดอร์บน Drive"),
        nbf.v4.new_code_cell(configure),
        nbf.v4.new_markdown_cell("## 2. แยกคลาส `normal` vs `infiltration` → วิเคราะห์ระดับ Noise → คัดภาพเข้าชุดทดลอง\nมี log สถานะทุก 10 วินาที และบันทึกชุดภาพที่คัดลง Google Drive"),
        nbf.v4.new_code_cell(prepare),
        nbf.v4.new_markdown_cell("## 3. โหลดชุดภาพที่คัดแล้ว (Train 160+160, Val 40+40, Test 50+50) เข้าสู่ Colab VM"),
        nbf.v4.new_code_cell(stage),
        nbf.v4.new_markdown_cell("## 4. ฝึกสอน 1 DAE และ 12 CNN (ResNet50) — บันทึก Checkpoint ทุก Epoch ลง Drive"),
        nbf.v4.new_code_cell(train),
        nbf.v4.new_markdown_cell("## 5. สรุปผลการจำแนกโรค (Normal vs Infiltration) และคุณภาพภาพ"),
        nbf.v4.new_code_cell(results),
    ])
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "run_v7_vinbigdata_colab.ipynb"},
        "accelerator": "GPU",
    }
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
