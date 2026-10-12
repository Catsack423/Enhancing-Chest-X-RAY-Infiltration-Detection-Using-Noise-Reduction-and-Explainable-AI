"""Build the multi-frequency notebook and portable package using EXISTING data."""
from pathlib import Path
import ast
import json
import zipfile

import nbformat as nbf

ROOT = Path(__file__).resolve().parent

SETUP = '''from pathlib import Path
import sys, json, zipfile, shutil, subprocess, importlib, os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Paths may be changed here; keep SOURCE_RUN pointing to the existing split.
DATA_ROOT = None
OUTPUT_ROOT = None  # separate runs/frequency_fusion_existing_split
SOURCE_RUN = None   # existing 320/80/100 run; NO split generation
PACKAGE_ZIP = Path("/content/drive/MyDrive/Notebook_2_Frequency_colab.zip")

if IN_COLAB:
    drive.mount("/content/drive")
    NOTEBOOK_2_DIR = Path("/content/drive/MyDrive/Notebook_2_Frequency")
    if not PACKAGE_ZIP.is_file():
        raise FileNotFoundError("Upload Notebook_2_Frequency_colab.zip to MyDrive first")
    with zipfile.ZipFile(PACKAGE_ZIP) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename == "Notebook_2/nih_cxr_subset.zip":
                continue
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "Notebook_2":
                raise ValueError("Invalid source package path")
            target = NOTEBOOK_2_DIR.joinpath(*relative.parts[1:]).resolve()
            if not target.is_relative_to(NOTEBOOK_2_DIR.resolve()):
                raise ValueError("Invalid extraction target")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "PyWavelets",
                           "opencv-python-headless", "scikit-learn", "pandas", "matplotlib"])
    os.environ["CXR_PREPROCESS_CACHE_ROOT"] = "/content/nih_cxr_frequency_cache"
else:
    candidates = [Path.cwd(), *Path.cwd().parents,
                  Path.cwd() / "Notebook_2 release", Path.cwd() / "Notebook_2"]
    NOTEBOOK_2_DIR = next((p for p in candidates if (p / "shared/frequency.py").is_file()), None)
    if NOTEBOOK_2_DIR is None:
        raise FileNotFoundError("Run from Notebook_2 release/Frequency or set NOTEBOOK_2_DIR")

sys.path.insert(0, str(NOTEBOOK_2_DIR))
# Ensure rerunning the setup does not use modules from an older notebook folder.
for name in tuple(sys.modules):
    if name == "shared" or name.startswith("shared."):
        del sys.modules[name]
importlib.invalidate_caches()

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image
from shared.experiment import Experiment, versions
from shared.frequency import (open_frequency_experiment, FrequencyPreprocessor,
                              train_frequency, evaluate_frequency, frequency_reports)
if IN_COLAB and DATA_ROOT is None:
    from shared.data_bundle import colab_data_from_package
    DATA_ROOT = colab_data_from_package(PACKAGE_ZIP, NOTEBOOK_2_DIR,
                                       cache_root="/content/nih_cxr_frequency_data")
if DATA_ROOT is None:
    DATA_ROOT = Experiment.open(NOTEBOOK_2_DIR).data_root
if SOURCE_RUN is None:
    bundled = NOTEBOOK_2_DIR / "Frequency/source_run"
    SOURCE_RUN = bundled if bundled.is_dir() else NOTEBOOK_2_DIR / "runs/pilot320_val80_test100"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if IN_COLAB and DEVICE != "cuda":
    raise RuntimeError("Select Runtime > Change runtime type > GPU")
exp, manifests = open_frequency_experiment(NOTEBOOK_2_DIR, DATA_ROOT, SOURCE_RUN, OUTPUT_ROOT)
print("Device:", DEVICE, "| Data:", exp.data_root, "| Output:", exp.output_root)
print(versions())
'''


def build():
    destination = ROOT / "Frequency"
    destination.mkdir(exist_ok=True)
    cells = [
        ("markdown", """# Multi-frequency: DWT L1–L4 / DFT / Baseline

ใช้ข้อมูล train เดิมของ Notebook_2 release โดยอ่าน **CSV manifests เดิมโดยตรง** ไม่สุ่ม split ใหม่
3 เงื่อนไข: baseline, DWT L1–L4 + A4 (ไม่มี CLAHE), DFT ต่ำ–กลาง–สูง

ค่าฝึกสืบทอดจาก experiment.json เดิม: train 320, validation 80, test 100, seed 42;
ResNet50 ImageNet V2, train layer4 + head, Adam lr=1e-4, batch=16 **ภาพต้นฉบับ**,
สูงสุด 50 epochs, early stopping patience=5 จาก validation loss, threshold=0.5
XAI ใช้ Grad-CAM และ SHAP บน bbox 123 ภาพ / 115 คนไข้ รวมภาพที่ทำนายผิด

**test 100 ภาพเป็น subset เดิม ไม่ใช่ official test เต็ม 12,081 ภาพ**
ผลใหม่อยู่ในโฟลเดอร์แยก และ baseline ฝึกใหม่ด้วย pipeline เดียวกัน ไม่ดึงตัวเลขเก่ามาปะปน
Notebook นี้ยังไม่มีผลการทดลองเต็ม; รันเซลล์ตามลำดับบน Colab GPU"""),
        ("markdown", """## 1. Setup
อัปโหลด `Notebook_2_Frequency_colab.zip` ไว้ที่ MyDrive แล้วเปิด notebook นี้บน Colab
ZIP มีข้อมูลภาพชุดเดิมและ manifests เดิมครบ ไม่ต้องโหลด NIH ใหม่
Local: ติดตั้ง torch, torchvision, shap, PyWavelets, opencv-python-headless, pandas,
scikit-learn, matplotlib และเปิดจากโฟลเดอร์ Frequency
บนเครื่องนี้เลือก kernel `.gpu/Scripts/python.exe` ของโปรเจกต์ ซึ่งติดตั้ง CUDA พร้อมแล้ว

ถ้าต้องการใช้ run เดิมจากที่อื่น ให้แก้ SOURCE_RUN ให้ชี้ไปโฟลเดอร์ที่มี
`shared/artifacts/experiment.json` และ `shared/artifacts/manifests/*.csv`
หากไฟล์หายหรือ hash ไม่ตรง notebook จะหยุด ไม่เลือกภาพชุดใหม่แทน"""),
        ("code", SETUP),
        ("markdown", "## 2. ยืนยันข้อมูลเดิมและค่าฝึก\nตรวจ SHA-256 ของทั้งห้า manifests, metadata, official split, patient leakage และ bbox ก่อนฝึก"),
        ("code", '''display(pd.DataFrame([{"split": name, "images": len(frame), "patients": frame.patient_id.nunique()}
                      for name, frame in manifests.items()]))
display(manifests["xai"].risk_flag.value_counts().rename("images").to_frame())
print(json.dumps({k: exp.config[k] for k in ("seed", "training", "cnn", "xai", "conditions")}, indent=2))
print("Inherited manifest hashes:", exp.config["frequency_fusion"]["source_manifest_hashes"])
assert all((manifests["train"].label == label).sum() == exp.config["data"]["train_per_class"] for label in (0, 1))'''),
        ("markdown", """## 3. รวม feature อย่างไร

แต่ละภาพแยกเป็น V ภาพความถี่ที่มีพิกัดตรงกับภาพต้นฉบับ:

* Baseline: ภาพดิบ 1 view; preprocessing เหมือน notebook เดิมทุกค่า
* DWT: ใช้ db1, periodization, 4 levels; inverse DWT แยก **D1,D2,D3,D4**
  (แต่ละ Dj รวมรายละเอียด 3 ทิศทางในระดับนั้น) และ **A4** ที่เก็บโครงสร้างความถี่ต่ำ
  รวม 5 views โดยไม่มี CLAHE และไม่มี threshold ตัด coefficient
* DFT: FFT2 → radial masks → inverse FFT2 เป็น low/mid/high 3 views
  กำหนดขอบเขตล่วงหน้า 0.125 และ 0.25 cycles/pixel; ความถี่แกนสูงสุด 0.5
  masks ไม่ซ้อนกันและครอบคลุมครบ ไม่ใช้ภาพ magnitude spectrum เพราะพิกัดไม่ตรงกับ bbox

ทำงานที่ 256×256 ตามเดิม แล้ว resize เป็น 224×224
แต่ละ view เป็น grayscale ทำซ้ำ RGB และใช้ ImageNet mean/std เดิม
low/A4 ใช้ค่าเดิม; signed details ใช้ 0.5 + 0.5×ค่า (ไม่มี clipping หรือ min-max ต่อภาพ)

`ภาพ → [view 1 ... view V] → ResNet50 น้ำหนักชุดเดียว → [f1 ... fV] → mean(f) → Linear(2048,1)`

**รวมด้วยการเฉลี่ย feature หลัง backbone** ไม่ใช่บวกภาพกลับก่อน CNN
เพราะ head เป็น Linear จึงคำนวณเทียบเท่าด้วยการเฉลี่ย logits ก่อน sigmoid ได้
จำนวนพารามิเตอร์เท่าเดิมทุกวิธี; ไม่มีการเพิ่ม classifier หรือเปลี่ยนเป็น ANN
L1–L4 ในที่นี้คือระดับ decomposition ที่รวมในหนึ่งวิธี ไม่ใช่ 4 ระดับความแรง denoising

batch 16 ภาพเดิมกลายเป็น 80 views สำหรับ DWT และ 48 views สำหรับ DFT:
เวลา/หน่วยความจำมากขึ้น และ BatchNorm ใน layer4 จะเห็น view distribution ต่างจาก baseline
จึงควบคุม hyperparameters และจำนวนภาพเดิมได้ แต่ไม่ได้ใช้ compute budget เท่ากัน
หากปรับ cutoff หรือ fusion ให้ใช้ validation เท่านั้น และตั้ง OUTPUT_ROOT ใหม่"""),
        ("code", '''import matplotlib.pyplot as plt
row = manifests["train"].iloc[0].to_dict()
for condition_id in exp.condition_ids:
    prep = FrequencyPreprocessor(exp, condition_id)
    bands = prep.components(row)
    names = {"baseline": ["raw"], "dwt_L1_L4_fusion": ["A4", "D1", "D2", "D3", "D4"],
             "dft_low_mid_high_fusion": ["low", "mid", "high"]}[condition_id]
    fig, axes = plt.subplots(1, len(bands), figsize=(4 * len(bands), 4), squeeze=False)
    for ax, band, name in zip(axes[0], bands, names):
        ax.imshow(band, cmap="gray" if name in ("raw", "A4", "low") else "seismic")
        ax.set_title(name); ax.axis("off")
    fig.suptitle(condition_id + " — display autoscaling only")
    display(fig); plt.close(fig)
    print(condition_id, "input shape:", tuple(prep.tensor(row).shape))
del prep'''),
        ("markdown", """## 4. ฝึกและวัด classification
ใช้ loss, optimizer, per-epoch image order, initialization seed และ early stopping เดิม
เซลล์นี้ฝึกจริงทั้ง 3 วิธี; รันซ้ำจะ resume จาก last.pt
RTX 4060 เครื่องนี้ประเมินรันครบ training + XAI ประมาณ 45–90 นาที; เวลา Colab ขึ้นกับ GPU
เลือก best.pt จาก validation loss; test ใช้ประเมิน checkpoint ที่เลือกแล้วเท่านั้น
บันทึก Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix, predictions, history และเวลา
ค่า delta ในตารางอยู่บนสเกล 0–1 เช่น +0.04 = +4 percentage points สำหรับ accuracy"""),
        ("code", '''for condition_id in exp.condition_ids:
    metrics = train_frequency(exp, manifests, condition_id, DEVICE)
    print(condition_id, {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
    cnn_table, xai_table, paired_deltas, status = frequency_reports(exp)
    display(status)
display(cnn_table)'''),
        ("markdown", """## 5. Grad-CAM บนทั้ง 123 ภาพ
อธิบาย Infiltration logit ผ่าน layer4 ของทุก view รวม signed contributions ก่อน ReLU
resize กลับพิกัดต้นฉบับ แล้ววัด IoU threshold 0.3/0.5 และ Pointing Game
ไม่มีการเลือกเฉพาะภาพที่จำแนกถูก; heatmap ว่างนับเป็น miss / IoU=0"""),
        ("code", '''for condition_id in exp.condition_ids:
    records = evaluate_frequency(exp, manifests, "Grad-CAM", condition_id, DEVICE)
    print(condition_id, "evaluated:", len(records))
    frequency_reports(exp)'''),
        ("markdown", """## 6. SHAP Expected Gradients
ใช้ GradientExplainer บนโมเดลเดียวกันและ background IDs เดิม 32 ภาพจาก train
background ผ่าน preprocessing ของวิธีนั้น; nsamples=200, batch_size=8 ตามเดิม
บันทึก signed attribution ทุก view/RGB ก่อนรวม แล้วใช้ส่วนบวกเทียบ bbox
SHAP อธิบาย **processed frequency inputs** ไม่ได้ backprop ผ่าน NumPy FFT/DWT ไปยัง raw pixels
แผนที่อยู่ในพิกัดภาพเดียวกัน แต่การกรองความถี่อาจกระจายสัญญาณเชิงพื้นที่/เกิด ringing
จึงไม่ตีความเป็น causal attribution ของ raw pixels หรือ lesion segmentation
เซลล์นี้ใช้เวลามากและ resume รายภาพได้"""),
        ("code", '''for condition_id in exp.condition_ids:
    records = evaluate_frequency(exp, manifests, "SHAP", condition_id, DEVICE)
    print(condition_id, "evaluated:", len(records))
    frequency_reports(exp)'''),
        ("markdown", """## 7. ตารางเปรียบเทียบและ 95% CI
XAI รายงาน Overall / Pure / Mixed-high-risk / Mixed-low-risk ด้วย patient bootstrap 2,000 รอบ
ตาราง paired_xai_deltas ใช้ภาพและคนไข้คู่เดียวกับ baseline; CI ของความต่างช่วยประเมินความไม่แน่นอน
classification delta เป็น descriptive และไม่ได้มี significance test ใน notebook นี้
อย่าตัดสินว่าดีขึ้นอย่างมีนัยสำคัญจากขนาด delta เพียงอย่างเดียว
ไม่รับประกันว่าวิธีใหม่ชนะ baseline; ชุด test 100 ภาพมีข้อจำกัดและยังไม่ใช่ full-test evaluation"""),
        ("code", '''cnn_table, xai_table, paired_deltas, status = frequency_reports(exp)
display(status)
display(cnn_table)
display(xai_table)
display(paired_deltas)
for condition_id in exp.condition_ids:
    display(Image(filename=str(exp.results("CNN", condition_id) / "confusion_matrix.png")))
example = manifests["xai"].iloc[0]["image"].rsplit(".", 1)[0]
for criterion in ("Grad-CAM", "SHAP"):
    for condition_id in exp.condition_ids:
        print(criterion, condition_id)
        display(Image(filename=str(exp.results(criterion, condition_id) / "overlays" / (example + ".png"))))
print("All measured outputs:", exp.output_root)'''),
        ("markdown", """## 8. Export
ZIP รวม checkpoints, manifests, predictions, heatmaps และตารางผลจริง ไม่รวม image cache
ผลหลักคงอยู่บน Drive แม้ Colab disconnect"""),
        ("code", '''export_path = exp.output_root.parent / "frequency_fusion_results.zip"
with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in exp.output_root.rglob("*"):
        if path.is_file() and "cache" not in path.relative_to(exp.output_root).parts:
            archive.write(path, path.relative_to(exp.output_root).as_posix())
print(export_path)
if IN_COLAB:
    from google.colab import files
    files.download(str(export_path))'''),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(text) if kind == "markdown" else nbf.v4.new_code_cell(text)
                for kind, text in cells]
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"name": "frequency_comparison.ipynb", "provenance": []}}
    nbf.validate(nb)
    path = destination / "frequency_comparison.ipynb"
    nbf.write(nb, path)
    print(path)
    return path


def package():
    """Reuse the verified existing data ZIP; include exact saved manifests."""
    notebook = build()
    source_run = ROOT / "runs/pilot320_val80_test100/shared/artifacts"
    destination = ROOT / "Notebook_2_Frequency_colab.zip"
    temporary = destination.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        sources = [*sorted((ROOT / "shared").glob("*.py")), ROOT / "shared/config.json",
                   ROOT / "build_frequency_notebook.py", notebook, ROOT / "Frequency/README.md",
                   ROOT / "Frequency/validate_notebook_gpu.py"]
        if (ROOT / "Frequency/VALIDATION.md").exists():
            sources.append(ROOT / "Frequency/VALIDATION.md")
        for path in sources:
            archive.write(path, "Notebook_2/" + path.relative_to(ROOT).as_posix())
        for path in [source_run / "experiment.json", *sorted((source_run / "manifests").glob("*"))]:
            if path.is_file():
                archive.write(path, "Notebook_2/Frequency/source_run/shared/artifacts/" + path.relative_to(source_run).as_posix())
        archive.write(ROOT / "nih_cxr_subset.zip", "Notebook_2/nih_cxr_subset.zip", compress_type=zipfile.ZIP_STORED)
    temporary.replace(destination)
    print(f"{destination}: {destination.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    package()
