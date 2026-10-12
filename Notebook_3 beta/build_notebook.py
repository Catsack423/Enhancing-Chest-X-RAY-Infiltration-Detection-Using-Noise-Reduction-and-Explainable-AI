"""Build the standalone v3 beta notebook and Colab package; never train."""
import ast
from pathlib import Path
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

DATA_ROOT = None
OUTPUT_ROOT = None  # defaults to runs/head_only_e2, separate from v2
SOURCE_RUN = None   # exact existing 320/80/100 manifests; never re-split
PACKAGE_ZIP = Path("/content/drive/MyDrive/Notebook_3_beta_colab.zip")

if IN_COLAB:
    drive.mount("/content/drive")
    NOTEBOOK_3_DIR = Path("/content/drive/MyDrive/Notebook_3_beta")
    if not PACKAGE_ZIP.is_file():
        raise FileNotFoundError("Upload Notebook_3_beta_colab.zip to MyDrive first")
    with zipfile.ZipFile(PACKAGE_ZIP) as archive:
        for member in archive.infolist():
            if member.is_dir() or member.filename == "Notebook_3/nih_cxr_subset.zip":
                continue
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "Notebook_3":
                raise ValueError("Invalid package path")
            target = NOTEBOOK_3_DIR.joinpath(*relative.parts[1:]).resolve()
            if not target.is_relative_to(NOTEBOOK_3_DIR.resolve()):
                raise ValueError("Invalid extraction target")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "PyWavelets",
                           "opencv-python-headless", "scikit-learn", "pandas", "matplotlib"])
    os.environ["CXR_PREPROCESS_CACHE_ROOT"] = "/content/nih_cxr_v3_cache"
else:
    candidates = [Path.cwd(), *Path.cwd().parents,
                  Path.cwd() / "Notebook_3 beta", Path.cwd() / "Notebook_3"]
    NOTEBOOK_3_DIR = next((p for p in candidates if (p / "shared/beta.py").is_file()), None)
    if NOTEBOOK_3_DIR is None:
        raise FileNotFoundError("Run from Notebook_3 beta or set NOTEBOOK_3_DIR")
    old_torch_cache = NOTEBOOK_3_DIR.parent / "Notebook_2 release/.runtime/torch"
    if old_torch_cache.is_dir():
        os.environ.setdefault("TORCH_HOME", str(old_torch_cache))

sys.path.insert(0, str(NOTEBOOK_3_DIR))
for name in tuple(sys.modules):
    if name == "shared" or name.startswith("shared."):
        del sys.modules[name]
importlib.invalidate_caches()

import numpy as np
import pandas as pd
import torch
from IPython.display import display, Image
from shared.experiment import Experiment, versions
from shared.frequency import open_frequency_experiment, FrequencyPreprocessor, frequency_reports
from shared.beta import train_beta, evaluate_beta, beta_model_factory, plot_training_history
if IN_COLAB and DATA_ROOT is None:
    from shared.data_bundle import colab_data_from_package
    DATA_ROOT = colab_data_from_package(PACKAGE_ZIP, NOTEBOOK_3_DIR,
                                       cache_root="/content/nih_cxr_v3_data")
if DATA_ROOT is None:
    DATA_ROOT = Experiment.open(NOTEBOOK_3_DIR).data_root
if SOURCE_RUN is None:
    SOURCE_RUN = NOTEBOOK_3_DIR / "source_run"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    raise RuntimeError("Select a CUDA/GPU runtime before training or XAI")
exp, manifests = open_frequency_experiment(NOTEBOOK_3_DIR, DATA_ROOT, SOURCE_RUN, OUTPUT_ROOT)
print("Device:", torch.cuda.get_device_name(0), "| Data:", exp.data_root, "| Output:", exp.output_root)
print(versions())
'''


def build():
    cells = [
        ("markdown", """# Notebook v3 beta — head only, regularization, epochs = 2

ต่อจากโค้ด v2 frequency comparison และใช้ **ภาพเดิมทุกภาพ**: train 320 (160/160),
validation 80 (40/40), test 100 (50/50), bbox XAI 123 ภาพ / 115 คนไข้
อ่าน manifests เดิมและตรวจ hashes ไม่มีการสุ่ม split ใหม่

เปรียบเทียบ **Baseline / DWT L1–L4 + A4 ไม่ใช้ CLAHE / DFT ต่ำ–กลาง–สูง**
ทั้ง 3 วิธี freeze ResNet50 ImageNet V2 และ BatchNorm ทุกชั้น
ฝึกเฉพาะ `Dropout(0.3) → Linear(2048,1)` ด้วย AdamW weight_decay=1e-4
lr=1e-4, batch 16, seed 42, **epochs=2** และ threshold 0.5

พารามิเตอร์ฝึกลดจากประมาณ 14.97 ล้านใน v2 เหลือ **2,049**
นี่คือ protocol ใหม่เพื่อบรรเทา overfit ต้องฝึก baseline ใหม่เทียบกันใน v3
ยังไม่มีผลวิจัยจาก notebook นี้ ไม่รับประกัน Accuracy เพิ่ม +10 จุดเปอร์เซ็นต์
และการฝึกเพียง 2 epochs อาจ underfit; ต้องอ่าน train/validation loss และผลจริงก่อนสรุป"""),
        ("markdown", """## 1. Setup
Colab: อัปโหลด `Notebook_3_beta_colab.zip` ไป MyDrive เปิด notebook นี้ และเลือก GPU
ZIP แนบภาพชุดเดิมของ v2, manifests และโค้ดครบ ไม่ต้องโหลด NIH ใหม่
Local: เลือก kernel `D:/kaggle_cache/datasets/nih-chest-xrays/.gpu/Scripts/python.exe`
เปิดจากโฟลเดอร์ `Notebook_3 beta` ค่าตั้งทั้งหมดอยู่ใน `shared/config.json`
ผลเขียนลง `runs/head_only_e2`; ห้ามนำ checkpoint v2 มา resume protocol นี้"""),
        ("code", SETUP),
        ("markdown", """## 2. ตรวจข้อมูลและยืนยันว่า freeze จริง
แสดงจำนวนคลาสและคนไข้จากไฟล์ที่รันจริง ไม่ใช้ class weighting เพราะชุด train สมดุล
ตรวจ metadata/official split/bbox/patient leakage และ hashes ก่อนเปิด experiment
model probe ไม่ใช้ pretrained download และไม่ฝึก"""),
        ("code", '''display(pd.DataFrame([{"split": name, "images": len(frame),
                          "patients": frame.patient_id.nunique(),
                          "normal": int((frame.label == 0).sum()) if "label" in frame else None,
                          "infiltration": int((frame.label == 1).sum()) if "label" in frame else None}
                         for name, frame in manifests.items()]))
print(json.dumps(exp.config["cnn"], indent=2))
print("Inherited manifest hashes:", exp.config["frequency_fusion"]["source_manifest_hashes"])
probe = beta_model_factory(exp)(pretrained=False, weights_name=exp.config["cnn"]["weights"])
probe.train()
trainable = {name: p.numel() for name, p in probe.named_parameters() if p.requires_grad}
assert sum(trainable.values()) == 2049
assert all(name.startswith("head.") for name in trainable)
assert all(not m.training for m in probe.backbone.modules())
assert not any(p.requires_grad for p in probe.backbone.parameters())
assert exp.config["cnn"]["epochs"] == 2
print("Trainable parameters:", trainable, "| Total:", sum(trainable.values()))
del probe'''),
        ("markdown", """## 3. Feature fusion และ regularization

- Baseline: raw 1 view ใช้ preprocessing เดิม
- DWT: inverse DWT ของ A4 + D1/D2/D3/D4 (db1, periodization) เป็น 5 views
  ไม่ใช้ CLAHE และไม่ตัด coefficient; Dj รวมรายละเอียด 3 ทิศทาง
- DFT: radial low <0.125, mid 0.125–<0.25, high ≥0.25 cycles/pixel แล้ว inverse FFT เป็น 3 views
  masks ไม่ซ้อนกันและครอบคลุมทุก coefficient

`views → ResNet50 (frozen/eval) → mean(feature 2048) → Dropout(0.3) → Linear → logit`

Dropout ใช้ **หลังเฉลี่ย feature** เป็นหนึ่ง mask ต่อภาพต้นฉบับทุกวิธี
ไม่ใช้ dropout แยกตาม view ซึ่งจะเปลี่ยนความแรง regularization ตามจำนวน views
ตอน validation/test/XAI ปิด dropout และใช้ BatchNorm statistics ของ pretrained backbone
optimizer เห็นเฉพาะ head; backbone ไม่มี gradient ของ weights แต่ยังหา gradient ของ input สำหรับ XAI ได้

ทำงานที่ 256×256/resize 224×224/RGB/ImageNet mean/std เดิม
low/A4 ใช้ค่าเดิม; signed detail ใช้ 0.5+0.5×component ไม่มี clipping หรือ min-max ต่อภาพ
ไม่มี augmentation หรือ class weighting เพิ่มใน beta นี้ เพื่อคง preprocessing เดิม
batch 16 ภาพเป็น 80 DWT views/48 DFT views; จำนวนภาพและ hyperparameters เท่ากัน แต่ compute ต่างกัน"""),
        ("code", '''import matplotlib.pyplot as plt
row = manifests["train"].iloc[0].to_dict()
names = {"baseline": ["raw"], "dwt_L1_L4_fusion": ["A4", "D1", "D2", "D3", "D4"],
         "dft_low_mid_high_fusion": ["low", "mid", "high"]}
for condition_id in exp.condition_ids:
    prep = FrequencyPreprocessor(exp, condition_id)
    bands = prep.components(row)
    fig, axes = plt.subplots(1, len(bands), figsize=(4*len(bands), 4), squeeze=False)
    for ax, band, name in zip(axes[0], bands, names[condition_id]):
        ax.imshow(band, cmap="gray" if name in ("raw", "A4", "low") else "seismic")
        ax.set_title(name); ax.axis("off")
    fig.suptitle(condition_id + " — display autoscaling only")
    display(fig); plt.close(fig)
    print(condition_id, "input shape:", tuple(prep.tensor(row).shape))
del prep'''),
        ("markdown", """## 4. CNN: ฝึกทั้ง 3 วิธี 2 epochs
เซลล์นี้ฝึกจริงจาก ImageNet initialization แบบเดียวกันทุกวิธี และบันทึก best.pt/last.pt
เลือก best epoch 1 หรือ 2 จาก validation loss; `e=2` คือจำนวน epochs ไม่ได้บังคับเลือก checkpoint epoch 2
ไม่ดู test เพื่อเลือก checkpoint/threshold และไม่ใช้ผล baseline 54% ของ v2 แทน baseline v3

RTX 4060 ประเมิน training + classification ประมาณ 1–5 นาที ไม่รวม download ครั้งแรก
XAI โดยเฉพาะ SHAP ใช้เวลามากกว่า; 2 epochs ไม่ได้ลดจำนวนภาพ XAI
รันซ้ำ resume ได้ หาก config/code/versions เปลี่ยนให้ใช้ OUTPUT_ROOT ใหม่"""),
        ("code", '''for condition_id in exp.condition_ids:
    metrics = train_beta(exp, manifests, condition_id, DEVICE)
    assert metrics["epochs_run"] == 2 and metrics["trainable_parameters"] == 2049
    print(condition_id, {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "best_epoch")})
    cnn_table, xai_table, paired_deltas, status = frequency_reports(exp)
    display(status)
display(cnn_table)'''),
        ("markdown", """## 5. ตรวจ train/validation gap
`train_loss` ระหว่าง update เปิด dropout และเปลี่ยนน้ำหนักไปแต่ละ batch
จึงบันทึก **train_eval_loss** หลังจบ epoch ด้วย model.eval() เช่นเดียวกับ validation
ใช้ train_eval_loss เทียบ validation_loss เพื่ออ่าน gap; ไม่ใช้ train Accuracy ว่าเป็น test Accuracy
มีเพียง 2 จุดต่อวิธี ยังสรุปว่าแก้ overfit สำเร็จไม่ได้ หากทั้งสอง loss สูงอาจเป็น underfit"""),
        ("code", '''curve_path = plot_training_history(exp)
display(Image(filename=str(curve_path)))
for condition_id in exp.condition_ids:
    print(condition_id)
    display(pd.read_csv(exp.results("CNN", condition_id) / "history.csv"))'''),
        ("markdown", """## 6. Grad-CAM: bbox 123 ภาพเดิม
อธิบาย Infiltration logit ผ่าน layer4 ของ frozen backbone ทุก view
เปิด gradient ที่ input และรวม signed contributions ก่อน ReLU; ไม่ใช้ no_grad ครอบ XAI
ใช้ checkpoint ที่เลือกจาก validation และอธิบายทุกภาพ รวมคำทำนายผิด
รายงาน IoU@0.3/0.5, Pointing Game และ energy inside; heatmap ว่างเป็น miss/IoU 0"""),
        ("code", '''for condition_id in exp.condition_ids:
    records = evaluate_beta(exp, manifests, "Grad-CAM", condition_id, DEVICE)
    print(condition_id, "evaluated:", len(records))
    frequency_reports(exp)'''),
        ("markdown", """## 7. SHAP Expected Gradients
background train IDs เดิม 32 ภาพ; nsamples=200, batch_size=8, local_smoothing=0
บันทึก signed attribution ของทุก view/RGB ก่อนรวม ใช้ส่วนบวกประเมิน bbox
อธิบาย processed bands ไม่ใช่ raw-pixel gradients ผ่าน NumPy DWT/FFT
อาจมี ringing/สัญญาณกระจายจาก frequency filters จึงไม่ตีความเป็น lesion segmentation

Training + XAI ครบชุดบน RTX 4060 ประเมินเผื่อ 45–90 นาทีจากการวัด pipeline v2 ก่อนหน้า
ยังไม่ใช่ benchmark เต็มของ v3; เวลา Colab ขึ้นกับ GPU เซลล์นี้ resume รายภาพได้"""),
        ("code", '''for condition_id in exp.condition_ids:
    records = evaluate_beta(exp, manifests, "SHAP", condition_id, DEVICE)
    print(condition_id, "evaluated:", len(records))
    frequency_reports(exp)'''),
        ("markdown", """## 8. Classification / XAI และเกณฑ์ +10 จุดเปอร์เซ็นต์
ค่าความต่างต้องเทียบ baseline v3 ใน protocol นี้เท่านั้น: target = baseline_v3 + 0.10
0.10 บนสเกล 0–1 คือ 10 จุดเปอร์เซ็นต์ ไม่ใช่ relative improvement 10%
XAI ใช้ patient bootstrap 2,000 ครั้ง รวม overall/pure/mixed และ paired deltas/95% CI
classification delta เป็น descriptive; ไม่อ้าง significance จากคะแนนเพิ่มเพียงอย่างเดียว
test 100 ภาพเป็น pilot เดิม ไม่ใช่ official binary test เต็ม 12,081 ภาพ"""),
        ("code", '''cnn_table, xai_table, paired_deltas, status = frequency_reports(exp)
display(status); display(cnn_table); display(xai_table); display(paired_deltas)
base_acc = float(cnn_table.set_index("condition_id").loc["baseline", "accuracy"])
acceptance = cnn_table[["condition_id", "accuracy", "delta_accuracy_vs_baseline"]].copy()
acceptance["accuracy_target"] = base_acc + 0.10
acceptance["meets_plus_10_percentage_points"] = acceptance.delta_accuracy_vs_baseline >= (0.10 - 1e-12)
acceptance.to_csv(exp.output_root / "comparison/accuracy_target.csv", index=False)
display(acceptance)
for condition_id in exp.condition_ids:
    display(Image(filename=str(exp.results("CNN", condition_id) / "confusion_matrix.png")))
example = manifests["xai"].iloc[0]["image"].rsplit(".", 1)[0]
for criterion in ("Grad-CAM", "SHAP"):
    for condition_id in exp.condition_ids:
        print(criterion, condition_id)
        display(Image(filename=str(exp.results(criterion, condition_id) / "overlays" / (example + ".png"))))
print("Measured outputs:", exp.output_root)'''),
        ("markdown", """## 9. Export ผลจริง
รวม checkpoints/history/predictions/manifests/XAI/ตารางและกราฟ ไม่รวม image cache
ผลหลักเก็บใน Drive แม้ Colab disconnect ผล smoke test ไม่อยู่ใน ZIP นี้"""),
        ("code", '''export_path = exp.output_root.parent / "Notebook_3_beta_results.zip"
with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in exp.output_root.rglob("*"):
        if path.is_file() and "cache" not in path.relative_to(exp.output_root).parts:
            archive.write(path, path.relative_to(exp.output_root).as_posix())
print(export_path)
if IN_COLAB:
    from google.colab import files
    files.download(str(export_path))'''),
    ]
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_markdown_cell(text) if kind == "markdown" else nbf.v4.new_code_cell(text)
                      for kind, text in cells]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "colab": {"name": "frequency_head_only_e2.ipynb", "provenance": []}}
    nbf.validate(notebook)
    destination = ROOT / "frequency_head_only_e2.ipynb"
    nbf.write(notebook, destination)
    print(destination)
    return destination


def package():
    notebook = build()
    target = ROOT / "Notebook_3_beta_colab.zip"
    temporary = target.with_suffix(".tmp.zip")
    sources = [*sorted((ROOT / "shared").glob("*.py")), *sorted((ROOT / "shared").glob("*.json")),
               *sorted((ROOT / "source_run").rglob("*.json")), *sorted((ROOT / "source_run").rglob("*.csv")),
               ROOT / "README.md", ROOT / "build_notebook.py", ROOT / "validate_notebook_gpu.py",
               ROOT / "V2_SOURCE_HASHES.json", notebook,
               *sorted((ROOT / "tests").glob("*.py"))]
    if (ROOT / "VALIDATION.md").is_file():
        sources.append(ROOT / "VALIDATION.md")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, "Notebook_3/" + path.relative_to(ROOT).as_posix())
        archive.write(ROOT / "nih_cxr_subset.zip", "Notebook_3/nih_cxr_subset.zip", compress_type=zipfile.ZIP_STORED)
    temporary.replace(target)
    print(f"{target}: {target.stat().st_size / 1024**2:.1f} MiB")


if __name__ == "__main__":
    package()
