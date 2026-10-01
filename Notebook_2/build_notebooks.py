"""Regenerate the three thin notebooks; all scientific logic lives in shared/."""
from pathlib import Path
import json
import nbformat as nbf

ROOT = Path(__file__).resolve().parent

SETUP = '''from pathlib import Path
import sys

# Copy the complete Notebook_2 folder to Drive before opening a notebook.
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    drive.mount("/content/drive")

NOTEBOOK_2_DIR = Path("/content/drive/MyDrive/nih-chest-xrays/notebooks/Notebook_2") if IN_COLAB else next(
    p for p in [Path.cwd(), Path.cwd().parent, Path.cwd() / "Notebook_2", Path.cwd() / "notebooks/Notebook_2"]
    if (p / "shared/config.json").is_file()
)
# None uses the local NIH directory, or the Colab path in shared/config.json.
DATA_ROOT = None
OUTPUT_ROOT = None  # Uses output_subdirectory in shared/config.json; same in all notebooks.
SMOKE = False  # True uses separate smoke_runs/ output and tiny validation data.

if IN_COLAB:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "opencv-python-headless", "PyWavelets", "scikit-learn", "pandas", "matplotlib"])
sys.path.insert(0, str(NOTEBOOK_2_DIR))
# Optional local test dependencies; nothing here changes the machine's global packages.
if not IN_COLAB and (NOTEBOOK_2_DIR / ".runtime").is_dir():
    sys.path.insert(0, str(NOTEBOOK_2_DIR / ".runtime"))

import torch
from shared.experiment import Experiment, versions
from shared.data import prepare_manifests
from shared.reporting import refresh_reports

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if IN_COLAB and DEVICE != "cuda":
    raise RuntimeError("Select Runtime > Change runtime type > GPU before running the experiment")
exp = Experiment.open(NOTEBOOK_2_DIR, DATA_ROOT, OUTPUT_ROOT, smoke=SMOKE)
print("Device:", DEVICE, "| Mode:", exp.config["run_mode"])
print("Data:", exp.data_root, "| Results:", exp.output_root)
print(versions())
'''

DATA = '''manifests = prepare_manifests(exp)
for name, frame in manifests.items():
    print(name, "images:", len(frame), "patients:", frame.patient_id.nunique())
print("Patient/image leakage checks passed. Shared manifests are locked.")
display(manifests["xai"].risk_flag.value_counts().to_frame("n_images"))
'''


def build():
    config = json.loads((ROOT / "shared/config.json").read_text(encoding="utf-8"))
    definitions = [
        ("CNN", "cnn_comparison.ipynb", [
            ("markdown", "## 3. ฝึก DAE\nฝึกเฉพาะ train subset ใช้ validation noise คงที่ เลือก checkpoint จาก validation loss แล้วใช้ DAE ตัวเดียวกันทั้ง L1–L3"),
            ("code", "from shared.training import train_dae, train_condition\ndae_path = train_dae(exp, manifests, DEVICE)\nprint(dae_path)"),
            ("markdown", "## 4. ฝึกและประเมิน CNN ครบ 10 เงื่อนไข\nทุก run ใช้ seed, ImageNet initialization, train/validation/test และ hyperparameters เดียวกัน ค่าที่เสร็จแล้วจะโหลดกลับ; งานที่ค้างจะเริ่มต่อจาก epoch ล่าสุด"),
            ("code", "for condition_id in exp.condition_ids:\n    metrics = train_condition(exp, manifests, condition_id, DEVICE)\n    print(condition_id, {k: metrics[k] for k in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')})\n    refresh_reports(exp)"),
        ]),
        ("Grad-CAM", "gradcam_comparison.ipynb", [
            ("markdown", "## 3. Grad-CAM ของ Infiltration logit\nต้องรัน CNN ก่อน โหลด best checkpoint ของเงื่อนไขนั้น โดยไม่เลือกคลาสจาก argmax ประเมินทั้ง 123 ภาพรวมคำทำนายที่ผิด ขยาย heatmap กลับขนาดต้นฉบับก่อนเทียบ bbox"),
            ("code", "from shared.xai import run_xai\nrecords = run_xai(exp, manifests, 'Grad-CAM', DEVICE)\ndisplay(records.head())"),
        ]),
        ("SHAP", "shap_comparison.ipynb", [
            ("markdown", "## 3. SHAP ของ Infiltration logit\nใช้ shap.GradientExplainer (Expected Gradients) จริง ไม่ใช่ gradient×activation\nBackground มาจาก train คลาสละ 16 ภาพ ใช้ IDs เดียวกันทุกเงื่อนไข และ nsamples=200\nเก็บ signed attribution ราย channel ที่ input resolution และ signed sum ที่ขนาดต้นฉบับ ก่อนแยกค่าบวก normalize เป็น [0,1] เพื่อประเมิน bbox\nOverlay สีแดงเพิ่มคะแนน Infiltration สีน้ำเงินลดคะแนน Infiltration"),
            ("code", "from shared.xai import run_xai\nrecords = run_xai(exp, manifests, 'SHAP', DEVICE)\ndisplay(records.head())"),
        ]),
    ]
    for criterion, filename, additional in definitions:
        nb = nbf.v4.new_notebook()
        cells = [
            ("markdown", f"# Notebook 2 — {criterion}: ครบ 10 เงื่อนไข\n\nรันหลักบน Colab GPU และเก็บผลบน Google Drive อ่าน RUN_GUIDE.md ก่อนเริ่ม\n\nClassification: Infiltration-only vs No Finding-only; official test หลังกรองครบ 12,081 ภาพ\nXAI: bbox Infiltration 123 ภาพเดิม รวม mixed cases; คนไข้ไม่ซ้ำกับชุดฝึก\n\nBaseline, Median L1–L3, CLAHE+DWT L1–L3, DAE+CLAHE L1–L3; ค่าทั้งหมดอยู่ใน shared/config.json\n\nCLAHE+DWT เป็น joint presets (clip/depth/threshold); DAE+CLAHE เป็นการเปลี่ยน contrast หลัง DAE"),
            ("markdown", "## 1. สภาพแวดล้อมและเส้นทาง\nปรับ NOTEBOOK_2_DIR และ DATA_ROOT ให้ตรงกับตำแหน่งบน Drive ใช้ OUTPUT_ROOT เดียวกันทั้งสาม notebook เลือก SMOKE ค่าเดียวกันเสมอ"),
            ("code", SETUP),
            ("markdown", f"## 2. ตรวจข้อมูลและใช้ manifests ร่วมกัน\nสุ่ม train {config['data']['train_per_class']:,}/คลาส และ validation {config['data']['validation_per_class']:,}/คลาสด้วย seed {config['seed']} แบบแยกคนไข้ ไม่มีการย้ายภาพระหว่าง official train_val/test"),
            ("code", DATA),
            *additional,
            ("markdown", "## ผลเปรียบเทียบและ README\nตารางรวมและรูปเปรียบเทียบอยู่ใน comparison/ ของแต่ละเกณฑ์ CI ของ XAI ใช้ patient bootstrap 2,000 รอบ seed เดียวกัน รายงาน Overall/Pure/Mixed-high-risk/Mixed-low-risk; ผลที่ยังไม่รันระบุ not_run"),
            ("code", "status = refresh_reports(exp)\ndisplay(status)\nfrom IPython.display import Markdown, display\ndisplay(Markdown((exp.output_root / 'README.md').read_text(encoding='utf-8')))"),
        ]
        nb.cells = [nbf.v4.new_markdown_cell(text) if kind == "markdown" else nbf.v4.new_code_cell(text) for kind, text in cells]
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}, "colab": {"name": filename, "provenance": []}}
        destination = ROOT / criterion / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbf.validate(nb)
        nbf.write(nb, destination)
        print(destination)


if __name__ == "__main__":
    build()
