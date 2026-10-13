"""Generate portable notebooks and a source-only package. Never train or pack data."""
from pathlib import Path
import ast
import zipfile
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
SETUP = '''from pathlib import Path
import importlib, importlib.util, sys, zipfile, shutil, subprocess

try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DATA_ROOT = Path("/content/drive/MyDrive/nih -xray/data/3")
    RELEASE_ROOT = Path("/content/drive/MyDrive/nih -xray/Notebook_3_release")
    CODE_ZIP = Path("/content/drive/MyDrive/nih -xray/Notebook_3_release_colab.zip")
    if CODE_ZIP.is_file():
        ROOT = Path("/content/Notebook_3_release")
        with zipfile.ZipFile(CODE_ZIP) as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "Notebook_3_release":
                    raise ValueError("Invalid source package path")
                if member.is_dir():
                    continue
                target = Path("/content").joinpath(*relative.parts).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise ValueError("Invalid source package destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
    elif (RELEASE_ROOT / "pipeline.py").is_file():
        ROOT = RELEASE_ROOT
    else:
        raise FileNotFoundError("Upload Notebook_3_release_colab.zip to MyDrive/nih -xray, or upload the source folder as Notebook_3_release")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    OUTPUT_ROOT = RELEASE_ROOT / "runs/release_v4"
    CACHE_ROOT = Path("/content/cxr_v3_cache")
else:
    candidates = [Path.cwd(), *Path.cwd().parents, Path.cwd() / "Notebook_3 release"]
    ROOT = next((p for p in candidates if (p / "configs/release_full.json").is_file()), None)
    if ROOT is None:
        raise FileNotFoundError("Open from Notebook_3 release or its repository directory")
    DATA_ROOT = None  # locate local data/versions/3 without changing the frozen manifests
    OUTPUT_ROOT = ROOT / "runs/release_v4"
    CACHE_ROOT = ROOT / ".runtime/cache"

sys.path.insert(0, str(ROOT))
for name in tuple(sys.modules):
    if name == "shared" or name.startswith("shared.") or name in ("pipeline", "check_overfit", "export_results_colab"):
        del sys.modules[name]
importlib.invalidate_caches()
import torch
from pipeline import open_run, run_stages
SEEDS = [42, 43, 44]
SMOKE = False  # True = isolated tiny validation run with random initialization; not research scores
CONDITIONS = None  # None = all 12 controlled conditions
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Data:", DATA_ROOT, "| Results:", OUTPUT_ROOT, "| Device:", DEVICE)
'''
OPEN = '''runs = {}
for seed in SEEDS:
    exp, manifests = open_run(DATA_ROOT, OUTPUT_ROOT, CACHE_ROOT, seed=seed, smoke=SMOKE)
    runs[seed] = (exp, manifests)
    print(seed, {name: len(frame) for name, frame in manifests.items()}, exp.output_root)
'''


def stage_code(stage):
    return f'''for seed, (exp, manifests) in runs.items():
    run_stages(exp, manifests, stages=["{stage}"], condition_ids=CONDITIONS, device=DEVICE)
'''


def write_notebook(relative_path, cells):
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell(text) if kind == "code" else nbf.v4.new_markdown_cell(text) for kind, text in cells]
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}, "colab": {"name": relative_path.name}}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nbf.validate(notebook)
    path = ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


def build():
    intro = """# Notebook v3 release — Full data / 12 methods

เลือก GPU บน Colab แล้วรันตามลำดับ อัปโหลด `Notebook_3_release_colab.zip` (โค้ดเท่านั้น)
ไป `MyDrive/nih -xray/` ข้อมูลอ่านจาก `MyDrive/nih -xray/data/3` ที่มีอยู่ ไม่อัปโหลดรูปซ้ำ

Train 46,108 / Validation 11,719 / Test 12,081; XAI123ภาพ ทุก method ใช้ manifests เดียวกัน
ค่าเริ่มต้น3seeds/20epochs รวม36CNN+3DAE; ไม่ใช่ notebook ที่รันเต็มได้ในไม่กี่นาที
Results/checkpoints ลง Drive; input cache อยู่ VM และอาจต้องสร้างใหม่เมื่อเปลี่ยน runtime
ทำซ้ำ cell เพื่อ resume; ถ้า config/code/dependency versions เปลี่ยน ใช้ output directory ใหม่
"""
    summary = '''from shared.reporting import summarize_seeds
summarize_seeds(next(iter(runs.values()))[0].base_output_root)
'''
    main = [("markdown", intro), ("code", SETUP), ("markdown", "## 1. ตรวจข้อมูลและเตรียม working cache บน VM"), ("code", OPEN)]
    for stage, title in [("dae", "2. ฝึก DAE จาก train เท่านั้น"), ("cnn", "3. ฝึก classifier ทั้ง12methods"), ("gradcam", "4. Grad-CAM"), ("shap", "5. SHAP"), ("reports", "6. ตาราง กราฟ CI และ paired comparisons")]:
        main += [("markdown", "## " + title), ("code", stage_code(stage))]
    main += [("markdown", "## 7. รวมคะแนนแต่ละ seed (mean±SD แยกจาก patient CI)"), ("code", summary)]
    paths = [write_notebook(Path("run_all.ipynb"), main)]
    for directory, name, stage in [("DAE", "train_dae", "dae"), ("CNN", "train_all_methods", "cnn"), ("Grad-CAM", "evaluate_gradcam", "gradcam"), ("SHAP", "evaluate_shap", "shap")]:
        paths.append(write_notebook(Path(directory) / (name + ".ipynb"), [("markdown", intro), ("code", SETUP), ("code", OPEN), ("code", stage_code(stage))]))
    report_open = '''runs = {}
for seed in SEEDS:
    exp, manifests = open_run(DATA_ROOT, OUTPUT_ROOT, CACHE_ROOT, seed=seed, smoke=SMOKE, prepare_cache=False)
    runs[seed] = (exp, manifests)
'''
    paths.append(write_notebook(Path("Analysis/compare_all_methods.ipynb"), [("markdown", intro), ("code", SETUP), ("code", report_open), ("code", stage_code("reports")), ("code", summary), ("code", '''from shared.epoch_comparison import compare_epochs
for exp, manifests in runs.values():
    compare_epochs(exp, manifests)
''')]))
    frequency = '''import numpy as np
import matplotlib.pyplot as plt
from shared.data import resolve_image_path
from shared.denoising import read_gray, working_image
from shared.frequency import dwt_components, dft_components
exp, manifests = open_run(DATA_ROOT, OUTPUT_ROOT, CACHE_ROOT, seed=42, smoke=SMOKE, prepare_cache=False)
row = manifests["train"].iloc[0].to_dict()
x = working_image(read_gray(resolve_image_path(exp, row)), exp.config).astype(np.float32) / 255
dwt, dft = dwt_components(x), dft_components(x)
views = [("Raw", x), *zip(["A4", "D1", "D2", "D3", "D4"], dwt), *zip(["DFT low", "DFT mid", "DFT high"], dft)]
fig, axes = plt.subplots(3, 3, figsize=(13, 11))
for ax, (name, band) in zip(axes.flat, views):
    ax.imshow(band, cmap="gray" if name in ("Raw", "A4", "DFT low") else "RdBu_r")
    ax.set_title(name); ax.axis("off")
fig.suptitle("Spatial frequency bands — display scaling only; CNN uses the fixed signed encoding")
fig.tight_layout(); plt.show()
assert np.allclose(dwt.sum(0), x, atol=1e-5) and np.allclose(dft.sum(0), x, atol=1e-5)
'''
    paths.append(write_notebook(Path("Frequency/inspect_bands.ipynb"), [("markdown", "# Inspect reconstructed DWT/DFT bands — ไม่ฝึกโมเดล"), ("code", SETUP), ("code", frequency)]))
    overfit_source = (ROOT / "check_overfit.py").read_text(encoding="utf-8")
    overfit_setup = '''from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
    RUN_ROOT = Path("/content/drive/MyDrive/nih -xray/Notebook_3_release/runs/release_v4/seed_42")
except ImportError:
    RUN_ROOT = Path.cwd() / "runs/release_v4/seed_42"
'''
    paths.append(write_notebook(Path("check_overfit.ipynb"), [("markdown", "# Check overfit — CPU ได้ ไม่โหลดโมเดล ไม่ฝึกซ้ำ\nแก้ RUN_ROOT ให้ตรง seed/output; ต้องมี history อย่างน้อยหนึ่ง method"), ("code", overfit_setup), ("code", overfit_source + "\nplot_overfit(RUN_ROOT)\n")]))
    export_source = (ROOT / "export_results_colab.py").read_text().split('if __name__ == "__main__":')[0]
    export_setup = overfit_setup + "\nINCLUDE_CHECKPOINTS = False  # เก็บ weights บน Drive แล้ว; เปิดได้หากต้องการ export ด้วย\n"
    paths.append(write_notebook(Path("export_results_colab.ipynb"), [("markdown", "# Export saved seed results — ไม่รวม dataset และ input cache"), ("code", export_setup), ("code", export_source + "\nexport_results(RUN_ROOT, include_checkpoints=INCLUDE_CHECKPOINTS)\n")]))
    return paths


def package():
    build()
    target = ROOT / "Notebook_3_release_colab.zip"
    temporary = target.with_suffix(".tmp.zip")
    allowed = {".py", ".json", ".csv", ".md", ".ipynb", ".txt"}
    sources = []
    folders = ("shared", "configs", "manifests", "tests", "DAE", "CNN", "Frequency", "Grad-CAM", "SHAP", "Analysis")
    candidates = [*ROOT.iterdir()]
    for name in folders:
        directory = ROOT / name
        if directory.is_dir():
            candidates.extend(directory.rglob("*"))
    for path in sorted(set(candidates)):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.suffix not in allowed or any(part in {".runtime", "runs", "__pycache__", ".ipynb_checkpoints"} for part in relative.parts):
            continue
        if path.suffix == ".ipynb":
            notebook = nbf.read(path, as_version=4)
            if any(cell.get("outputs") for cell in notebook.cells):
                raise ValueError("Source package must not include execution outputs")
        sources.append(path)
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, "Notebook_3_release/" + path.relative_to(ROOT).as_posix())
    temporary.replace(target)
    print(f"Source-only package: {target} ({target.stat().st_size / 1024**2:.2f} MiB), {len(sources)} files; no dataset or weights")
    return target


if __name__ == "__main__":
    package()
