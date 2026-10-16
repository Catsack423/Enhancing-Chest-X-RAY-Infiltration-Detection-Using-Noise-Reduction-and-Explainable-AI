"""Build run_all_L4.ipynb from run_all.ipynb and the L4 source-only package.

Only the L4 strength conditions (median 9x9, Seminae DWT 0.35*max detail, DAE twice) are run.
Results go to runs/release_v4_L4 because adding conditions changes the config hash, so the
release_v4 output cannot be resumed. The finished DAE of release_v4 is imported when
tools/import_dae.py proves it identical. Never trains or packs data.
"""
from pathlib import Path
import ast
import zipfile
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Notebook_3_release_L4_colab.zip"
L4_CONDITIONS = ["median_L4", "clahe_dwt_strength_L4", "dae_clahe_strength_L4"]

INTRO = """# Notebook v4 L4 — Median / DWT / DAE strength L4 เท่านั้น

เหมือน `run_all.ipynb` (ข้อมูล, manifests, ResNet50, protocol v4) แต่ฝึกเฉพาะ 3 conditions ใหม่ และ **seed 42 seed เดียว**:

| Condition | ค่า L4 |
|---|---|
| `median_L4` | median kernel 9x9 (L1–L3 = 3/5/7) |
| `clahe_dwt_strength_L4` | DWT db1 1 ระดับ, soft threshold = 0.35 × max detail coefficient (จาก `Seminae_source_code` DWT L4) แล้ว CLAHE clip 2 |
| `dae_clahe_strength_L4` | DAE เดิมผ่าน 2 รอบ (alpha 1) แล้ว CLAHE clip 2 |

อัปโหลด `Notebook_3_release_L4_colab.zip` ไป `MyDrive/nih -xray/` (คนละไฟล์กับ `Notebook_3_release_colab.zip` ของ v4 ไม่ต้องลบของเดิม)
Dataset ZIP ใช้ชุดเดิมใน `MyDrive/nih -xray/data/v3_selected_zip/`

ผลลง `MyDrive/nih -xray/Notebook_3_release/runs/release_v4_L4/seed_N` แยกจาก `release_v4` (config เปลี่ยนจึง resume ของเดิมไม่ได้)
DAE คัดลอกจาก `release_v4/seed_N` เมื่อ config/code/data/versions ของ DAE ตรงกัน ถ้า seed นั้นยังไม่มี DAE ที่ฝึกครบจะฝึกใหม่
Report ของ run นี้มีเฉพาะ L4; conditions อื่นขึ้น `not_run` ให้เทียบกับตารางของ `release_v4`
"""

IMPORT_MD = """### ใช้ DAE จาก release_v4 (ถ้าตรวจผ่าน)
คัดลอก DAE ที่ train ครบแล้วจาก `release_v4/seed_N` เมื่อ config/code/data/versions ของ DAE ตรงกันทุกอย่าง ไม่ตรงจะหยุด ตั้ง `IMPORT_DAE_FROM = None` เพื่อ train ใหม่"""

GUARD = '''
import json
_configured = {c["id"] for c in json.loads((ROOT / "configs/methods_release_controlled.json").read_text(encoding="utf-8"))}
if not set(CONDITIONS) <= _configured:
    raise RuntimeError("Source is not the L4 package; upload Notebook_3_release_L4_colab.zip to MyDrive/nih -xray")
'''


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of: {old[:70]!r}")
    return text.replace(old, new)


def build():
    notebook = nbf.read(ROOT / "run_all.ipynb", as_version=4)
    cells = notebook.cells
    if any(cell.get("outputs") for cell in cells):
        raise ValueError("run_all.ipynb must not hold execution outputs")
    cells[0].source = INTRO
    setup = cells[1].source
    setup = replace_once(setup, 'CODE_ZIP = Path("/content/drive/MyDrive/nih -xray/Notebook_3_release_colab.zip")',
                         'CODE_ZIP = Path("/content/drive/MyDrive/nih -xray/Notebook_3_release_L4_colab.zip")')
    setup = replace_once(setup, 'raise FileNotFoundError("Upload Notebook_3_release_colab.zip to MyDrive/nih -xray, or upload the source folder as Notebook_3_release")',
                         'raise FileNotFoundError("Upload Notebook_3_release_L4_colab.zip to MyDrive/nih -xray")')
    setup = replace_once(setup, '    OUTPUT_ROOT = RELEASE_ROOT / "runs/release_v4"\n',
                         '    OUTPUT_ROOT = RELEASE_ROOT / "runs/release_v4_L4"\n')
    setup = replace_once(setup, '    OUTPUT_ROOT = ROOT / "runs/release_v4"\n',
                         '    OUTPUT_ROOT = ROOT / "runs/release_v4_L4"\n')
    setup = replace_once(setup, 'SEEDS = [42, 43, 44]\n', 'SEEDS = [42]  # L4: seed 42 only (config still lists 42/43/44)\n')
    setup = replace_once(setup, 'CONDITIONS = None  # None = all 12 controlled conditions\n',
                         f'CONDITIONS = {L4_CONDITIONS!r}  # L4 only; the other 12 are in runs/release_v4\n' + GUARD)
    cells[1].source = setup
    import_index = next(i for i, c in enumerate(cells) if c.cell_type == "code" and "IMPORT_DAE_FROM" in c.source)
    cells[import_index - 1].source = IMPORT_MD
    code = cells[import_index].source
    code = replace_once(code, "# Reuse a completed DAE from the cancelled release_full run instead of retraining (~1.9 h per seed).",
                        "# Reuse the completed DAE of release_v4 instead of retraining (~1.9 h per seed).")
    code = replace_once(code, 'IMPORT_DAE_FROM = Path(OUTPUT_ROOT).parent / "release_full_dae_b064_gpu"',
                        'IMPORT_DAE_FROM = Path(OUTPUT_ROOT).parent / "release_v4"')
    cells[import_index].source = code
    cnn_title = next(c for c in cells if c.cell_type == "markdown" and c.source.startswith("## 3. "))
    cnn_title.source = "## 3. ฝึก classifier เฉพาะ L4 (3 methods)"
    for cell in cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    notebook.metadata["colab"] = {"name": "run_all_L4.ipynb"}
    nbf.validate(notebook)
    path = ROOT / "run_all_L4.ipynb"
    nbf.write(notebook, path)
    return path


def package():
    """Same file selection as build_notebooks.package plus tools/, without rebuilding v4 notebooks."""
    allowed = {".py", ".json", ".csv", ".md", ".ipynb", ".txt"}
    folders = ("shared", "configs", "manifests", "tests", "tools", "DAE", "CNN", "Frequency", "Grad-CAM", "SHAP", "Analysis")
    candidates = [*ROOT.iterdir()]
    for name in folders:
        if (ROOT / name).is_dir():
            candidates.extend((ROOT / name).rglob("*"))
    sources = []
    for path in sorted(set(candidates)):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.suffix not in allowed or any(part in {".runtime", "runs", "__pycache__", ".ipynb_checkpoints"} for part in relative.parts):
            continue
        if path.suffix == ".ipynb" and any(cell.get("outputs") for cell in nbf.read(path, as_version=4).cells):
            raise ValueError(f"Source package must not include execution outputs: {relative}")
        sources.append(path)
    temporary = PACKAGE.with_suffix(".tmp.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, "Notebook_3_release/" + path.relative_to(ROOT).as_posix())
    temporary.replace(PACKAGE)
    print(f"L4 source-only package: {PACKAGE} ({PACKAGE.stat().st_size / 1024**2:.2f} MiB), {len(sources)} files")
    return PACKAGE


if __name__ == "__main__":
    print("Notebook:", build())
    package()
