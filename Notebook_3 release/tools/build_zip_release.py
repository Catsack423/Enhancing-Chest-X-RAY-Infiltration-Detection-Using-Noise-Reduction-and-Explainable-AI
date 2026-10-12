"""Generate ZIP-backed notebooks without changing scientific source fingerprints."""
from pathlib import Path
import sys
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_notebooks as builder

def zip_intro(text):
    return text.replace(
        'ไป `MyDrive/nih -xray/` ข้อมูลอ่านจาก `MyDrive/nih -xray/data/3` ที่มีอยู่ ไม่อัปโหลดรูปซ้ำ',
        'ไป `MyDrive/nih -xray/` และอัปโหลด dataset ZIP 13 ส่วนพร้อม index ไป `MyDrive/nih -xray/data/v3_selected_zip/`\n'
        'Setup คัดลอก ZIP จาก Drive แล้วแตกภาพต้นฉบับลง `/content/cxr_v3_originals` ก่อนสร้าง cache; ดู `ZIP_DATA_GUIDE.md`')

original_write = builder.write_notebook
def stable_write(relative_path, cells):
    cells = [(kind, zip_intro(text) if kind == 'markdown' else text) for kind, text in cells]
    path = ROOT / relative_path
    if path.exists():
        existing = nbformat.read(path, as_version=4)
        wanted = [('code' if kind == 'code' else 'markdown', text) for kind, text in cells]
        if [(c.cell_type, c.source) for c in existing.cells] == wanted and not any(c.get('outputs') for c in existing.cells):
            return path
    return original_write(relative_path, cells)
builder.write_notebook = stable_write

standalone = {name: (ROOT / name).read_bytes() for name in
              ('check_overfit.ipynb', 'export_results_colab.ipynb') if (ROOT / name).exists()}
builder.OPEN = (ROOT / 'prepare_progress_cell.txt').read_text(encoding='utf-8')
builder.SETUP = builder.SETUP.replace(
    '    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])',
    '''    import runpy
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")])
    runpy.run_path(str(ROOT / "tools/check_environment.py"))["check_numpy"]()
    ZIP_ROOT = Path("/content/drive/MyDrive/nih -xray/data/v3_selected_zip")
    stage_dataset = runpy.run_path(str(ROOT / "tools/stage_dataset.py"))["stage_dataset"]
    DATA_ROOT = stage_dataset(ROOT, ZIP_ROOT, Path("/content/cxr_v3_originals"))
''')
setup_file = ROOT / 'setup_zip_cell.txt'
if not setup_file.exists() or setup_file.read_text(encoding='utf-8') != builder.SETUP:
    setup_tmp = setup_file.with_suffix('.txt.tmp')
    setup_tmp.write_text(builder.SETUP, encoding='utf-8')
    setup_tmp.replace(setup_file)
package = builder.package()
for name, payload in standalone.items():
    (ROOT / name).write_bytes(payload)
with zipfile.ZipFile(package, 'a', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((ROOT / 'tools').glob('*.py')):
        archive.write(path, 'Notebook_3_release/tools/' + path.name)

# stable_write already normalizes the introduction; avoid rewriting open notebooks.
with zipfile.ZipFile(package) as archive:
    names = archive.namelist()
tmp = package.with_suffix('.repack.zip')
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        archive.write(ROOT / name.removeprefix('Notebook_3_release/'), name)
tmp.replace(package)
print('ZIP-backed source package ready:', package)
