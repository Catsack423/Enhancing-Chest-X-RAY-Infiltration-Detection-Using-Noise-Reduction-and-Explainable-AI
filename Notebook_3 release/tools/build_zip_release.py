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
    '    subprocess.check_call([sys.executable,',
    '''    import runpy
    ZIP_ROOT = Path("/content/drive/MyDrive/nih -xray/data/v3_selected_zip")
    stage_dataset = runpy.run_path(str(ROOT / "tools/stage_dataset.py"))["stage_dataset"]
    DATA_ROOT = stage_dataset(ROOT, ZIP_ROOT, Path("/content/cxr_v3_originals"))
    subprocess.check_call([sys.executable,''')
(ROOT / 'setup_zip_cell.txt').write_text(builder.SETUP, encoding='utf-8')
package = builder.package()
for name, payload in standalone.items():
    (ROOT / name).write_bytes(payload)
with zipfile.ZipFile(package, 'a', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted((ROOT / 'tools').glob('*.py')):
        archive.write(path, 'Notebook_3_release/tools/' + path.name)

# The intro comes from the older portable builder; update the generated artifacts.
for path in ROOT.rglob('*.ipynb'):
    if any(p in {'.runtime', 'runs'} for p in path.relative_to(ROOT).parts):
        continue
    if path.name in standalone:
        continue
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            cell.source = zip_intro(cell.source)
    nbformat.write(notebook, path)
with zipfile.ZipFile(package) as archive:
    names = archive.namelist()
tmp = package.with_suffix('.repack.zip')
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        archive.write(ROOT / name.removeprefix('Notebook_3_release/'), name)
tmp.replace(package)
print('ZIP-backed source package ready:', package)
