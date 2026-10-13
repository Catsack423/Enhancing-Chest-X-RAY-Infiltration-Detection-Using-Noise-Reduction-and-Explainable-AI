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
dae_cell = (ROOT / 'train_dae_progress_cell.txt').read_text(encoding='utf-8')
cnn_cell = dae_cell.replace('TRAIN_STAGE = "dae"', 'TRAIN_STAGE = "cnn"')
cnn_path = ROOT / 'train_cnn_progress_cell.txt'
if not cnn_path.exists() or cnn_path.read_text(encoding='utf-8') != cnn_cell:
    tmp_cell = cnn_path.with_suffix('.txt.tmp')
    tmp_cell.write_text(cnn_cell, encoding='utf-8')
    tmp_cell.replace(cnn_path)
original_stage_code = builder.stage_code
activity_cell = (ROOT / 'activity_progress_cell.txt').read_text(encoding='utf-8')
activity_cells = {}
for activity_stage in ('gradcam', 'shap', 'reports'):
    cell = activity_cell.replace('PIPELINE_STAGE = "gradcam"', f'PIPELINE_STAGE = "{activity_stage}"')
    activity_cells[activity_stage] = cell
    cell_path = ROOT / f'{activity_stage}_progress_cell.txt'
    if not cell_path.exists() or cell_path.read_text(encoding='utf-8') != cell:
        cell_tmp = cell_path.with_suffix('.txt.tmp')
        cell_tmp.write_text(cell, encoding='utf-8')
        cell_tmp.replace(cell_path)
summary_code = activity_cell.split('\nPIPELINE_STAGE = ')[0] + '''
with activity_progress("all", "seed_summary"):
    from shared.reporting import summarize_seeds
    summarize_seeds(next(iter(runs.values()))[0].base_output_root)
'''
summary_path = ROOT / 'seed_summary_progress_cell.txt'
if not summary_path.exists() or summary_path.read_text(encoding='utf-8') != summary_code:
    summary_tmp = summary_path.with_suffix('.txt.tmp')
    summary_tmp.write_text(summary_code, encoding='utf-8')
    summary_tmp.replace(summary_path)
old_write_for_summary = builder.write_notebook
def write_with_summary(relative_path, cells):
    cells = [(kind, summary_code if kind == 'code' and text.startswith('from shared.reporting import summarize_seeds\n') else text) for kind, text in cells]
    if relative_path.as_posix() in ('run_all.ipynb', 'DAE/train_dae.ipynb'):
        gpu_code = 'USE_GPU_DAE_BATCH = False  # config pins DAE batch 64 (measured on L4); True re-probes into a separate output root\nif USE_GPU_DAE_BATCH:\n' + ''.join('    '+line+'\n' for line in (ROOT / 'dae_gpu_batch_cell.txt').read_text(encoding='utf-8').splitlines())
        insertion = next(i for i, (kind, text) in enumerate(cells) if kind == 'code' and 'TRAIN_STAGE = "dae"' in text)
        cells[insertion:insertion] = [('markdown', '### GPU batch tuning — เพิ่ม DAE batch จากผลวัด VRAM/ความเร็ว\nปิดไว้เป็นค่าเริ่มต้น: config ใช้ batch64 ที่วัดบน L4 แล้ว ตั้ง USE_GPU_DAE_BATCH=True เพื่อ probe ใหม่ ซึ่งจะเริ่ม output แยก'), ('code',gpu_code)]
    if relative_path.as_posix() in ('run_all.ipynb', 'DAE/train_dae.ipynb', 'CNN/train_all_methods.ipynb'):
        # Before the first training cell: CNN also trains a missing DAE.
        insertion = next(i for i, (kind, text) in enumerate(cells) if kind == 'code' and ('TRAIN_STAGE = "dae"' in text or 'TRAIN_STAGE = "cnn"' in text))
        cells[insertion:insertion] = [('markdown', '### ใช้ DAE เดิมจาก run ที่ยกเลิก (ถ้าตรวจผ่าน)\nคัดลอก DAE ที่ train ครบแล้วจาก `release_full_dae_b064_gpu` เมื่อ config/code/data/versions ของ DAE ตรงกันทุกอย่าง ไม่ตรงจะหยุด ตั้ง `IMPORT_DAE_FROM = None` เพื่อ train ใหม่'), ('code', import_dae_cell)]
    return old_write_for_summary(relative_path, cells)
import_dae_cell = (ROOT / 'import_dae_cell.txt').read_text(encoding='utf-8')
builder.write_notebook = write_with_summary
def logged_stage_code(stage):
    return {'dae': dae_cell, 'cnn': cnn_cell, **activity_cells}.get(stage) or original_stage_code(stage)
builder.stage_code = logged_stage_code
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
import runpy
runpy.run_path(str(ROOT / 'tools/build_parallel_notebook.py'))['build']()
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
