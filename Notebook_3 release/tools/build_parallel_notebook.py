"""Build run_parallel.ipynb: each Colab session owns one seed of the shared release run.

Reuses the existing setup/progress cells verbatim so observers and fingerprints stay identical.
"""
from pathlib import Path
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def text(name):
    return (ROOT / name).read_text(encoding='utf-8')


INTRO = """# Notebook v3 release — Parallel sessions (1 seed ต่อ 1 session)

เปิด notebook นี้ใน Colab ได้หลาย session พร้อมกัน (เช่น 2 session) แต่ละ session ถือ seed ละหนึ่งตัว
ทุก session เขียนลง run เดียวกัน `runs/release_v4/seed_<n>/` จึงไม่มีไฟล์ที่เขียนชนกัน
ยกเว้น claim ใน `parallel_claims/` และ `seed_summary/` ซึ่งรันครั้งเดียวหลังทุก seed เสร็จ

- `SESSION_SEED = "auto"` เลือก seed แรกที่ยังไม่มี report และไม่มี session อื่นถืออยู่ 2 session จะได้ 42 และ 43
  session ที่เสร็จก่อนให้ Run all ใหม่ จะได้ seed 44 ต่อ
- DAE batch 64 อยู่ใน config แล้ว ต้องใช้ GPU ที่มี VRAM >= 20 GiB ถ้า DAE ของ seed นั้นยังไม่เสร็จ (L4 หรือ A100) ควรใช้ GPU รุ่นเดียวกันทุก session
- Package versions ต้องตรงกับ seed ที่มีอยู่แล้ว ไม่งั้น `summarize_seeds` จะปฏิเสธ preflight จะหยุดก่อน train
- Claim ส่ง heartbeat ไป Drive ทุก 1 นาที ถ้าอีก session ถือ seed เดียวกันอยู่จะถูกปฏิเสธ หลัง runtime ตาย claim หมดอายุใน 15 นาที
- Resume: เปิดซ้ำด้วย seed เดิมแล้วรันทุก cell ใหม่ งานที่ complete แล้วจะข้าม
- ช่วยกันทำ seed เดียว: session หลักถือ seed ตามปกติ session ช่วยตั้ง `SESSION_SEED=42`, `SHARE_SEED=True`, `REVERSE_ORDER=True`
  CNN/Grad-CAM/SHAP แบ่งกันทีละ condition ผ่าน lease ใน `parallel_claims/seed_<n>/` session ช่วยไม่ train DAE และไม่ทำ report
"""

SETTINGS = '''SESSION_SEED = "auto"  # "auto" หรือระบุ 42 / 43 / 44 ไม่ซ้ำกันในแต่ละ session
ALL_SEEDS = [42, 43, 44]
FORCE_CLAIM = False  # True เฉพาะเมื่อแน่ใจว่า session เดิมของ seed นี้หยุดแล้ว
SHARE_SEED = False     # True = ช่วย session อื่นทำ seed เดียวกัน (ต้องระบุ SESSION_SEED)
REVERSE_ORDER = False  # True ใน session ช่วย: เริ่มจาก condition ท้ายสุด
'''

CLAIM = '''import runpy
parallel = runpy.run_path(str(Path(ROOT) / "tools/parallel_session.py"))
if SESSION_SEED == "auto" and SHARE_SEED:
    raise ValueError("SHARE_SEED=True ต้องระบุ SESSION_SEED ของ session หลัก เช่น 42")
if SESSION_SEED == "auto":
    SESSION_SEED = parallel["next_seed"](ROOT, OUTPUT_ROOT, ALL_SEEDS)
    if SESSION_SEED is None:
        raise RuntimeError("ทุก seed มี report แล้วหรือมี session อื่นถืออยู่ ไม่มีงานเหลือให้ session นี้")
    print("auto seed:", SESSION_SEED, flush=True)
SEEDS = [SESSION_SEED]
parallel["preflight"](OUTPUT_ROOT, SESSION_SEED, ALL_SEEDS)
claim = parallel["NoClaim"]() if SHARE_SEED else parallel["SeedClaim"](OUTPUT_ROOT, SESSION_SEED).acquire(force=FORCE_CLAIM)
parallel["print_status"](ROOT, OUTPUT_ROOT, ALL_SEEDS)
'''

STATUS = '''parallel["print_status"](ROOT, OUTPUT_ROOT, ALL_SEEDS)
'''

HELPER_DAE_GUARD = '''if SHARE_SEED and not (runs[SESSION_SEED][0].artifacts / "dae_checkpoint/complete.json").is_file():
    raise RuntimeError("session ช่วยไม่ train DAE: รอให้ session หลักทำ DAE ของ seed นี้เสร็จก่อน")
'''

REPORT_GUARD = '''pending = parallel["pending_conditions"](runs[SESSION_SEED][0])
if SHARE_SEED:
    print("session ช่วย: report ทำใน session หลักของ seed นี้", flush=True)
elif any(pending.values()):
    print("ยังไม่ทำ report: condition ที่ยังไม่เสร็จ", pending, "- รัน cell นี้อีกครั้งเมื่อครบ", flush=True)
else:
'''


def queue_cell(cell_text, marker, stage, observer):
    """Replace the run_stages loop of a progress cell with the shared condition queue."""
    head = cell_text.split(marker, 1)[0]
    return head + marker + f'''
for seed, (exp, manifests) in runs.items():
    with {observer}(seed, {marker.split(" = ")[0]}):
        parallel["run_queue"](exp, manifests, OUTPUT_ROOT, "{stage}", device=DEVICE, reverse=REVERSE_ORDER)
'''


def write_text(name, content):
    path = ROOT / name
    if not path.exists() or path.read_bytes().decode('utf-8') != content:
        path.write_bytes(content.encode('utf-8'))


HANDOFF = '''# วางใน session ที่รัน run_parallel.ipynb (v4) อยู่แล้ว แทน cell 5: หยุด CNN cell เดิมก่อน แล้วรัน cell นี้
# โหลด tools/parallel_session.py รุ่นที่มี condition queue จาก ZIP บน Drive (tools ไม่อยู่ใน source fingerprint)
import runpy, zipfile
with zipfile.ZipFile(CODE_ZIP) as archive:
    (Path(ROOT) / "tools/parallel_session.py").write_bytes(archive.read("Notebook_3_release/tools/parallel_session.py"))
parallel = runpy.run_path(str(Path(ROOT) / "tools/parallel_session.py"))
SHARE_SEED, REVERSE_ORDER = False, False
'''


def build():
    guarded = lambda cell: 'claim.check()\n' + cell
    cnn = queue_cell(text('train_cnn_progress_cell.txt'), 'TRAIN_STAGE = "cnn"', 'cnn', 'training_progress')
    gradcam = queue_cell(text('gradcam_progress_cell.txt'), 'PIPELINE_STAGE = "gradcam"', 'gradcam', 'activity_progress')
    shap = queue_cell(text('shap_progress_cell.txt'), 'PIPELINE_STAGE = "shap"', 'shap', 'activity_progress')
    report = text('reports_progress_cell.txt')
    head, tail = report.split('\nPIPELINE_STAGE = "reports"', 1)
    report = head + '\n' + REPORT_GUARD + '    PIPELINE_STAGE = "reports"' + tail.replace('\n', '\n    ').rstrip(' ')
    # Paste-in cells for a session already running the v4 notebook (no new VM needed).
    write_text('queue_cnn_cell.txt', HANDOFF + cnn)
    write_text('queue_gradcam_cell.txt', gradcam)
    write_text('queue_shap_cell.txt', shap)
    summary = text('seed_summary_progress_cell.txt')
    head, tail = summary.split('\nwith activity_progress("all", "seed_summary"):', 1)
    summary = head + '''
waiting = parallel["missing_reports"](OUTPUT_ROOT, ALL_SEEDS)
if waiting:
    print("ยังไม่รวม seeds: report ยังไม่เสร็จสำหรับ", waiting, "- รัน cell นี้ใน session ที่เสร็จหลังสุด", flush=True)
else:
    with activity_progress("all", "seed_summary"):''' + tail.replace('\n    ', '\n        ')
    cells = [
        ('markdown', INTRO),
        ('markdown', '## 0. ตั้งค่า session'), ('code', SETTINGS),
        ('markdown', '## 1. Setup: ติดตั้ง packages, stage dataset ZIP ลง VM'), ('code', text('setup_zip_cell.txt')),
        ('markdown', '## 2. เลือก seed + preflight + claim'), ('code', CLAIM),
        ('markdown', '## 3. ตรวจข้อมูลและเตรียม working cache บน VM'), ('code', guarded(text('prepare_progress_cell.txt'))),
        ('markdown', '## 4a. ใช้ DAE เดิมจาก run ที่ยกเลิก (ถ้าตรวจผ่าน)\nคัดลอก DAE ที่ train ครบแล้วจาก `release_full_dae_b064_gpu` เมื่อ config/code/data/versions ของ DAE ตรงกันทุกอย่าง ไม่ตรงจะหยุด ตั้ง `IMPORT_DAE_FROM = None` เพื่อ train ใหม่'),
        ('code', guarded(HELPER_DAE_GUARD + text('import_dae_cell.txt'))),
        ('markdown', '## 4. DAE ของ seed นี้ (ข้ามถ้า import แล้ว)'), ('code', guarded(HELPER_DAE_GUARD + text('train_dae_progress_cell.txt'))),
        ('markdown', '## 5. CNN ทั้ง 12 methods (early stopping, แบ่ง condition กับ session อื่นได้)'), ('code', guarded(cnn)),
        ('markdown', '## 6. Grad-CAM'), ('code', guarded(gradcam)),
        ('markdown', '## 7. SHAP'), ('code', guarded(shap)),
        ('markdown', '## 8. Report ของ seed นี้ (session หลัก เมื่อทุก condition เสร็จ)'), ('code', guarded(report)),
        ('markdown', '## 9. ปล่อย claim และดูสถานะทุก seed\nถ้ายังมี seed เหลือ ให้ Runtime → Run all ใหม่ใน session นี้ `auto` จะเลือก seed ถัดไป'),
        ('code', 'claim.release()\n' + STATUS),
        ('markdown', '## 10. รวม 3 seeds (รันใน session ที่เสร็จหลังสุด)'), ('code', summary),
    ]
    notebook = nbf.v4.new_notebook()
    notebook.cells = [nbf.v4.new_code_cell(s) if kind == 'code' else nbf.v4.new_markdown_cell(s) for kind, s in cells]
    notebook.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                         # Colab opens a fresh copy on an L4 instead of the default CPU runtime.
                         'colab': {'name': 'run_parallel.ipynb', 'gpuType': 'L4', 'machine_shape': 'hm'}}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
    nbf.validate(notebook)
    path = ROOT / 'run_parallel.ipynb'
    nbf.write(notebook, path)
    return path


if __name__ == '__main__':
    print('Built', build())
