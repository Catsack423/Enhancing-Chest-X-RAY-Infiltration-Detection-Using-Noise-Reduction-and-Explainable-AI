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
"""

SETTINGS = '''SESSION_SEED = "auto"  # "auto" หรือระบุ 42 / 43 / 44 ไม่ซ้ำกันในแต่ละ session
ALL_SEEDS = [42, 43, 44]
FORCE_CLAIM = False  # True เฉพาะเมื่อแน่ใจว่า session เดิมของ seed นี้หยุดแล้ว
'''

CLAIM = '''import runpy
parallel = runpy.run_path(str(Path(ROOT) / "tools/parallel_session.py"))
if SESSION_SEED == "auto":
    SESSION_SEED = parallel["next_seed"](ROOT, OUTPUT_ROOT, ALL_SEEDS)
    if SESSION_SEED is None:
        raise RuntimeError("ทุก seed มี report แล้วหรือมี session อื่นถืออยู่ ไม่มีงานเหลือให้ session นี้")
    print("auto seed:", SESSION_SEED, flush=True)
SEEDS = [SESSION_SEED]
parallel["preflight"](OUTPUT_ROOT, SESSION_SEED, ALL_SEEDS)
claim = parallel["SeedClaim"](OUTPUT_ROOT, SESSION_SEED).acquire(force=FORCE_CLAIM)
parallel["print_status"](ROOT, OUTPUT_ROOT, ALL_SEEDS)
'''

STATUS = '''parallel["print_status"](ROOT, OUTPUT_ROOT, ALL_SEEDS)
'''


def build():
    guarded = lambda cell: 'claim.check()\n' + cell
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
        ('code', guarded(text('import_dae_cell.txt'))),
        ('markdown', '## 4. DAE ของ seed นี้ (ข้ามถ้า import แล้ว)'), ('code', guarded(text('train_dae_progress_cell.txt'))),
        ('markdown', '## 5. CNN ทั้ง 12 methods (early stopping)'), ('code', guarded(text('train_cnn_progress_cell.txt'))),
        ('markdown', '## 6. Grad-CAM'), ('code', guarded(text('gradcam_progress_cell.txt'))),
        ('markdown', '## 7. SHAP'), ('code', guarded(text('shap_progress_cell.txt'))),
        ('markdown', '## 8. Report ของ seed นี้'), ('code', guarded(text('reports_progress_cell.txt'))),
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
