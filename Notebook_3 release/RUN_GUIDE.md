# ใช้ v3 release บน Colab

1. อัปโหลด `Notebook_3_release_colab.zip` ไป **MyDrive/nih -xray/** และเปิด `run_all.ipynb` บน Colab
2. อัปโหลด dataset ZIP 13 ส่วนและ `dataset_zip_index.json` ไป **MyDrive/nih -xray/data/v3_selected_zip/** ตาม `ZIP_DATA_GUIDE.md` เลือก GPU แล้วรัน setup เพื่อ copy/extract ZIP ลง VM; DATA_ROOT ต้องเป็น `/content/cxr_v3_originals`
3. หาก pip แจ้งให้ restart runtime เพราะเปลี่ยน library ให้ restart แล้วรัน setup ใหม่ก่อนเปิดrun
4. รันตรวจข้อมูล/cache: ยืนยัน source hashes/counts แล้วอ่านภาพต้นฉบับจาก VM เพื่อสร้าง working shards โดยไม่เปิดภาพผ่าน Drive ทีละไฟล์
5. รันDAE → CNN → Grad-CAM → SHAP → Reports ตามลำดับ; default3seeds ใช้fullcountsและ20epochs สลับไปรันnotebookแยกได้จากoutputเดียวกัน
6. เปิดผลใน `MyDrive/nih -xray/Notebook_3_release/runs/release_full/seed_42/` และseed43/44; รวม3seedsที่ `seed_summary/`

## Resume

### ใช้ GPU VRAM เพิ่มสำหรับ DAE

run_all / DAE notebook รุ่นล่าสุดมี GPU batch tuning ก่อนฝึก (USE_GPU_DAE_BATCH=True): ลองbatch16/32/48/64/80/96/... วัดforward/backward/Adamจริง เลือกbatchเร็วที่สุดที่อยู่ในงบ90%ของVRAM และเลือกbatchใหญ่กว่าเมื่อความเร็วห่างไม่เกิน5% มีAMPตามconfigเดิม และเผื่อOOM ผลprobeเป็นsyntheticGPUsteps ไม่ใช่เวลาของepochรวมโหลดข้อมูล
หยุดDAEเดิมก่อน ใช้ `dae_gpu_batch_cell.txt` หลังsetup/preparation หรือเปิดnotebookใหม่ หากGPUยังมีmodelเก่าค้างให้Restart sessionบนVMเดิมแล้วรันsetup/preparationใหม่ ไม่ลบcache/dataset
เปลี่ยนbatchแล้วจะเริ่มexperimentใหม่ใน `release_full_dae_bNNN_gpu/seed_42` เป็นต้น โดยเก็บoutputเดิม ใช้runsชุดใหม่กับCNN/XAI/reportsต่อทั้งหมด เก็บ20epochs/3seeds/ข้อมูล/architecture/learning rateเดิม ไม่resumecheckpointbatch16ปนกับbatchใหม่ CNNmicrobatchและSHAPsettingsยังเดิม จึงไม่กระทบeffectivebatch16ของCNN
หากกลับมาอีกsession ให้รันtuningและเปิดbatchprofileเดิมก่อนresume ถ้าbatchที่เลือกเปลี่ยนจะเปิดoutputคนละชื่อ ตั้งUSE_GPU_DAE_BATCH=Falseเพื่อใช้batch16แผนเดิม

### Log ระหว่างฝึก DAE / CNN

Notebook รุ่นล่าสุดมี observer ใน cells ฝึก: START ทันที, heartbeat ทุก10วินาที, epoch/batch/images/s/ETA ของ pass ปัจจุบัน, validation, ตรวจ checkpoint และบันทึก Drive โดยไม่เปลี่ยน production Python หรือ run fingerprint
ถ้าเปิด notebook เก่า ให้แทนทั้ง cell DAE ด้วย `train_dae_progress_cell.txt` หรือ CNN ด้วย `train_cnn_progress_cell.txt` ไม่ต้อง reset runtime/upload dataset ใหม่ ต้องหยุด cell เดิมก่อนแก้และรันใหม่: epoch ที่ยังไม่จบจะเริ่มใหม่จาก checkpoint ของ epoch ที่จบล่าสุด หรือรอ epoch ปัจจุบันจบก่อนเปลี่ยน observer
ETA เป็นประมาณการเฉพาะ pass ปัจจุบัน ไม่รวม validation/การบันทึกหรือทั้ง20epochs heartbeat แสดงสถานะล่าสุด ไม่ยืนยันว่าการเรียก Drive/CUDA ที่รอนานเสร็จแล้ว

Grad-CAM/SHAP มี START และ heartbeat พร้อม method, image, จำนวนภาพที่ตรวจผลผ่าน, คำนวณ explanation, preprocessing cache, publish/verify Drive outputs; ใช้ cells ใน `gradcam_progress_cell.txt` และ `shap_progress_cell.txt`
Reports แสดงตรวจผลแต่ละ method/criterion, patient-bootstrap CI, paired comparisons, สร้างกราฟและเผยแพร่ผล ใช้ `reports_progress_cell.txt`; รวม seeds ใช้ `seed_summary_progress_cell.txt` ทุก observer คืน hooks หลังจบ/error และไม่เปลี่ยน production Python/fingerprint

### NumPy `_blas_supports_fpe` AttributeError

หลัง pip เปลี่ยน NumPy อาจยังมี binary รุ่นเก่าอยู่ใน kernel เลือก **Runtime → Restart session** เพื่อ restart Python บน VM เดิม แล้วรัน setup และ preparation ใหม่ก่อน DAE ไม่เลือก disconnect/delete runtime และไม่ลบ dataset/cache ส่วนที่ตรวจครบใช้ต่อได้
Setup รุ่นใหม่ตรวจ `numpy.testing` หลัง pip ก่อน copy/extract ZIP และก่อน training ถ้ายังผิดหลัง restart ให้รัน `repair_numpy_cell.txt` (ติดตั้ง NumPy 2.4.6 เดิมใหม่) แล้ว restart session อีกครั้ง ไม่ downgrade dependencies ของงานเดิม

### Progress ระหว่างเตรียม cache

Notebook รุ่นเพิ่มlogแสดงphaseทันที และheartbeatทุก10วินาที พร้อมจำนวนภาพ/ภาพที่กำลังอ่าน/เวลาที่ใช้
ขั้นนี้ยังไม่ฝึกmodel จำนวนภาพuniqueทั้งหมดประมาณ70,012ภาพ; versionเดิมพิมพ์หลัง10shardsหรือ10,240ภาพจึงดูเงียบ
หากกำลังรันversionเดิม ให้หยุดเฉพาะcellเตรียมcache แล้วแทนcellนั้นด้วยข้อความใน`prepare_progress_cell.txt`และรันใหม่
ไม่ต้องresetruntimeหรือลบcache/output; verifiedshardsเดิมใช้ต่อได้ ส่วนshardที่ยังไม่ครบจะเตรียมใหม่
Observerอยู่ในnotebookเท่านั้น ไม่เปลี่ยนproductionPython/sourcefingerprintหรือphysicsของrun
สร้างnotebooks/ZIPพร้อมprogressซ้ำด้วย:

```python
from pathlib import Path
import build_notebooks as builder
import runpy
runpy.run_path("tools/build_zip_release.py", run_name="__main__")
```

หลังruntimeหลุด เปิดnotebookเดิม รันsetupและcacheใหม่ แล้วรันstageที่ต้องการต่อ
Checkpointsคืนmodel/optimizer/scaler/RNGและtrainingstage; epochที่ยังไม่จบเริ่มใหม่จากcheckpointล่าสุดที่จบแล้ว
XAIทำต่อเฉพาะภาพที่ยังไม่มีverifiedrecord; artifactsที่เขียนไม่ครบไม่ถือว่าcompleted
VMcacheสูญหายได้ แต่models/history/markers/manifestsอยู่บนDrive
ภาพต้นทางที่เปลี่ยนแม้metadataเหมือนเดิมจะถูกตรวจจากdurablepixel-fingerprintและปฏิเสธresumeเดิม

หากเปลี่ยนconfig/code/เวอร์ชันlibraries ให้ตั้งOUTPUT_ROOTใหม่ ไม่ลบหรือเขียนทับผลเก่าเพื่อให้รันผ่าน
การใช้Torch/torchvisionCUDAคนละเวอร์ชันกับcheckpointเดิมอาจต้องคืนenvironmentเดิมก่อนresume

## เลือกรันบางส่วน

ในsetup: `SEEDS=[42]` สำหรับเริ่มseedแรก และ `CONDITIONS=["baseline"]` สำหรับวัดbaselineก่อน
การเปลี่ยนSEEDS/CONDITIONSเลือกstageที่รันเท่านั้น ไม่เปลี่ยนรายชื่อข้อมูลหรือhyperparametersของrun
กลับเป็น`SEEDS=[42,43,44]`, `CONDITIONS=None` แล้วรันต่อได้โดยไม่ฝึกcompletedconditionsซ้ำ
ค่าSMOKE=Trueแยกoutputใน`smoke_runs` ใช้randomweights/tinydata ไม่ใช้รายงานAccuracyการทดลองจริง

Notebookย่อย: DAE/train_dae, CNN/train_all_methods, Grad-CAM/evaluate_gradcam, SHAP/evaluate_shap
Frequency/inspect_bandsอ่านภาพตัวอย่างและแสดงbands ไม่ฝึก; Analysis/compare_all_methodsอ่านผลและoptionalepochsnapshots
check_overfit.ipynbเป็นstandalone CPU ไม่ต้องsourceZIP/GPU; เปลี่ยนRUN_ROOTให้ตรงseedและรัน2cells

## Export

ใช้export_results_colab.ipynb ระบุRUN_ROOTของseedที่ต้องการ
Defaultไม่รวมweightsเพราะมีDrivecheckpointอยู่แล้ว เปิดINCLUDE_CHECKPOINTS=Trueหากต้องการarchiveweightsด้วย
Exportไม่รวมdatasetหรือinputcache; sourcepackageก็ไม่มีภาพ/weights/results

## Local

```powershell
python pipeline.py --data-root D:/kaggle_cache/datasets/nih-chest-xrays/data/versions/3 --stage prepare
python pipeline.py --data-root D:/kaggle_cache/datasets/nih-chest-xrays/data/versions/3 --stage all --all-seeds
```

รันจากNotebook_3 releaseด้วยPythonenvironmentCUDA; localdefaultsใช้`.runtime/cache`และ`runs/release_full`
Fulltrainingต้องGPU; CPUใช้unit tests/check_overfit/readingreportsได้
