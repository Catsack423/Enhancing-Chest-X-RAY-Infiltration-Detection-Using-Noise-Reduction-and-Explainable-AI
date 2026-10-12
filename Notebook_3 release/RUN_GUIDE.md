# ใช้ v3 release บน Colab

1. อัปโหลด `Notebook_3_release_colab.zip` ไป **MyDrive/nih -xray/** และเปิด `run_all.ipynb` บน Colab
2. เลือก Runtime → Change runtime type → GPU แล้วรัน setup cell; mount Drive และตรวจว่า DATA_ROOT คือ `/content/drive/MyDrive/nih -xray/data/3`
3. หาก pip แจ้งให้ restart runtime เพราะเปลี่ยน library ให้ restart แล้วรัน setup ใหม่ก่อนเปิดrun
4. รันตรวจข้อมูล/cache: จะยืนยัน source hashes และ counts แล้วอ่านภาพที่ใช้เพื่อสร้างtemporaryshardsบนVM ไม่อัปโหลดภาพใหม่
5. รันDAE → CNN → Grad-CAM → SHAP → Reports ตามลำดับ; default3seeds ใช้fullcountsและ20epochs สลับไปรันnotebookแยกได้จากoutputเดียวกัน
6. เปิดผลใน `MyDrive/nih -xray/Notebook_3_release/runs/release_full/seed_42/` และseed43/44; รวม3seedsที่ `seed_summary/`

## Resume

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
