# Notebook v3 release — Full data / 12 methods

เปิด [run_all.ipynb](run_all.ipynb) บน Colab เลือก GPU แล้วรันตามลำดับ
อัปโหลด **Notebook_3_release_colab.zip** ไป `MyDrive/nih -xray/` เพียงไฟล์เดียวสำหรับ source package
ข้อมูลอ่านจาก **`MyDrive/nih -xray/data/3`** ที่มีอยู่ ไม่สร้าง/อัปโหลด dataset ZIP ใหม่
ชื่อ `nih -xray` มีช่องว่างก่อน `-`; แก้ DATA_ROOT ใน setup cell ได้หากชื่อจริงต่างจากรูป

| ชุด | No Finding | Infiltration-only | รวม |
|---|---:|---:|---:|
| Train | 40,371 | 5,737 | **46,108** |
| Validation | 10,129 | 1,590 | **11,719** |
| Test | 9,861 | 2,220 | **12,081** |

ใช้ official NIH lists และ validation-patient partition เดิมของ v2/class200; เก็บ400trainเดิมครบ ไม่ re-split
Positive classification เป็น `Infiltration` อย่างเดียว Negative เป็น `No Finding` อย่างเดียว
XAI123ภาพ/115คนไข้เดิม: pure19, mixed-high-risk64, mixed-low-risk40 รวมคำทำนายผิดและ empty maps
SHAP backgrounds32ภาพเดิมจาก train; pure19XAIภาพอยู่ใน classificationtestด้วย

## Methods

| กลุ่ม | Conditions |
|---|---|
| Baseline | raw1view |
| Median | kernels3/5/7 |
| DWT → CLAHE controlled strength | db1/depth3/periodization, threshold0.5/1/2, CLAHEclip2 |
| DAE → CLAHE controlled strength | α0.25/0.5/1, CLAHEclip2, DAEcheckpointเดียวกันต่อseed |
| DWT fusion | A4+D1–D4, 5views ไม่ threshold/CLAHE |
| DFT fusion | low/mid/high radial bands, 3views |

รวม12conditions ใช้ ResNet50 ImageNet V2 ตัวเดียวกันต่อmodel; เฉลี่ยfeatures2,048ค่าก่อนDropout0.3+Linear
20epochs: head-only1–3, layer4+head4–20, freezeearlylayersและBatchNormstatisticsทั้งหมด
AdamW, headlr1e-4/layer4lr1e-5, weightedBCEpos_weight≈7.037, effectivebatch16 (microbatch4×accumulation4)
เลือกbestcheckpointด้วยvalidationbalancedBCE ไม่เลือกจากtest; เก็บbest/last/snapshots2/7/10
Trainทุกภาพหนึ่งครั้งต่อepoch ไม่ downsample/oversample; lastaccumulationwindowหารตามจำนวนsourceimagesจริง
ค่าเริ่มต้น3seeds42/43/44 รวม36CNN+3DAE; fullrunต้องGPUและใช้เวลามากกว่ารุ่นpilotมาก

## Drive, cache และ resume

- Datasetต้นทางอ่านอย่างเดียว รองรับ `images_NNN/images/<id>` หรือ `images_NNN/<id>`
- Frozenmanifestsแนบในsourcepackage; ตรวจmetadata/splits/bboxhashesและpixelhashก่อนใช้ ไม่สแกนDriveทุกepoch
- Workingcacheuint8shards256²บน`/content/cxr_v3_cache` ประมาณ4.27GiB plusnativeXAI; derivedcacheจำกัด6GiB
- Resultsอยู่`MyDrive/nih -xray/Notebook_3_release/runs/release_full/seed_42` และseed43/44
- Runtimeเปลี่ยนแล้วcacheอาจหาย; สร้างใหม่และresumeได้จากDriveเมื่อsourcepixels/config/code/packageversionsตรงกัน
- ห้ามเปลี่ยนภาพ/manifests/configแล้วresumeoutputเดิม ระบบปฏิเสธprovenanceที่ต่างแทนใช้คะแนนเก่าปนใหม่
- CNN-onlycellจะตรวจ/ฝึกDAEให้ครบbudgetก่อนใช้; `best.pt` จากDAEที่ยังฝึกไม่จบใช้ไม่ได้

## ผลลัพธ์

CNN: Accuracy/BalancedAccuracy/Precision/Recall/Specificity/F1/ROC-AUC/PR-AUC, counts+row%-confusion, ROC/PR, histories
Grad-CAM/SHAP: IoU0.3/0.5, PointingGame, energyinside, emptymaprate, patient-bootstrap95%CI2,000draws และpairedbaseline deltas
SHAPอธิบายprocessedviews ไม่ใช่rawgradientผ่านNumPyFFT/DWT เก็บsignedRGB/viewarraysแยกจากpositiveheatmap
`comparison/` เก็บตารางกราฟstrength/frequency/compute และriskstrata; `seed_summary/` mean±SDแยกจากpatientCIs
Normal-onlypredictorบนtestได้Accuracy81.62%; ไม่อ่านAccuracyเดี่ยว ๆ และไม่เทียบกับbalancedpilotเหมือนเป็นชุดเดียวกัน

ใช้ [check_overfit.ipynb](check_overfit.ipynb) ดูกราฟย้อนหลังด้วยCPUโดยไม่ฝึก/โหลดmodels
Notebookแยกอยู่DAE/CNN/Grad-CAM/SHAP/Frequency/Analysis; [RUN_GUIDE.md](RUN_GUIDE.md) อธิบายการเปิดและresume
Optional `legacy_v2` presetและ`head_only_legacy` ใช้outputแยก; ไม่รวมใน36runsหลัก

## สร้างและตรวจ

```powershell
python build_notebooks.py
python -m unittest discover -s tests -v
python validate_notebook_gpu.py --data-root D:/kaggle_cache/datasets/nih-chest-xrays/data/versions/3
```

ใช้environmentที่มีPyTorchCUDA+torchvisionpairที่ตรงกัน; lightweight dependenciesระบุในrequirements.txt
`prepare_data.py --build-manifests --data-root ...` สร้างเฉพาะCSV/auditบนlocal ไม่ packภาพ; บนColabใช้bundledmanifests
GPUvalidationใช้4train/4val/4test/2XAI, 4epochsเพื่อทดสอบwarmuptransition; randomweights/FP32smokeไม่ใช่ผลวิจัย
ดู [VALIDATION.md](VALIDATION.md) สำหรับผลตรวจและข้อจำกัดที่ตรวจได้จริง
