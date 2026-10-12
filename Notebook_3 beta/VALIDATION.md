# Validation — Notebook v3 beta

ตรวจเมื่อ 9 ตุลาคม 2026 ผ่านทั้งหมด ไม่ใช่ผลวิจัยบน train 320/test 100/XAI 123 ภาพ ยังไม่ได้รันเต็มเพื่อวัด Accuracy v3 หรือยืนยันว่า overfit ลดลง

## Focused checks

รันด้วย `.gpu/Scripts/python.exe -m unittest discover -s "Notebook_3 beta/tests" -v`: **9 tests ผ่าน** ใช้เวลาทดสอบ 4.51 วินาที (ไม่รวม Python import/startup)

- Optimizer อัปเดตเฉพาะ head 2,049 พารามิเตอร์ ตรวจ backbone state ทุก tensor ทั้ง weights และ buffers เท่าเดิมหลัง optimizer step; ไม่มี backbone weight gradients
- BatchNorm ทุกชั้น รวม layer4 อยู่ eval ตอน training และ head Dropout อยู่ train; eval ปิด Dropout
- เฉลี่ย pooled features แล้วเข้า head ได้ผลตรง forward; input gradients ครบทุก view และ Grad-CAM ใช้ frozen backbone ได้กับ raw/DFT/DWT
- ค่าตั้ง epochs=2/AdamW/Dropout ร่วมทุกวิธี และ config ที่เปลี่ยน resume lock เดิมไม่ได้
- ทั้ง 5 manifests byte/hash ตรงกับ v2; epoch 1–2 ใช้ภาพ train 320 เดิมครบ คลาสละ 160
- Baseline pixels เดิมและรูปทรง frequency views 1/3/5 views ถูกต้อง
- Notebook schema และ syntax ผ่าน ไม่มี outputs/results ค้างใน notebook ที่ส่ง
- Package CRC/members ผ่าน ไม่มี runtime/results ปน; nested data ZIP byte-identical กับ v2 และ source manifests ตรงทุกไฟล์
- โค้ดและ source manifests v2 ไม่เปลี่ยน เทียบ SHA-256 กับ V2_SOURCE_HASHES.json

## GPU smoke: execute notebook cells

NVIDIA GeForce RTX 4060; torch 2.13.0+cu130, torchvision 0.28.0+cu130, SHAP 0.51.0 ใช้ notebook ที่ส่งจริงและแทรก smoke overrides เฉพาะขนาดข้อมูล/batch/XAI budget ในสำเนาที่รัน

- 4 train / 4 validation / 4 test, 2 bbox XAI, 2 SHAP backgrounds
- **2 epochs** ทั้ง baseline, DWT และ DFT; batch 2, SHAP nsamples 4, bootstrap 20
- 9 code cells (ไม่รวม export); ผ่าน CNN/Grad-CAM/SHAP ทั้ง 3 วิธีใน **33.90 วินาที**
- Save/reload best/last checkpoints, predictions, histories รวม train_eval_loss, confusion matrices, graph, heatmaps/overlays, XAI summaries และ paired deltas ทำงาน
- Best checkpoint เลือกด้วย validation จาก epoch 1 หรือ 2 ไม่บังคับเลือก epoch 2

GPU smoke script: [validate_notebook_gpu.py](validate_notebook_gpu.py); outputs แยกไว้ใน `.runtime/gpu_smoke/` ไม่อยู่ใน output หลักหรือ Colab ZIP ไฟล์ `smoke_validation.json` และ `executed_smoke.ipynb` บันทึกสถานะและโค้ดที่รัน ไม่รายงานคะแนนจากชุดเล็กนี้เป็นผลวิจัย

## Portable data loader

เรียก `colab_data_from_package` จาก ZIP ใหม่จริงบนเครื่อง local ตรวจและแตก nested data ZIP จาก member `Notebook_3/nih_cxr_subset.zip` แล้วเปิด v3 experiment ด้วยข้อมูลที่แตกออก สำเร็จพร้อม manifests train 320 / validation 80 / test 100 / XAI 123 / background 32; epochs=2, head_only

เป็นการตรวจ path/extraction/data hashes ของ package ด้วยตัว loader เดียวกับ Colab ยังไม่ได้ execute บน Colab GPU หรือทดสอบ Drive mount ใน Colab

## ขอบเขตที่ยังไม่ได้วัด

ยังไม่รันเต็ม 320/80/100 ภาพหรือ XAI 123 ภาพ และไม่ยืนยัน default batch 16/full SHAP budget 200 ด้วย v3 benchmark เต็ม GPU smoke ใช้ batch 2 เพื่อทดสอบ pipeline, code และ gradients ส่วนค่าตั้งของ notebook จริงเป็น batch 16/nsamples 200

Training + classification ประเมิน 1–5 นาที; training + XAI เผื่อ 45–90 นาทีจากการวัด v2 ก่อนหน้า ไม่ใช่เวลารันเต็ม v3 ที่วัดแล้ว การ freeze ไม่ตัด input gradients ของ XAI และ epochs=2 ไม่ได้ลดจำนวนภาพที่ SHAP ต้องอธิบาย

