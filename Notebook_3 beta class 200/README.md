# Notebook v3 beta class — 200 ภาพต่อคลาส / 10 epochs

เปิด [frequency_class200_e10.ipynb](frequency_class200_e10.ipynb) หรืออัปโหลด [Notebook_3_beta_class_colab.zip](Notebook_3_beta_class_colab.zip) ไป MyDrive แล้วเปิด notebook บน Colab เลือก GPU เช่น T4/L4 โค้ดใช้ CUDA

## ข้อมูลที่รันจริง

| ชุด | No Finding | Infiltration-only | รวม |
| --- | ---: | ---: | ---: |
| Train | **200** | **200** | **400** |
| Validation เดิม | 40 | 40 | 80 |
| Test เดิม | 50 | 50 | 100 |
| SHAP background เดิม (subset ของ train) | 16 | 16 | 32 |

XAI ใช้ bbox 123 ภาพ / 115 คนไข้เดิม รวม pure/mixed Infiltration และคำทำนายผิด ภาพที่มีโรคอื่นร่วมไม่ถูกนำมาฝึก CNN

เก็บภาพฝึกเดิม 320 ภาพครบทุกภาพ เพิ่ม **40 ภาพต่อคลาส** จาก NIH official train_val และ patient partition ฝั่ง train เดิมด้วย seed 42 / stream 200400 ไม่ใช้คนไข้จากฝั่ง validation ที่สงวนไว้ทั้งหมด 4,815 คน หรือ official test การเลือกภาพไม่อิงความพร้อมของไฟล์หรือคะแนนโมเดล

Train มี 385 คนไข้และไม่ซ้ำกับ validation/test/XAI ส่วน manifests ของ validation/test/XAI/background เหมือนเดิมทุก byte ทั้งสามวิธีอ่าน train 400 ภาพเดียวกันทุก epoch ไม่สุ่ม split ใหม่ตอนเปิด notebook

`reference_run/` เก็บ manifests เดิม 320 ภาพ; `source_run/` เก็บ manifests 400 ภาพพร้อม [รายงานการเพิ่มข้อมูล](source_run/shared/artifacts/dataset_selection.json) และ [80 ภาพที่เพิ่ม](source_run/shared/artifacts/added_train_images.csv) ภาพใน ZIP รวม 702 ไฟล์: XAI ใช้ native pixels ส่วน classification เก็บ lossless PNG ของ working pixels 256×256 และตรวจ round-trip แล้ว

## โมเดลและการฝึก

คงมาตรการลด overfit ของ v3 beta: ResNet50 ImageNet V2 freeze backbone weights และ BatchNorm statistics ทั้งหมด ฝึกเฉพาะ head **2,049 parameters** ด้วย `mean(feature 2048) → Dropout(0.3) → Linear(2048,1)` ใช้ AdamW, lr 1e-4, weight_decay 1e-4, batch 16, seed 42, threshold 0.5 ไม่เพิ่ม augmentation หรือ class weighting

**epochs=10 และ patience=10** ทำให้ฝึกครบ 10 รอบแม้ validation loss แย่ลงตั้งแต่รอบ 2 ประเมินด้วย `best.pt` ที่ validation loss ต่ำสุดในรอบ 1–10 ซึ่งอาจอยู่ก่อนรอบ 10 เก็บ `last.pt` รอบ 10 เพื่อ resume ภายใน protocol เดิม

| วิธี | Views ที่ใช้ | การรวม |
| --- | --- | --- |
| Baseline | Raw 1 view | Feature ของ raw image |
| DWT ไม่ใช้ CLAHE | A4 + D1/D2/D3/D4, db1 periodization, 5 views | เฉลี่ย feature จาก shared ResNet50 |
| DFT ต่ำ–กลาง–สูง | Inverse FFT ของ low <0.125, mid 0.125–<0.25, high ≥0.25 cycles/pixel, 3 views | เฉลี่ย feature จาก shared ResNet50 |

ใช้ Dropout หลังรวม feature เท่ากันทุกวิธี DWT ไม่มี coefficient threshold ส่วน signed detail ใช้ `0.5 + 0.5 × component` ไม่มี clipping/min-max ต่อภาพ ใช้ working size 256 / CNN size 224 / RGB และ ImageNet mean/std เดิม

## ผลและตำแหน่งไฟล์

ฝึก baseline/DWT/DFT ใหม่ด้วยชุด 400 ภาพเดียวกันทั้งหมด โดยแยกจากรุ่น e2/e7:

| รายการ | ตำแหน่ง |
| --- | --- |
| Local output | `runs/class200_e10/` |
| Colab source | `MyDrive/Notebook_3_beta_class/` |
| Colab output | `MyDrive/Notebook_3_beta_class/runs/class200_e10/` |
| History ต่อวิธี | `CNN/results/<condition_id>/history.csv` ภายใต้ output |
| ตารางและกราฟ | `comparison/` ภายใต้ output |
| Local kernel | `D:/kaggle_cache/datasets/nih-chest-xrays/.gpu/Scripts/python.exe` |

รายงาน Accuracy/Precision/Recall/F1/ROC-AUC/confusion matrix และกราฟ `train_eval_loss` เทียบ `validation_loss` ตลอด 10 รอบเพื่อดู overfit การ freeze/regularization ไม่รับประกันว่าปัญหาจะหมดหรือ Accuracy จะดีขึ้น

Grad-CAM และ SHAP ใช้ checkpoint เดียวกับ classification วัด IoU@0.3/0.5, Pointing Game, energy inside และ patient-bootstrap 95% CI 2,000 รอบ พร้อม paired deltas SHAP backgrounds 32/nsamples 200/batch 8 แผนที่อธิบาย processed frequency views ไม่ใช่ gradient ผ่าน NumPy transform ของ raw image

เป้าหมาย +10 จุดเปอร์เซ็นต์คิดจาก **baseline ของชุด 400 ภาพ / epochs 10 + 0.10** เทียบกับ e2/e7 ได้เชิงภาพรวม แต่จำนวนภาพและ epochs เปลี่ยนพร้อมกัน จึงอ้างผลของการเพิ่ม epochs อย่างเดียวไม่ได้ Test 100 ภาพเป็น pilot เดิม ไม่ใช่ official binary test เต็ม 12,081 ภาพ

CNN บันทึกทุก epoch และ XAI บันทึกรายภาพลง Drive รันซ้ำ resume ได้จาก output เดิม ถ้า config/code/package versions เปลี่ยนให้ใช้ output ใหม่ เซลล์ท้าย export ผลจริงเป็น `Notebook_3_beta_class_results.zip`

## การสร้างและตรวจ

`python build_notebook.py` สร้าง notebook/Colab ZIP โดยไม่ฝึก ส่วน `python prepare_data.py` ทำซ้ำการเลือกภาพที่กำหนดไว้และสร้าง image ZIP ต้องมี NIH ต้นทางบนเครื่อง

[shared/config.json](shared/config.json) คือค่าฝึกของรุ่นนี้ [E2_SOURCE_HASHES.json](E2_SOURCE_HASHES.json) และ [V2_SOURCE_HASHES.json](V2_SOURCE_HASHES.json) ใช้ตรวจว่าต้นทางยังเหมือนเดิม ผลการตรวจจะบันทึกใน [VALIDATION.md](VALIDATION.md) GPU smoke อยู่ `.runtime/` และไม่ปนใน ZIP หรือผลวิจัย

ผ่าน **14 tests** รวมการตรวจข้อมูลใหม่/คนไข้/held-out manifests, frozen backbone/BatchNorm, gradients สำหรับ XAI, ฝึกครบ 10 รอบแม้ validation แย่ลง และเปิดชุด 400 ภาพผ่าน Colab package จริง GPU notebook smoke ทั้งสามวิธีฝึก 10 รอบ พร้อม Grad-CAM/SHAP ผ่านใน **53.03 วินาที** ใช้เพียง 4 ภาพฝึก จึงไม่ใช่ผล Accuracy ของชุดเต็ม

เวลาเต็มบน RTX 4060 ประเมิน training/classification 2–10 นาที และรวม XAI เผื่อ 45–90 นาทีตาม pipeline เดิม ยังไม่ใช่ benchmark เต็มของรุ่น 400/e10 และไม่รวม download ครั้งแรก ยังไม่ได้รันการทดลองเต็มเพื่อรายงาน Accuracy
