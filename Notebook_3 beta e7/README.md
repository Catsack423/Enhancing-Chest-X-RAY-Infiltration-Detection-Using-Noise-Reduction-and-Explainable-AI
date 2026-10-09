# Notebook v3 beta e7 — เปรียบเทียบ budget 2 กับ 7 epochs

เปิด [frequency_head_only_e7.ipynb](frequency_head_only_e7.ipynb) หรืออัปโหลด [Notebook_3_beta_e7_colab.zip](Notebook_3_beta_e7_colab.zip) ไป MyDrive แล้วเลือก GPU บน Colab (T4/L4/A100 ฯลฯ; โค้ดใช้ CUDA ไม่รองรับ TPU)

## เปลี่ยนเฉพาะ budget การฝึก

| ค่าตั้ง | v3 beta e2 | v3 beta e7 |
| --- | --- | --- |
| Epochs | 2 | **7** |
| Patience | 2 | **7** เพื่อไม่หยุดก่อนครบ 7 epochs |
| Checkpoint ที่ประเมิน | Validation loss ต่ำสุดใน epoch 1–2 | Validation loss ต่ำสุดใน epoch 1–7 |
| Model/regularization | Frozen ResNet50/BatchNorm + head 2,049 parameters, Dropout 0.3 | เหมือน e2 |
| Optimizer | AdamW, lr 1e-4, weight_decay 1e-4 | เหมือน e2 |
| Seed/batch/data | Seed 42, batch 16, train 320/validation 80/test 100 เดิม | เหมือน e2 |

**ฝึกครบ 7 epochs แต่ best checkpoint อาจอยู่ก่อน epoch 7** เลือกด้วย validation ไม่เลือก epoch จาก test หรือบังคับใช้ last.pt นี่เป็นการเปรียบเทียบ budget ภายใต้กติกาเลือก checkpoint เดียวกัน ไม่รับประกัน Accuracy หรือ XAI ดีขึ้น และยังไม่มีผลรันเต็มในไฟล์ที่ส่ง

Baseline/DWT/DFT ฝึกจาก ImageNet initialization ใหม่ใน output แยก ไม่ต่อจาก checkpoint e2 ส่วน code model, loss/optimizer loop และ preprocessing คงเดิม ตรวจเอกสารต้นทางของ e2 ได้ที่ [README e2](../Notebook_3%20beta/README.md)

## Data / frequency / XAI เดิม

- Train 320 (No Finding 160 / Infiltration-only 160), validation 80 (40/40), balanced test 100 (50/50); manifests ทั้ง 5 ไฟล์ byte-identical กับ e2/v2 ไม่สุ่ม split ใหม่ และไม่เปลี่ยน official NIH train_val/test
- Working size 256, CNN 224, RGB/ImageNet normalization, threshold 0.5 เดิม
- Baseline raw 1 view; DWT db1 periodization: A4 + D1/D2/D3/D4 รวม 5 views ไม่มี CLAHE; DFT inverse FFT ของ low/mid/high 3 views ที่ cutoff 0.125/0.25 cycles/pixel
- เฉลี่ย feature 2048 ค่าก่อน Dropout/Linear, freeze backbone weights และ BatchNorm ทั้งหมด ไม่เพิ่ม augmentation หรือ class weighting
- XAI bbox 123 ภาพ / 115 คนไข้เดิม รวม pure/mixed และคำทำนายผิด; Grad-CAM/SHAP, IoU 0.3/0.5, Pointing Game, energy inside, patient-bootstrap CI 2,000 รอบ; SHAP backgrounds 32/nsamples 200/batch 8

## ตำแหน่งที่แยกจาก e2

| รายการ | ตำแหน่ง |
| --- | --- |
| Colab source | `MyDrive/Notebook_3_beta_e7/` |
| Output e7 | `MyDrive/Notebook_3_beta_e7/runs/head_only_e7/` |
| Output e2 ที่เซลล์อ่าน | `MyDrive/Notebook_3_beta/runs/head_only_e2/` |
| Local kernel | `D:/kaggle_cache/datasets/nih-chest-xrays/.gpu/Scripts/python.exe` |

ถ้าผล e2 อยู่ที่อื่น ให้แก้ `E2_RESULTS_ROOT` หลัง Setup ส่วน output ของ e2 ไม่ถูกเขียนทับหรือเปลี่ยน code/config

เปลี่ยน config/code/package versions แล้วใช้ output ใหม่ ระบบล็อก provenance และ resume ได้ภายใน protocol เดิม CNN บันทึกทุก epoch และ XAI บันทึกรายภาพลง Drive

## เปรียบเทียบผล

เซลล์ท้าย notebook อ่านผล e2 จริง โดยตรวจ manifests, CNN settings (ยกเว้น epochs/patience), data/preprocessing/seed/XAI settings, เวอร์ชัน packages, test IDs และ checkpoint hashes ก่อนเทียบ

- ตาราง classification ทั้ง e2/e7: Accuracy, Precision, Recall, F1, ROC-AUC, best epoch และ loss สุดท้าย
- ผลต่าง `e7−e2` รายวิธี และกราฟ train_eval_loss/validation_loss ของทั้งสอง budget
- ตาราง XAI mean/95% CI ของทั้งสอง budget เมื่อมี summary ครบคู่
- เก็บ comparison ใน `runs/head_only_e7/comparison/e2_vs_e7/`

ถ้ายังไม่มีผล e2 หรือไม่ครบ 3 วิธี จะแสดงไฟล์ที่ขาด ไม่ใช้คะแนนเก่า v2 หรือ GPU smoke แทน รันเซลล์เปรียบเทียบใหม่ได้เมื่อมีผล e2 โดยไม่ต้องฝึก e7 ซ้ำ หาก packages/เงื่อนไขไม่ตรงจะหยุดการเปรียบเทียบและแจ้งเหตุผล

Improvement ของ DWT/DFT เทียบ baseline ของแต่ละ budget ส่วนเป้าหมาย +10 จุดเปอร์เซ็นต์คิดจาก baseline e7 + 0.10 ผลต่าง e2/e7 เป็น descriptive ยังไม่มี paired statistical test ของความต่างระหว่าง budgets อย่าสรุปนัยสำคัญจาก delta หรือจำนวน epochs อย่างเดียว

## เวลาและการตรวจ

Training/classification ประเมิน 1–5 นาทีบน RTX 4060 ไม่รวม download; พร้อม XAI ครบชุดเผื่อ 45–90 นาทีตาม pipeline เดิม จำนวน XAI เท่าเดิมแม้เพิ่ม epochs เวลาเหล่านี้ยังไม่ใช่ benchmark เต็มของ e7

[VALIDATION.md](VALIDATION.md) บันทึกขอบเขต tests/GPU smoke; outputs ของการทดสอบอยู่ `.runtime/` ไม่ปนในผลจริงหรือ Colab ZIP ดูค่าฝึกที่ [shared/config.json](shared/config.json) และโค้ดเปรียบเทียบที่ [shared/epoch_comparison.py](shared/epoch_comparison.py)

ผ่าน 13 tests และ GPU smoke ทั้ง training 7 epochs/Grad-CAM/SHAP รวมถึงเซลล์เปรียบเทียบกับ e2 smoke ตรวจสอง epochs แรกได้ loss/IDs ตรงกันทุกค่าทั้ง 3 วิธี **ยังไม่มีผลรันเต็มเพื่อสรุป Accuracy e2 เทียบ e7**

`python build_notebook.py` สร้าง notebook/ZIP จาก source โดยไม่ฝึก nested image ZIP เดิมทุก byte และ source manifests เดิมอยู่ใน package เก็บ [E2_SOURCE_HASHES.json](E2_SOURCE_HASHES.json) สำหรับพิสูจน์ว่ารุ่น e2 ไม่เปลี่ยน
