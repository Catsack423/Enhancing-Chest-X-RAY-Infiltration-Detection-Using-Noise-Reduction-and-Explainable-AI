# Notebook v3 beta — ลด overfit ด้วย frozen backbone และ head only

เปิด [frequency_head_only_e2.ipynb](frequency_head_only_e2.ipynb) แล้วรันตามลำดับ เปรียบเทียบ **baseline, DWT L1–L4 + A4 ไม่ใช้ CLAHE และ DFT ต่ำ–กลาง–สูง** ต่อจาก v2 frequency comparison

## สิ่งที่เปลี่ยนจาก v2

| ค่าตั้ง | v2 frequency | v3 beta |
| --- | --- | --- |
| ResNet50 ImageNet V2 | Freeze early layers; ฝึก layer4 + head | Freeze backbone ทุกชั้น; ฝึกเฉพาะ head |
| BatchNorm | layer4 อัปเดต statistics | ทุกชั้น eval แม้ตอนฝึก |
| พารามิเตอร์ที่ฝึก | 14,966,785 | 2,049 |
| Head | Linear(2048,1) | mean feature → Dropout(0.3) → Linear(2048,1) |
| Optimizer | Adam | AdamW, weight_decay=1e-4 |
| Epochs | Max 50, หยุดจริง 7 | **2** |
| Best checkpoint | Validation loss ต่ำสุด | Validation loss ต่ำสุดใน epoch 1–2 |
| Overfit diagnostics | Train loss ขณะ update / val loss | เพิ่ม train_eval_loss และ validation-minus-train gap |

Dropout ใช้หลังเฉลี่ย feature เป็น mask ต่อภาพต้นฉบับ ไม่ขึ้นกับจำนวน views และปิดใน validation/test/XAI Backbone weights และ BatchNorm statistics คงที่ แต่ input gradients ยังทำงานสำหรับ Grad-CAM/SHAP ไม่ครอบ backbone ด้วย no_grad ใน forward

ใช้ regularization/optimizer/seed/จำนวน epochs เหมือนกันทั้ง 3 วิธี ไม่เพิ่ม class weighting เพราะชุด train สมดุล และไม่เพิ่ม augmentation ใน beta นี้เพื่อคง preprocessing เดิม

**นี่เป็น protocol ใหม่เพื่อบรรเทา overfit ยังไม่ใช่หลักฐานว่าแก้สำเร็จ** Training 2 epochs อาจ underfit ต้องอ่าน train_eval_loss เทียบ validation_loss และผล classification/XAI จริง ไม่รับประกันว่าจะชนะ baseline หรือเพิ่ม Accuracy อย่างน้อย 10 จุดเปอร์เซ็นต์

## ข้อมูลและค่าที่คงเดิม

- Train 320 ภาพ (No Finding 160 / Infiltration-only 160), validation 80 (40/40), test 100 (50/50) จาก manifests เดิม ไม่เลือกภาพหรือ split ใหม่
- Official NIH train_val/test ไม่เปลี่ยน คนไข้ train/validation/test ไม่ซ้ำกัน; test เป็น balanced pilot ไม่ใช่ official binary test เต็ม 12,081 ภาพ
- Bbox XAI 123 ภาพ / 115 คนไข้ รวม pure/mixed และคำทำนายผิด; SHAP background train 32 ภาพเดิม
- Seed 42, batch 16 ภาพ, lr 1e-4, threshold 0.5, working size 256/CNN 224, ImageNet mean/std เดิม
- DWT db1 periodization 4 levels → A4/D1/D2/D3/D4; DFT inverse FFT ของ radial low <0.125 / mid 0.125–<0.25 / high ≥0.25 cycles/pixel
- IoU@0.3/0.5, Pointing Game, energy inside, patient-bootstrap 95% CI 2,000 รอบ; SHAP nsamples 200/batch 8

ตัว loader ตรวจ hashes ของ metadata, official lists และทั้ง 5 manifests ก่อนล็อก experiment หากไฟล์หายหรือไม่ตรงจะหยุด ไม่สุ่มชุดใหม่ทดแทน

## รันบน Colab หรือเครื่องนี้

Colab: อัปโหลด [Notebook_3_beta_colab.zip](Notebook_3_beta_colab.zip) ไป MyDrive เปิด notebook และเลือก GPU ZIP มีโค้ด/ข้อมูลเดิม/manifests ครบ ไม่ต้องดาวน์โหลด NIH ใหม่ source/output แยกจาก Notebook_2 บน Drive

Local: เปิด notebook จากโฟลเดอร์นี้ เลือก Python kernel `D:/kaggle_cache/datasets/nih-chest-xrays/.gpu/Scripts/python.exe` ใช้ data เดิมในเครื่องและ pretrained cache เดิมของ v2

Training + classification บน RTX 4060 ประเมิน 1–5 นาที (ไม่รวม download ครั้งแรก); training + XAI ครบทุกภาพประเมินเผื่อ 45–90 นาทีจาก pipeline v2 ก่อนหน้า ยังไม่ได้ benchmark v3 เต็ม 2 epochs ไม่ได้ลดงาน SHAP ซึ่งเป็นส่วนที่ใช้เวลาหลัก

ค่าตั้ง v3: [shared/config.json](shared/config.json) ส่วน [shared/data_bundle_config.json](shared/data_bundle_config.json) เก็บค่าตั้ง v2 สำหรับตรวจ config hash ของ ZIP ภาพเดิม ไม่ใช่ค่าฝึก v3 บันทึก config/code hashes/package versions จริงใน output ทุกครั้ง เปลี่ยน config/โค้ด/packages แล้วให้ใช้ OUTPUT_ROOT ใหม่

## ผลที่สร้างเมื่อรัน

`runs/head_only_e2/` แยกจาก v2 และต้องฝึก baseline ใหม่ภายใต้ v3:

- `CNN/results/<condition>/`: best.pt/last.pt/history.csv/metrics.json/test_predictions.csv/confusion_matrix.png
- `Grad-CAM/results/`, `SHAP/results/`: heatmaps, overlays, per-image metrics, summary และ resumable records
- `comparison/`: classification.csv, xai.csv, paired_xai_deltas.csv, status.csv, training_curves.png, accuracy_target.csv
- `shared/artifacts/`: ค่าตั้ง, manifests เดิม, hashes และรายชื่อภาพต่อ epoch

ใช้ `train_eval_loss` เทียบ `validation_loss` ใน history เพราะทั้งสองวัดด้วย eval mode หลังจบ epoch ส่วน `train_loss` ระหว่าง update เปิด dropout และเปลี่ยนน้ำหนักทุก batch ไม่มี train Accuracy ถูกนำมาแสดงเป็น test Accuracy

เกณฑ์ +10 จุดเปอร์เซ็นต์คิดจาก **baseline v3 + 0.10** ไม่ใช้ baseline เก่า 54% หาก baseline ใหม่เปลี่ยน เกณฑ์ต้องเปลี่ยนตาม ห้ามเลือก epoch/threshold/DFT cutoff จาก test เพื่อให้ผ่านเป้าหมาย ค่า delta อย่างเดียวไม่พิสูจน์นัยสำคัญ

SHAP อธิบาย processed frequency bands ไม่ใช่ attribution ของ raw pixels ผ่าน DWT/FFT; bbox scores ไม่ใช่ CNN Accuracy ผลรอบนี้ยังเป็น pilot และ seed เดียว ไม่ตอบ denoising strength/over-smoothing sweep ของ 10 เงื่อนไขเดิม

## Source และการตรวจ

ยกโค้ด data/preprocessing/evaluation/XAI/provenance จาก v2 มาไว้ในโฟลเดอร์นี้ให้ใช้งานแยกได้ เก็บ SHA-256 ของไฟล์ต้นทางใน [V2_SOURCE_HASHES.json](V2_SOURCE_HASHES.json) ไม่แก้โค้ดหรือผล v2

- [shared/beta.py](shared/beta.py): frozen model, head fusion และ wrapper สำหรับ training/XAI
- [shared/training.py](shared/training.py): AdamW และ train_eval_loss เพิ่มจาก v2
- [tests/test_beta.py](tests/test_beta.py): freeze/BatchNorm/input gradients/XAI/split/checkpoint/notebook checks
- `python build_notebook.py` สร้าง notebook และ Colab ZIP โดยไม่ฝึก
- [VALIDATION.md](VALIDATION.md): ขอบเขตการตรวจและผล smoke แยกจากผลวิจัย

ตรวจผ่าน 9 tests และ GPU notebook smoke บน RTX 4060 (2 epochs, subset เล็ก) ใช้ 33.90 วินาที ครบ training/Grad-CAM/SHAP และตรวจ portable data loader จาก ZIP จริงแล้ว **ยังไม่มีผล Accuracy v3 จากการรันเต็ม**

หลักการอ้างอิง: [PyTorch AdamW](https://docs.pytorch.org/docs/2.13/generated/torch.optim.AdamW.html), [PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html) และ [PyTorch autograd/freeze](https://docs.pytorch.org/docs/stable/notes/autograd.html)
