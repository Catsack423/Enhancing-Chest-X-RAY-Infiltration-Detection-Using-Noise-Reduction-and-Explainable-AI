# DAE + CLAHE

โมดูล fusion denoising หลัก: Denoising Autoencoder (DAE) ที่ฝึกกับข้อมูลในโปรเจกต์ ตามด้วย CLAHE. เป้าหมายคือดูการรักษารายละเอียดฝ้าเมื่อเทียบกับ Median และ CLAHE+DWT รวมถึงดูผลต่อ Grad-CAM; ผลหลักตาม guideline ยังต้องยืนยันบนการทดลองที่ควบคุมครบ

## ขั้นตอนและระดับ

DAE รับภาพ grayscale แล้วสร้างภาพลด noise ด้วย convolution/skip connections; CLAHE ใช้ grid 8×8 และ clip limit L1=2, L2=4, L3=8. ภาพเทียบแสดง raw, noisy, DAE และผลหลัง CLAHE. ใช้ [Median_CLAHE_DWT](../Median_CLAHE_DWT/README.md) เป็นตัวเปรียบเทียบ

## ไฟล์ในโฟลเดอร์

- [dae_clahe_pipeline.ipynb](dae_clahe_pipeline.ipynb): ทดลองภาพตัวอย่าง ระดับ CLAHE และ Grad-CAM
- [dae_clahe_utils.py](dae_clahe_utils.py): โมเดล DAE, preprocessing และ metrics
- [train_dae_model.py](train_dae_model.py): ฝึก DAE โดยกรองตาม official train_val_list.txt; [checkpoints/dae_trained.pth](checkpoints/dae_trained.pth) เป็นน้ำหนักที่บันทึกไว้เดิม
- [generate_dae_clahe_artifacts.py](generate_dae_clahe_artifacts.py): สร้างภาพและตารางใหม่; [build_dae_notebook.py](build_dae_notebook.py): สร้าง notebook ใหม่
- [output/](output/): pipeline stages, ระดับ CLAHE, faceoff สี่วิธี, Grad-CAM และ confusion matrix สองภาพ
- [reports/dae_vs_traditional_metrics.csv](reports/dae_vs_traditional_metrics.csv): ค่ารายภาพ PSNR, SSIM, EPI และ CIR

## ผลที่มีในไฟล์ CSV

CSV ปัจจุบันมี 30 แถว (Normal 16, Infiltration 14) สำหรับภาพที่ใส่ noise สังเคราะห์. ค่าเฉลี่ยจากไฟล์:

| วิธี | PSNR (dB) | SSIM | EPI |
|---|---:|---:|---:|
| Noisy input | 22.93 | 0.336 | — |
| Median | 31.12 | 0.800 | 0.764 |
| DWT | 18.09 | 0.408 | 0.353 |
| DAE | 22.79 | 0.350 | 0.577 |

CIR ของ DAE+CLAHE เฉลี่ย 1.20×. ตารางนี้ไม่รองรับข้อความใน report เก่าที่อ้างว่า DAE ชนะ PSNR/SSIM/EPI หรือประเมิน 200 ภาพ; จึงยังสรุปประสิทธิภาพเหนือวิธีอื่นไม่ได้. ภาพ Grad-CAM เป็นตัวอย่างเชิงภาพ ไม่ใช่การทดสอบนัยสำคัญ

## การรัน

ตั้ง working directory เป็นโฟลเดอร์นี้แล้วรัน python generate_dae_clahe_artifacts.py เพื่อสร้างภาพ/CSV หรือเปิด notebook. สคริปต์ฝึกปัจจุบันใช้ manifest จาก xai/Gradcam ซึ่ง Infiltration ทั้ง 100 ภาพอยู่ใน official test; ตัวกรอง train_val จะหยุดการฝึกจนกว่าจะส่ง manifest ฝึกที่ถูกต้องให้ train_dae(). ไม่ทราบที่มาของ checkpoint เดิมละเอียดพอที่จะยืนยันว่าไม่มี leakage จึงไม่ใช้ผล DAE เดิมเป็นข้อสรุปสุดท้าย. การเทียบกับเกณฑ์หลัก 123 bbox และ 95% bootstrap CI อยู่ในงานต่อไป.
