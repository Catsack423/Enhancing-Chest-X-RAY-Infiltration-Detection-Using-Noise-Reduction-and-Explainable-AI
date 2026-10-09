# DWT L1–L4 / DFT frequency feature fusion

## ผลรันจริงที่นำเข้าแล้ว

ผลที่ผู้ใช้ส่งเมื่อ 9 ตุลาคม 2026 อยู่ใน [รายงานผลรอบ frequency](../runs/frequency_fusion_existing_split/README.md): Baseline Accuracy 54%, DWT 55% (+1 จุดเปอร์เซ็นต์), DFT 63% (+9 จุดเปอร์เซ็นต์) ยังไม่ถึงเกณฑ์ +10 จุดเปอร์เซ็นต์ (เป้าหมายอย่างน้อย 64% เมื่อ baseline 54%)

CNN มี checkpoints/history/predictions ครบทั้ง 3 วิธี ส่วน XAI ที่ส่งมามี summary และ paired-delta CSV ยังไม่มี heatmap/per-image outputs ใน archive นี้ ดู [classification.csv](../runs/frequency_fusion_existing_split/comparison/classification.csv), [xai.csv](../runs/frequency_fusion_existing_split/comparison/xai.csv), [กราฟ train/validation loss](../runs/frequency_fusion_existing_split/analysis/cnn_training_curves.png) และ [README หลัก](../README.md)

ชุดที่ฝึกสมดุล 160/160 และ history แสดง overfit หลัง epoch 2; Grad-CAM ลดลงทั้ง DWT/DFT แม้ Accuracy ของ DFT เพิ่ม ส่วน SHAP DFT เพิ่มเล็กน้อยแต่ paired CI ของผลต่าง IoU@0.3 คร่อม 0 การนำเข้าไฟล์ไม่ได้ฝึกเพิ่มหรือเปลี่ยนเงื่อนไขเดิม

## การรัน notebook

เปิด `frequency_comparison.ipynb` แล้วรันตามลำดับ (รวม training, classification, Grad-CAM, SHAP)
สำหรับ Colab ให้อัปโหลด `Notebook_2_Frequency_colab.zip` ไป MyDrive ก่อน และเลือก GPU
ZIP ใช้ภาพจาก `nih_cxr_subset.zip` เดิมทุก byte และแนบ manifests จาก
`runs/pilot320_val80_test100/shared/artifacts/` โดยตรง ไม่มีการสุ่มรายชื่อใหม่

Local บนเครื่องนี้: เปิด notebook ใน VS Code/Jupyter แล้วเลือก Python kernel
`D:/kaggle_cache/datasets/nih-chest-xrays/.gpu/Scripts/python.exe`
environment นี้ติดตั้ง PyTorch CUDA และ dependencies พร้อมแล้ว
ทดสอบ RTX 4060 ผ่านทั้ง training/Grad-CAM/SHAP และ batch เดิม
ประเมินรันเต็มพร้อม XAI ประมาณ 45–90 นาที อ่านขอบเขตการวัดใน `VALIDATION.md`

## วิธีที่เปรียบเทียบ

- Baseline: raw view เดียวและ preprocessing เดิม
- DWT: db1, 4 levels, no CLAHE / thresholding; reconstruct A4 และรายละเอียด D1–D4
  แยกเป็น 5 ภาพในพิกัดเดียวกับภาพต้นฉบับ
- DFT: radial low (<0.125), mid (0.125–<0.25), high (>=0.25) cycles/pixel;
  inverse FFT กลับเป็นภาพก่อนเข้า CNN; masks ครบทุก coefficient และไม่ซ้อนกัน

ทุก view ผ่าน ResNet50 น้ำหนักชุดเดียว จากนั้นเฉลี่ย feature 2048 ค่าก่อน Linear head เดิม
คำนวณเทียบเท่าด้วย mean(logits) ก่อน sigmoid เพราะ head เป็น linear
จำนวนพารามิเตอร์เท่ากัน แต่ DWT/DFT ใช้ compute มากกว่าและ train BatchNorm layer4 เห็นหลาย views
นี่เป็นการทดลอง frequency feature fusion เพิ่มเติม ไม่ใช่ denoising strength sweep

สืบทอด train 320 / validation 80 / test 100 เดิม, 50 epochs, batch 16 ภาพ,
Adam lr=1e-4, patience 5, ImageNet V2, seed 42 และ frozen early layers
Test เป็น balanced subset เดิม ไม่ใช่ official filtered test ทั้ง 12,081 ภาพ
Baseline ฝึกใหม่ใน output แยกเพื่อใช้ implementation/environment เดียวกับวิธีใหม่

## XAI และผลลัพธ์

123 bbox images / 115 patients เดิม รวม mixed cases และคำทำนายที่ผิด
Grad-CAM รวม contributions ทุก view; SHAP ใช้ train backgrounds 32 ภาพเดิม,
Expected Gradients nsamples=200 และเก็บ signed attribution แยกทุก view/RGB
SHAP นี้อธิบาย processed bands ไม่ใช่ raw-pixel attribution ผ่าน FFT/DWT
รายงาน IoU 0.3/0.5, Pointing Game, energy inside, patient-bootstrap 95% CI 2,000 รอบ
แยก overall/pure/mixed_high_risk/mixed_low_risk และ paired XAI differences กับ baseline
classification differences เป็น descriptive; ไม่อ้าง statistical significance จาก delta อย่างเดียว

เมื่อรัน notebook จนจบ โครงสร้าง output ที่สร้างคือ `runs/frequency_fusion_existing_split/` (archive ที่นำเข้าตอนนี้มีเฉพาะไฟล์ที่ระบุในรายงานข้างต้น):

- `CNN/results/<condition>/`: best/last checkpoints, history, predictions, metrics, confusion matrix
- `Grad-CAM/results/` และ `SHAP/results/`: native-resolution heatmaps, overlays, per-image metrics, summary
- `comparison/`: classification.csv, xai.csv, paired_xai_deltas.csv, status.csv
- `shared/artifacts/`: copied manifests, config/source hashes และ per-epoch image IDs

output จาก notebook ครบชุดรองรับ resume ทั้ง training และ XAI; archive ที่นำเข้ายังไม่มี shared epoch artifacts และ raw XAI ของรอบ Colab ครบ จึงควรใช้ output ใหม่หากจะรันในเครื่องนี้ หากเปลี่ยน config/code/package versions ต้องใช้ output ใหม่
ปรับ cutoff โดยใช้ validation เท่านั้น ห้ามเลือกจาก test หรือ bbox test
หาก GPU memory ไม่พอ อย่าลด batch เฉพาะวิธีเดียว: เลือก GPU ที่พอ หรือกำหนดการทดลองใหม่ที่
เปลี่ยน batch เท่ากันทุกเงื่อนไขและรายงานว่าไม่ใช่ค่าฝึกเดิมแล้ว

## สร้างไฟล์ใหม่จาก source

`python build_frequency_notebook.py` สร้าง notebook และ ZIP โดยไม่ฝึกโมเดล
อ่าน `VALIDATION.md` ในโฟลเดอร์นี้สำหรับขอบเขตที่ตรวจได้บนเครื่อง local

อ้างอิง implementation: [PyWavelets multilevel DWT](https://pywavelets.readthedocs.io/en/stable/ref/2d-decompositions-overview.html)
และ [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html)
