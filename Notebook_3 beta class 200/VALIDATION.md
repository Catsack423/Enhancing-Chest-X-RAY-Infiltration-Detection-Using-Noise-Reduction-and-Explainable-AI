# Validation — v3 beta class 200/class, epochs 10

ตรวจวันที่ 2026-10-10 บน Windows / NVIDIA GeForce RTX 4060 8 GB ด้วย `.gpu/Scripts/python.exe` โดยไม่ได้รันการทดลองเต็ม 400 ภาพ

## ข้อมูลและ package

- Train 400 ภาพ: pure No Finding 200 และ pure Infiltration 200; 385 คนไข้
- ภาพฝึกเดิม 320 ภาพถูกเก็บครบ เพิ่ม 40 ภาพต่อคลาสด้วย seed 42 / stream 200400
- ไม่ใช้คนไข้จาก original reserved-validation partition ทั้ง 4,815 คน และไม่ซ้ำกับ validation/test/XAI
- Validation 80/test 100/XAI 123/background 32 ใช้ manifests เดิมทุก byte; official NIH metadata/split hashes ตรงต้นทาง
- Train manifest SHA-256: `35c831fa401dc84aa906123f9dede8e70e3de460647b42d4cde229928f4c8683`
- Data bundle รวม 702 ภาพ ตรวจ working pixels ตรงหลัง encode/decodeทุกภาพ; XAI คง native pixels
- ภาพ/metadata เดิมใน bundle ตรง SHA-256 ของ v2 ทุกไฟล์ และไฟล์ภาพใหม่ครบ 80 ไฟล์
- เปิด nested ZIP ผ่าน `colab_data_from_package` และ `open_frequency_experiment` ได้ครบ 400 train images พร้อมตรวจ metadata, labels, official split, bbox และ file paths

## Tests

คำสั่งจาก workspace:

```powershell
& '.gpu/Scripts/python.exe' -m unittest discover -s 'Notebook_3 beta class/tests' -v
```

**14 tests passed** ใน 12.738 วินาที ครอบคลุม:

- น้ำหนัก backbone และทุก BatchNorm statistics คงเดิมหลัง optimizer update; optimizer เห็น head เท่านั้น (2,049 parameters)
- Feature mean ตรงกับสูตรและ gradient ของ input ผ่านทุก view; Grad-CAM ใช้ frozen parameters ได้
- เก็บ 320 ภาพเดิมครบ/เพิ่ม 80 ภาพ/balanced 200 ต่อคลาส/ไม่รั่วคนไข้/selection ทำซ้ำได้
- epoch 1/2/10 ใช้ 400 IDs เดียวกัน; config เปลี่ยนแล้ว resume lock ต้องปฏิเสธ
- Public training loop ฝึกครบ 10 รอบแม้ validation loss แย่ลงทุกครั้ง และเลือก best epoch 1 ตาม validation
- Model class และ training loop ตรงกับ e2; ต่างเฉพาะจำนวน train, epochs/patience, output และ metadata ของการเพิ่มข้อมูล; e2/v2 ต้นทางตรง recorded hashes
- Notebook schema/syntax/ไม่มี saved outputs, ZIP CRC/member paths/ข้อมูลครบ และ portable loader จาก ZIP

## GPU notebook smoke

แจ้งผู้ใช้ก่อนรัน โดยประเมิน 2–5 นาที แล้วรัน:

```powershell
& '.gpu/Scripts/python.exe' 'Notebook_3 beta class/validate_notebook_gpu.py'
```

**Passed ใน 53.029 วินาที**: รัน 9 code cells จริงของ notebook (เพิ่ม cell จำกัดข้อมูลทดสอบ; ข้าม export เพื่อไม่ทำสำเนา checkpoints)

- ทั้ง baseline / DWT / DFT ฝึกครบ **10 epochs** ด้วย ResNet50 ImageNet V2, frozen backbone/BatchNorm + head 2,049 parameters
- ชุด smoke: train 4/validation 4/test 4, XAI 2/background 2; batch 2, SHAP nsamples 4, bootstrap 20
- ทั้งสามวิธี classification / Grad-CAM / SHAP มี status complete; บันทึก history/checkpoints/heatmaps/overlays/ตาราง/grาฟได้
- Preflight ก่อนลดขนาดข้อมูลอ่านและตรวจ source manifests ชุดเต็ม 400/80/100/123/32 สำเร็จ
- Smoke outputs อยู่ `.runtime/gpu_smoke/` พร้อม `smoke_validation.json` และ `executed_smoke.ipynb` ไม่อยู่ใน Colab ZIP หรือ `runs/class200_e10`

Runtime: torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / SHAP 0.51.0 / NumPy 2.4.6 / pandas 3.0.6 / PyWavelets 1.9.0 / scikit-learn 1.9.1; OpenCV ใช้ headless distribution

## ขอบเขตของหลักฐาน

ยังไม่มีผล training/classification/XAI เต็มของชุด 400 ภาพ การทดสอบนี้ยืนยันว่า notebook และ package ทำงานตามค่าตั้ง ไม่ใช้ smoke Accuracy อ้าง improvement +10 จุดเปอร์เซ็นต์หรืออ้างว่า overfit หาย รุ่นนี้เปลี่ยนทั้งจำนวนภาพและ epochs เทียบ e2/e7 จึงไม่ได้แยกผลของการเพิ่ม epochs อย่างเดียว
