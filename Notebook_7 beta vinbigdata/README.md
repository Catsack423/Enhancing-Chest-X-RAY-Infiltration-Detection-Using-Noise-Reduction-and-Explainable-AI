# Notebook_7 beta (VinBigData 512 Edition)

ชุดทดลอง **v7 beta** ที่เปลี่ยนชุดข้อมูลเป็น **`awsaf49/vinbigdata-512-image-dataset`** (ดึงผ่าน `kagglehub` อัตโนมัติบน Google Colab) โดยแบ่งเป็น 2 คลาสตามโครงสร้างของ VinBigData:

1. **`normal` (Class 0 — ปอดปกติ)**: ภาพที่รังสีแพทย์ระบุว่า `No finding` (`class_id == 14`) และไม่มีภาวะความผิดปกติอื่นใดในภาพ (มีในชุดข้อมูลทั้งหมด ~10,606 ภาพ)
2. **`infiltration` (Class 1 — มีภาวะ Infiltration)**: ภาพปอดที่มีการตรวจพบภาวะ `Infiltration` (`class_name == 'Infiltration'` หรือ `class_id == 6`) อย่างน้อย 1 รายการ แม้ว่าในภาพเดียวกันจะมีภาวะโรคอื่นเกิดร่วมด้วยก็ตาม (มีในชุดข้อมูลทั้งหมด ~613 ภาพ)

---

## ขั้นตอนการทำงาน (Pipeline)

1. **ดาวน์โหลดและแยกคลาส (`v7/colab.py`, `v7/audit.py`)**:
   - ดึงชุดข้อมูล `awsaf49/vinbigdata-512-image-dataset` ลงบน Colab VM ผ่าน `kagglehub`
   - คัดกรองเฉพาะภาพในกลุ่ม `normal` (~10,606 ภาพ) และ `infiltration` (~613 ภาพ) รวม ~11,219 ภาพ
   - แบ่งกลุ่มผู้สมัคร (Candidate Split) แบบไม่ซ้ำภาพด้วย `seed = 42` ในสัดส่วน 64% Train / 16% Validation / 20% Test
   - คำนวณค่า Estimated Noise (`noise_sigma`) ด้วย Wavelet (`db2-patch64-central80-v1`)
2. **คัดเลือกภาพที่มี Estimated Noise ต่ำสุด (`v7/prepare.py`)**:
   - **Train**: 160 `normal` + 160 `infiltration` = **320 ภาพ**
   - **Validation**: 40 `normal` + 40 `infiltration` = **80 ภาพ**
   - **Test**: 50 `normal` + 50 `infiltration` = **100 ภาพ**
   - **XAI Reference**: เก็บพิกัดกรอบ Bounding Box ของรังสีแพทย์สำหรับภาพ `infiltration` ในชุด Test ทั้ง 50 ภาพ
3. **จำลองสัญญาณรบกวน 16 สภาวะ (`v7/noise.py`)**:
   - `clean` + Noise 5 ชนิด (`gaussian`, `poisson`, `poisson_gaussian`, `salt_pepper`, `speckle`) × 3 ระดับ (`N1`, `N2`, `N3`)
4. **ฝึกสอนโมเดลและประเมินผล (`v7/runner.py`, `v7/report.py`)**:
   - เทรน **1 Shared DAE** + **12 ResNet50 CNNs** (Baseline, Median L1–L3, DWT+CLAHE L1–L3, DAE+CLAHE L1–L3, DWT Fusion, DFT Fusion)
   - ประเมินผลทั้ง 16 สภาวะ Noise พร้อมทำ Bootstrap 2,000 รอบ (95% CI) และวัด PSNR/SSIM

---

## โครงสร้างไฟล์

- `run_v7_vinbigdata_colab.ipynb`: ไฟล์ Notebook สำหรับนำไปเปิดรันบน Google Colab ได้ทันที (มีซอร์สโค้ดฝังในตัว)
- `build_notebook.py`: สคริปต์สำหรับสร้างไฟล์ `run_v7_vinbigdata_colab.ipynb` ใหม่เมื่อแก้ไขโค้ดใน `v7/`
- `configs/protocol.json`: ไฟล์ตั้งค่าพารามิเตอร์และนิยามคลาส `normal` vs `infiltration`
- `v7/`: โมดูลซอร์สโค้ดหลัก (`audit.py`, `prepare.py`, `dataset.py`, `noise.py`, `runner.py`, `report.py`, `colab.py`, `smoke.py`) และ `v7/release/`
