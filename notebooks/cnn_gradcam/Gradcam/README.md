# 🔬 Grad-CAM: Normal vs Infiltration Comparison (NIH Chest X-ray)

โฟลเดอร์นี้จัดทำขึ้นสำหรับการศึกษาเชิงอธิบายผลด้วย **Grad-CAM (Gradient-weighted Class Activation Mapping)** บนภาพถ่ายรังสีทรวงอก (Chest X-ray) โดยเน้นการเปรียบเทียบแบบฐานนิยม (Baseline) ระหว่าง:
1. **ภาพปอดปกติ (Normal / No Finding):** 100 ภาพ (คัดจากคนไข้ที่ไม่ซ้ำกัน)
2. **ภาพปอดอักเสบสารแทรกซึม (Infiltration):** 100 ภาพ (คัดจากคนไข้ที่มีพิกัด Ground-Truth Bounding Box จากแพทย์)

> **เงื่อนไข:** ใช้ภาพเอกซเรย์ดิบ (Raw CXR) ล้วน ๆ ยังไม่มีการทำกระบวนการลดสัญญาณรบกวน (Noise Reduction)

---

## โครงสร้างไฟล์ในโฟลเดอร์นี้

- **`gradcam_normal_vs_infiltration.ipynb`**: สมุดโน้ตบุ๊กหลักสำหรับรัน Grad-CAM, พล็อตภาพ Overlay, เปรียบเทียบ Normal vs Infiltration และประเมินความสอดคล้องกับพิกัด BBox ของแพทย์
- **`sample_manifest_200.csv`**: ตารางบันทึกรายชื่อและ Metadata ของภาพตัวอย่างทั้ง 200 ภาพ (Normal 100 ภาพ, Infiltration 100 ภาพ พร้อมบอก path และสถานะ BBox)
- **`generate_preview.py`**: สคริปต์รันประมวลผลโมเดลและเซฟภาพตัวอย่างผลลัพธ์
- **`output/`**: โฟลเดอร์เก็บภาพตัวอย่างผลลัพธ์ที่ประมวลผลเสร็จแล้ว:
  - `comparison_single_pair.png`: ภาพเปรียบเทียบ Normal vs Infiltration แบบคู่เดี่ยว
  - `batch_comparison_4x4.png`: ตารางเปรียบเทียบ 4 Normal vs 4 Infiltration พร้อม BBox

---

## โมเดลและการทำงานของ Grad-CAM
- **Architecture:** ResNet50 (Pretrained ImageNet)
- **Target Layer:** `layer4[-1]` (Bottleneck Block สุดท้ายก่อน Global Average Pooling)
- **Visuals:** ผสมสีด้วย Jet Colormap และ Overlay ทับภาพ Chest X-ray ในอัตราส่วน Alpha 0.45 พร้อมวาดกรอบเขียว BBox ของแพทย์
