# 🔬 Denoise + Grad-CAM: การศึกษาผลของการลดสัญญาณรบกวนต่อ Explainable AI

โฟลเดอร์นี้จัดทำขึ้นเพื่อทดลองและเปรียบเทียบผลของ **Denoising ครบทั้ง 7 รูปแบบ** (เหมือนกับที่ทำในการทดลอง CNN) ร่วมกับการสร้างภาพอธิบายผลด้วย **Grad-CAM** บนภาพ Chest X-ray ที่มีภาวะ **Infiltration** และมีพิกัดรอยโรคจริงของแพทย์ (**Ground-Truth Bounding Box**)

---

## รูปแบบการเตรียมภาพ 7 วิธีที่ครอบคลุม

1. **Baseline**: ภาพเอกซเรย์ดิบ (Raw CXR)
2. **Median Filter L1**: Kernel $3 \times 3$
3. **Median Filter L2**: Kernel $5 \times 5$
4. **Median Filter L3**: Kernel $7 \times 7$
5. **CLAHE + DWT L1**: Clip Limit 2.0 / DWT Level 1 / Threshold Scale 0.5
6. **CLAHE + DWT L2**: Clip Limit 4.0 / DWT Level 2 / Threshold Scale 1.0
7. **CLAHE + DWT L3**: Clip Limit 8.0 / DWT Level 3 / Threshold Scale 2.0

---

## การวัดผลด้วย Confusion Matrix ทั้ง 2 รูปแบบ

### 1. แบบที่ 1: Classification Confusion Matrix (Normal vs Infiltration)
* วัดการทำนายผลคลาสของโมเดล CNN (True Normal, False Infiltration, False Normal, True Infiltration) เทียบระหว่าง Baseline vs Median vs CLAHE+DWT
* ดูภาพผลลัพธ์: [output/confusion_matrix_classification.png](output/confusion_matrix_classification.png)

### 2. แบบที่ 2: XAI Localization Confusion Matrix (Pointing Game: BBox Hit vs Miss)
* วัดว่าจุดที่ **Grad-CAM สว่างเข้มที่สุด (Peak Attention)** ตกอยู่ใน **Bounding Box ของแพทย์จริง (Hit)** หรือชี้หลุดกรอบรอยโรค (**Miss**)
* ดูภาพผลลัพธ์: [output/confusion_matrix_xai_localization.png](output/confusion_matrix_xai_localization.png)

---

## โครงสร้างไฟล์ในโฟลเดอร์นี้

- **`denoise_gradcam_comparison.ipynb`**: สมุดโน้ตบุ๊กหลักสำหรับรันเปรียบเทียบ Denoise + Grad-CAM และแสดง Confusion Matrix ทั้ง 2 รูปแบบ
- **`denoise_methods.py`**: โมดูล Python รวบรวมฟังก์ชัน Denoising ทั้ง 7 รูปแบบตามสูตรวิจัย
- **`generate_preview.py`**: สคริปต์เรนเดอร์ภาพเปรียบเทียบ Visual & Grad-CAM
- **`generate_confusion_matrices.py`**: สคริปต์คำนวณและพล็อต Confusion Matrix ทั้ง 2 รูปแบบ
- **`output/`**:
  - `confusion_matrix_classification.png`: ตาราง Confusion Matrix ด้านการทำนายโรค (Classification)
  - `confusion_matrix_xai_localization.png`: ตาราง Confusion Matrix ด้านการชี้ตำแหน่งรอยโรค (XAI Hit vs Miss)
  - `denoise_methods_visual_comparison.png`: ภาพเปรียบเทียบผลลัพธ์ของฟิลเตอร์ทั้ง 7 แบบ
  - `denoise_gradcam_7methods_comparison.png`: ภาพเปรียบเทียบ Grad-CAM Heatmap + BBox แพทย์ครบทั้ง 7 รูปแบบ
