# 🔬 SHAP: Explainable AI for Chest X-ray Infiltration Detection

โฟลเดอร์นี้จัดทำขึ้นเพื่อศึกษาและวิเคราะห์โมเดล CNN ด้วย **SHAP (SHapley Additive exPlanations)** บนภาพถ่ายรังสีทรวงอก (Chest X-ray) โดยเน้นการเปรียบเทียบระหว่าง:
1. **ภาพปอดปกติ (Normal / No Finding)**
2. **ภาพปอดอักเสบมีฝ้าแทรกซึม (Infiltration)** ร่วมกับพิกัดรอยโรคจริงของแพทย์ (**Ground-Truth Bounding Box**)
3. **ผลกระทบของฟิลเตอร์ลดสัญญาณรบกวน (Denoising)** 7 รูปแบบ (Baseline, Median L1-L3, CLAHE+DWT L1-L3) ต่อค่าความสำคัญของพิกเซล

---

## โครงสร้างไฟล์ในโฟลเดอร์นี้

- **`shap_normal_vs_infiltration.ipynb`**: สมุดโน้ตบุ๊กหลักสำหรับรันดูการอธิบายผลด้วย SHAP และแสดงตาราง Confusion Matrix
- **`shap_utils.py`**: โมดูล Python สำหรับคำนวณ Layer-wise Expected Gradients / SHAP
- **`generate_all_shap_artifacts.py`**: สคริปต์ประมวลผลโมเดล สร้างภาพตัวอย่างทั้งหมด และออกรายงานสรุปผล
- **`reports/`**: โฟลเดอร์เก็บรายงานผลการวิจัยและสรุปตัวเลขสถิติ
  - `summary_report.md`: รายงานสรุปผลการประเมินเชิงวิชาการอย่างละเอียด
  - `confusion_matrix_summary.csv`: ตารางสรุปค่า Accuracy และ BBox Hit Rate
- **`output/`**: โฟลเดอร์เก็บภาพผลลัพธ์
  - `shap_comparison_normal_vs_infil.png`: ภาพเปรียบเทียบ Normal vs Infiltration พร้อมแยกค่า Positive/Negative Attribution
  - `shap_denoise_effects_comparison.png`: ภาพเปรียบเทียบการเปลี่ยนแปลงของ SHAP เมื่อผ่านฟิลเตอร์ Denoise แต่ละแบบ
  - `confusion_matrix_shap_classification.png`: Confusion Matrix แบบที่ 1 (การทำนายโรค Normal vs Infiltration)
  - `confusion_matrix_shap_localization.png`: Confusion Matrix แบบที่ 2 (การชี้ตำแหน่งรอยโรค Hit vs Miss เทียบกับ BBox)
