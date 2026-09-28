# Gamma Correction (งานเสริม)

Gamma เป็นการปรับ contrast แบบไม่เชิงเส้น ไม่อยู่ใน 4 กลุ่ม denoising หลักตาม guideline. ทดลองค่า 0.5, 0.8, 1.0, 1.2 และ 1.5 บนภาพตัวอย่าง; gamma=1 คือภาพเดิม

## ไฟล์ในโฟลเดอร์

- [gamma_correction_pipeline.ipynb](gamma_correction_pipeline.ipynb): ทดลอง gamma และเปรียบเทียบกับ CLAHE/DAE+CLAHE
- [gamma_utils.py](gamma_utils.py): lookup table และ contrast metrics
- [generate_gamma_artifacts.py](generate_gamma_artifacts.py), [build_gamma_notebook.py](build_gamma_notebook.py): สร้างภาพ/ตารางและ notebook
- [output/gamma_levels_comparison.png](output/gamma_levels_comparison.png), [output/gamma_vs_clahe_benchmark.png](output/gamma_vs_clahe_benchmark.png): ภาพเทียบ
- [reports/gamma_metrics_summary.csv](reports/gamma_metrics_summary.csv): ค่าความเข้ม, standard deviation, entropy และ dynamic range ตาม gamma

## ผลและข้อจำกัด

ใน CSV ภาพตัวอย่างนี้ gamma 0.5 → 1.5 ทำให้ mean intensity ลด 197.91 → 129.37; contrast std เพิ่ม 34.88 → 47.52. เป็นผลรายภาพ ไม่ใช่คะแนนจำแนกหรือ IoU บน test set. หากรันใหม่ให้ตั้ง working directory เป็นโฟลเดอร์นี้แล้วใช้ python generate_gamma_artifacts.py. สคริปต์อ่านภาพ NIH จาก data/versions/3 และใช้ manifest ใน xai/Gradcam.
