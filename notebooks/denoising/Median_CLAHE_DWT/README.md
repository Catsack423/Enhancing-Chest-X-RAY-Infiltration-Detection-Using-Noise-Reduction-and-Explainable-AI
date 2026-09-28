# Median Filter และ CLAHE+DWT

โมดูลวิธีเตรียมภาพหลัก 7 สภาวะ: raw baseline, Median L1–L3 และ CLAHE+DWT L1–L3. มี Grad-CAM แบบภาพตัวอย่างเพื่อดูการเปลี่ยนตำแหน่งความสนใจหลังปรับภาพ. ตารางจำแนกที่ฝึกโมเดลจริงอยู่ใน [classification/cnn](../../classification/cnn/README.md)

## วิธีและระดับ

| วิธี | L1 | L2 | L3 |
|---|---|---|---|
| Median | kernel 3×3 | 5×5 | 7×7 |
| CLAHE+DWT preset | clip 2, depth 1, threshold ×0.5 | clip 4, depth 2, threshold ×1.0 | clip 8, depth 3, threshold ×2.0 |

CLAHE+DWT ใช้ wavelet db1. ระดับ preset เปลี่ยน clip limit, depth และ threshold พร้อมกัน; การเปรียบเทียบระดับจึงไม่แยกผลของแต่ละพารามิเตอร์

## ไฟล์ในโฟลเดอร์

- [denoise_gradcam_comparison.ipynb](denoise_gradcam_comparison.ipynb): แสดง 7 ภาพและ Grad-CAM เทียบ bbox
- [denoise_methods.py](denoise_methods.py): ฟังก์ชันเตรียมภาพทั้ง 7 สภาวะ
- [build_notebook.py](build_notebook.py): สร้าง notebook ด้านบนใหม่
- [generate_preview.py](generate_preview.py): สร้างภาพเปรียบเทียบ
- [generate_confusion_matrices.py](generate_confusion_matrices.py): สร้างภาพ confusion matrix
- [output/denoise_methods_visual_comparison.png](output/denoise_methods_visual_comparison.png), [output/denoise_gradcam_7methods_comparison.png](output/denoise_gradcam_7methods_comparison.png): ภาพตัวอย่างภาพเตรียมและ heatmap
- [output/confusion_matrix_classification.png](output/confusion_matrix_classification.png), [output/confusion_matrix_xai_localization.png](output/confusion_matrix_xai_localization.png): ภาพ matrix สองชนิด

## การรันและขอบเขตผล

ตั้ง working directory เป็นโฟลเดอร์นี้แล้วเปิด notebook หรือรัน python generate_preview.py / python generate_confusion_matrices.py. สคริปต์อ่าน manifest 200 ภาพจาก [xai/Gradcam](../../xai/Gradcam/README.md) และภาพต้นฉบับจาก data/versions/3. สคริปต์ภาพใช้ ResNet50 น้ำหนัก ImageNet; ภาพ matrix ที่เก็บไว้เป็นตัวอย่างของ pipeline และไม่ใช่ผลหลักจาก CNN pilot ที่เทรนสองคลาสหรือการประเมิน 123 bbox พร้อม CI.
