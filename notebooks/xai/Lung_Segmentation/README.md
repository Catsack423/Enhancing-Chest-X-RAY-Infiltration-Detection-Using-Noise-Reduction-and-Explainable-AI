# Lung field segmentation (งาน XAI เสริม)

ทดลองสร้าง lung mask เพื่อจำกัดบริเวณภาพก่อนแสดง Grad-CAM. เป็น preprocessing สำหรับการสำรวจตำแหน่ง XAI ไม่ใช่หนึ่งใน 4 กลุ่ม denoising หลัก

## ไฟล์ในโฟลเดอร์

- [lung_segmentation_pipeline.ipynb](lung_segmentation_pipeline.ipynb): ขั้นตอนการสร้าง mask และภาพก่อน/หลัง
- [lung_segmentation_utils.py](lung_segmentation_utils.py): ฟังก์ชัน segmentation/masking
- [generate_lung_mask_artifacts.py](generate_lung_mask_artifacts.py), [build_lung_mask_notebook.py](build_lung_mask_notebook.py): สร้างภาพและ notebook
- [output/lung_segmentation_stages.png](output/lung_segmentation_stages.png): ลำดับสร้าง mask
- [output/gradcam_before_vs_after_masking.png](output/gradcam_before_vs_after_masking.png): ตัวอย่าง Grad-CAM ก่อน/หลัง mask

## สถานะและการรัน

ภาพที่มีเป็นกรณีตัวอย่าง; ไม่มีตาราง IoU/Pointing Game หรือ CI ที่ยืนยันว่าการ mask ช่วยทุกกรณี. Mask อาจตัดรอยโรคที่ขอบปอด จึงต้องวัดกับ bbox ก่อนสรุป. ตั้ง working directory เป็นโฟลเดอร์นี้แล้วเปิด notebook หรือรัน python generate_lung_mask_artifacts.py.
