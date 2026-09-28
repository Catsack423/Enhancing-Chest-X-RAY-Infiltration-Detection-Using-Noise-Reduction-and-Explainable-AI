# DenseNet121 เทียบ ResNet50 (งานเสริม)

สำรวจความต่างของ backbone สองแบบโดยดู Grad-CAM บน raw และ DAE+CLAHE. เป็นการเปรียบเทียบสถาปัตยกรรมเสริม; การทดลอง denoising หลักใน guideline กำหนดให้ใช้ backbone เดียวกันทุกวิธี

## ไฟล์ในโฟลเดอร์

- [densenet_vs_resnet_pipeline.ipynb](densenet_vs_resnet_pipeline.ipynb): ภาพเทียบใน notebook
- [densenet_utils.py](densenet_utils.py): utility ของ DenseNet/Grad-CAM
- [generate_densenet_artifacts.py](generate_densenet_artifacts.py), [build_densenet_notebook.py](build_densenet_notebook.py): สร้างภาพและ notebook
- [output/densenet_vs_resnet_gradcam_comparison.png](output/densenet_vs_resnet_gradcam_comparison.png): ภาพที่บันทึกไว้

## สถานะ

มีเพียงภาพตัวอย่าง ไม่มีตาราง classification metrics หรือ XAI IoU ที่ยืนยันว่า backbone ใดดีกว่า. การเปรียบเทียบกับ [classification/cnn](../cnn/README.md) ต้องควบคุมการฝึกและ test set ก่อน. ตั้ง working directory เป็นโฟลเดอร์นี้แล้วรัน python generate_densenet_artifacts.py หรือเปิด notebook.
