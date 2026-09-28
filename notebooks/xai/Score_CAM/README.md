# Score-CAM (งาน XAI เสริม)

Score-CAM ใช้ activation map ทำ mask แล้ววัด class score ด้วย forward pass เพื่อเปรียบเทียบกับ Grad-CAM. เป็นวิธี XAI เสริมจาก Grad-CAM/SHAP ใน guideline

## ไฟล์ในโฟลเดอร์

- [score_cam_pipeline.ipynb](score_cam_pipeline.ipynb): ตัวอย่าง Score-CAM เทียบ Grad-CAM
- [score_cam_utils.py](score_cam_utils.py): ScoreCAMGenerator
- [generate_score_cam_artifacts.py](generate_score_cam_artifacts.py), [build_score_cam_notebook.py](build_score_cam_notebook.py): สร้างภาพ/ตารางและ notebook
- [output/score_cam_vs_gradcam_comparison.png](output/score_cam_vs_gradcam_comparison.png): ภาพที่บันทึกไว้

## สถานะ

ภาพที่มีอยู่เป็นตัวอย่างการชี้ตำแหน่งด้วย pretrained ResNet50 และ bbox; ไม่มี CSV ผลเชิงปริมาณอยู่ในโฟลเดอร์นี้ในขณะนี้ แม้สคริปต์รองรับการสร้างผลเพิ่ม. ยังไม่มีผล 123 bbox หรือ bootstrap CI. ตั้ง working directory เป็นโฟลเดอร์นี้แล้วรัน python generate_score_cam_artifacts.py หากต้องการสร้างผลใหม่.
