# Grad-CAM บนภาพดิบ

จุดเริ่มต้นของ XAI: เปรียบเทียบภาพ No Finding 100 ภาพกับ Infiltration 100 ภาพที่มี bbox. [sample_manifest_200.csv](sample_manifest_200.csv) มี 200 แถวจากคนไข้ 200 คน และระบุ path ของภาพต้นฉบับ; Infiltration ทั้ง 100 ภาพมี bbox. ชุดนี้เป็นตัวอย่างย่อย ไม่ใช่ bbox ทั้ง 123 ภาพที่ guideline กำหนดให้ประเมิน

## ไฟล์ในโฟลเดอร์

- [gradcam_normal_vs_infiltration.ipynb](gradcam_normal_vs_infiltration.ipynb): overlay Grad-CAM กับภาพดิบและ bbox
- [sample_manifest_200.csv](sample_manifest_200.csv): ภาพ, class, patient ID, path และ flag bbox
- [generate_preview.py](generate_preview.py): สร้าง [output/comparison_single_pair.png](output/comparison_single_pair.png) และ [output/batch_comparison_4x4.png](output/batch_comparison_4x4.png)
- [build_gradcam_notebook.py](build_gradcam_notebook.py): สร้าง notebook ใหม่

## วิธีอ่านผล

ภาพ overlay ใช้ target layer ResNet50 layer4[-1] และ bbox สีเขียวเป็นกรอบอ้างอิง. สคริปต์ใช้ ResNet50 น้ำหนัก ImageNet ไม่ใช่ classifier สองคลาสที่เทรนใน [classification/cnn](../../classification/cnn/README.md); จึงใช้ภาพเพื่อสำรวจ pipeline เท่านั้น. ตัวเลข Pointing Game/IoU ที่คำนวณแล้วอยู่ใน [XAI_Evaluation](../XAI_Evaluation/README.md)

ตั้ง working directory เป็นโฟลเดอร์นี้เมื่อเปิด notebook หรือรัน python generate_preview.py. ภาพต้นฉบับและ BBox_List_2017.csv อยู่ใน data/versions/3.
