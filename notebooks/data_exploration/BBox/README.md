# สำรวจ Bounding Box ของ NIH

ใช้ [BBox_List_2017.csv](../../../data/versions/3/BBox_List_2017.csv) เพื่อดูจำนวนกรอบ, ภาพที่มีหลายกรอบ และตัวอย่างโรค 8 กลุ่ม. ส่วนนี้เป็นการตรวจรูปแบบข้อมูล ไม่ใช่การประเมินตำแหน่ง XAI

## ไฟล์ในโฟลเดอร์

- [bbox_visualization.ipynb](bbox_visualization.ipynb): สำรวจ metadata และวาดกรอบบนภาพ
- [generate_notebook.py](generate_notebook.py): สร้าง notebook ใหม่
- [run_bbox_preview.py](run_bbox_preview.py): สร้างภาพตัวอย่าง
- [output/sample_single.png](output/sample_single.png), [output/sample_multi_bbox.png](output/sample_multi_bbox.png), [output/sample_8_diseases_grid.png](output/sample_8_diseases_grid.png): กรอบเดี่ยว, หลายกรอบ และตัวอย่างแต่ละโรค

ข้อมูลต้นฉบับอยู่ใน data/versions/3 ไม่ได้คัดลอกไว้ในโฟลเดอร์นี้. ตั้ง working directory เป็นโฟลเดอร์นี้หรือราก repo เมื่อเปิด notebook; สคริปต์ python run_bbox_preview.py ใช้ path จากตำแหน่งไฟล์.
