# ชุดภาพทดลองสำหรับ Teachable Machine

ชุดนำร่องจาก NIH ChestX-ray14: ฝึกคลาสละ 200 ภาพ ทดสอบคลาสละ 50 ภาพ
รวมภาพต้นทาง 500 ภาพ จากคนไข้ 500 คน ใช้ภาพต้นทางเหมือนกันทุกวิธี
Normal หมายถึง label `No Finding` เท่านั้น; Infiltration หมายถึง label `Infiltration` เดี่ยว
ไม่ได้เป็นชุดข้อมูลจำแนกสาเหตุ และไม่มีคลาส COVID-19

## วิธีใช้

1. เริ่มด้วยโฟลเดอร์ `baseline` หรือแตกไฟล์ `baseline.zip` ก่อน
2. สร้างโปรเจกต์ภาพ ตั้งชื่อสองคลาสเป็น `Normal` และ `Infiltration`
3. อัปโหลดเฉพาะภาพใน `train/Normal` และ `train/Infiltration` ให้ตรงคลาส
4. ฝึกผ่านหน้าเว็บ โดยใช้ค่า epoch, batch size และ learning rate เดียวกันทุกการทดลอง
5. เก็บภาพใน `test` ไว้ทดสอบหลังฝึก ห้ามนำไปเพิ่มในคลาสสำหรับฝึก
6. บันทึกผล แล้วสร้างโปรเจกต์แยกสำหรับวิธีถัดไป ใช้สองคลาสเดิม

อย่าใช้ Baseline/Median/CLAHE เป็นชื่อคลาส เพราะจะกลายเป็นการจำแนกวิธีแต่งภาพ
อย่ารวมภาพต่างวิธีหรือระดับเข้าด้วยกันในการทดลองเดียว
ZIP มีไว้ขนย้ายไฟล์ ให้แตก ZIP แล้วเลือก PNG ในโฟลเดอร์คลาส

## ชุดทดลอง 7 ชุด

| โฟลเดอร์ | พารามิเตอร์ |
|---|---|
| baseline | ไม่ denoise; มีการย่อภาพเช่นเดียวกับทุกชุด |
| median_L1 | kernel 3 |
| median_L2 | kernel 5 |
| median_L3 | kernel 7 |
| clahe_dwt_L1 | clip limit 2, DWT depth 1, threshold multiplier 0.5 |
| clahe_dwt_L2 | clip limit 4, DWT depth 2, threshold multiplier 1.0 |
| clahe_dwt_L3 | clip limit 8, DWT depth 3, threshold multiplier 2.0 |

เริ่มเทียบ baseline, median_L2, clahe_dwt_L2 ก่อน แล้วจึงทดลองระดับอื่น
CLAHE+DWT ใช้ implementation เดิม: DWT ก่อน แล้วจึง CLAHE, wavelet db1, grid 8x8
ระดับของ fusion เปลี่ยนหลายพารามิเตอร์พร้อมกัน จึงเป็นการทดลอง preset ร่วม
ยังใช้สรุปไม่ได้ว่าพารามิเตอร์ใดทำให้ผลเปลี่ยน และ CLAHE เป็นการเพิ่ม contrast
ระดับที่สูงกว่าจึงไม่ได้รับประกันว่า noise จะลดมากกว่าเสมอ

## รายละเอียดและข้อจำกัดของชุดนำร่อง

- ประมวลผล grayscale ที่ขนาดต้นฉบับ แล้วจึงย่อเป็น 256x256 ด้วย INTER_AREA
- บันทึก PNG แบบ RGB โดยทั้งสาม channel เหมือนกัน ไม่เติม noise สังเคราะห์
- สุ่ม seed 42 เลือกไม่เกินหนึ่งภาพต่อคนไข้ ไม่ซ้ำข้ามคลาสหรือ train/test
- ใช้ membership ของ official train_val_list/test_list เดิม ไม่สลับ split
- `selection.csv` บันทึกภาพต้นทาง คนไข้ คลาส และ split; `config.json` บันทึกค่าประมวลผล
- `before_after.png` แสดงตัวอย่างฝึก 4 ภาพ (สองภาพต่อคลาส) ในทุกวิธี
- ตรวจ label จาก metadata ต้นฉบับ ตรวจ official split และตรวจไฟล์ PNG ทุกภาพหลังบันทึก
- ไม่มีการฝึก DAE หรือ classifier ด้วยสคริปต์นี้
- เป็นชุดขนาดเล็กและสมดุลสำหรับทดลอง ไม่แทนผลวิจัยบนข้อมูลเต็มหรือสัดส่วนจริง
- การเปลี่ยนมาใช้ Teachable Machine เป็นการเปลี่ยน workflow จาก ResNet50/XAI เดิม
  ผลชุดนี้ยังไม่ตอบการประเมิน XAI IoU และไม่ควรนำไปปนกับผล protocol เดิม
- ใช้ผลบน test เพื่อรายงานการประเมิน แยกจากผล validation ที่เว็บแสดงระหว่างฝึก

สคริปต์เตรียมภาพ: [prepare_teachable_images.py](../../../scripts/prepare_teachable_images.py) (ป้องกันการเขียนทับผลเดิม)
