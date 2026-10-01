# การตรวจรับ Notebook 2

ตรวจเมื่อ 2 ตุลาคม 2026 บน CPU ของเครื่องนี้

- Unit/integration checks ผ่าน **17 รายการ** รวมข้อมูลจริง, การแยกคนไข้, bbox alignment, heatmap ศูนย์, SHAP ที่ไม่มีค่าบวก, signed Expected Gradients และ checkpoint ที่ผิด config/เงื่อนไข/manifest
- ตรวจ manifests ของค่าตั้งใหม่แล้ว: train **200 ภาพ / 195 คนไข้** (100 ภาพต่อคลาส), validation **50 ภาพ / 50 คนไข้** (25 ภาพต่อคลาส), official binary test **12,081 ภาพ / 2,589 คนไข้**, XAI **123 ภาพ / 115 คนไข้**, SHAP background **32 ภาพจาก train**
- ผลรอบใหม่นี้ใช้ `runs/train_100_per_class/` และเก็บ artifacts เดิมไว้; ตรวจค่าเริ่มต้นของเส้นทางและสร้างรายงานสถานะ not_run สำเร็จแล้ว
- ไม่พบภาพหรือคนไข้ XAI ปนใน train/validation และไม่พบคนไข้ซ้ำระหว่าง train/validation/test
- การตรวจ smoke ก่อนเปลี่ยนค่าตั้ง: ใช้ข้อมูล NIH จริง DAE 1 epoch, CNN **10 เงื่อนไข** อย่างละ 1 epoch, Grad-CAM **20 รายการ**, SHAP **20 รายการ** (ผลเดิมเก็บไว้ใน `smoke_runs/`; ยังไม่ได้รัน smoke หรือฝึกใหม่หลังเปลี่ยนค่าตั้ง)
- ตรวจ heatmap จริงทั้ง **40 รายการ**: ขนาดตรงภาพต้นฉบับ 1024×1024, finite, normalize [0,1] และคำนวณ IoU/Pointing Game ซ้ำแล้วตรงกับผลที่บันทึกไว้
- SHAP เก็บ signed attribution ราย channel และที่ขนาดต้นฉบับ; positive heatmap เป็นศูนย์ทุกตำแหน่งที่ signed attribution ไม่เป็นบวก
- โหลด CNN checkpoint กลับและตรวจคะแนนตรงกัน Confusion Matrix คำนวณจาก probabilities จริง
- รัน smoke ซ้ำสำเร็จครบสามเกณฑ์ โดย checkpoint, heatmap และ records **102 ไฟล์** มี hash และเวลาแก้ไขเดิม ไม่ฝึกหรือสร้าง attribution ใหม่
- Notebook ทั้งสามไฟล์ผ่าน nbformat validation และตรวจไวยากรณ์ทุก code cell

**งานฝึก/ประเมินเต็มบน Colab GPU ยังไม่ได้รัน** ตารางและ Confusion Matrix ใน README หลักจึงระบุ not_run ไม่มีค่าจำลอง ผล smoke อยู่ใน `smoke_runs/` และไม่ใช้สรุปงานวิจัย ผลเต็มจะสร้างเป็น 10 CNN checkpoints และ XAI 1,230 รายการต่อเกณฑ์เมื่อรันครบ

อ่าน [RUN_GUIDE.md](RUN_GUIDE.md) เพื่อเตรียมข้อมูลใน Drive และรัน CNN → Grad-CAM → SHAP แพ็กเกจ `Notebook_2_colab.zip` มี source และ notebook ครบ แต่ไม่รวมข้อมูล NIH, dependencies ทดสอบบน Windows หรือ artifacts ของการทดสอบนี้
