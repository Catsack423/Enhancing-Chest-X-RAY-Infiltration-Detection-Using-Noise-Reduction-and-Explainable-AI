# การตรวจรับ Notebook 2 — epoch 320 ภาพ / test subset 100 ภาพ

ตรวจเมื่อ 2 ตุลาคม 2026 บน CPU ของเครื่องนี้

- Unit/integration checks ผ่าน **24 รายการ** รวม balanced sampling, train/validation/test patient leakage, native bbox alignment, empty heatmaps, signed SHAP, checkpoint compatibility, ZIP integrity และ runtime preprocessing cache
- Train pool **320 ภาพ / 311 คนไข้**, validation **80 ภาพ / 80 คนไข้**, balanced official test subset **100 ภาพ / 90 คนไข้**, XAI **123 ภาพ / 115 คนไข้**, SHAP background **32 ภาพจาก train**
- ทดสอบ training loader แล้วว่า epoch หลักใช้ **320 ภาพรวมไม่ซ้ำ** (160 ภาพต่อคลาส) ใช้ train ทั้งชุดทุก epoch และทุกวิธี สับลำดับด้วย seed และหมายเลข epochและบันทึก epoch manifest เพื่อรันต่อได้
- ไม่พบภาพหรือคนไข้ XAI ปนใน train/validation และไม่พบคนไข้ซ้ำระหว่าง train/validation/test
- ชุด Colab ใหม่มี **622 ภาพไม่ซ้ำ** ขนาดประมาณ **67 MB** ตรวจ working pixels หลัง encode/reload ครบทุกภาพว่าตรงกับขั้นเตรียมภาพต้นฉบับ; เก็บ XAI 123 ไฟล์ตรงต้นฉบับทุก byte และขนาด 1024×1024
- ตรวจ ZIP ที่ส่งจริง: จำลองเซลล์ bootstrap บนเครื่องนี้ แตกโค้ดและข้อมูลจริง ตรวจ SHA-256 ทุกไฟล์ เปรียบเทียบ manifests ทั้งโหมดหลักและ smoke แล้วตรงกับ NIH ต้นทาง และตรวจ cache ใช้ซ้ำสำเร็จ
- Smoke ใช้ข้อมูล NIH จริง train/validation คลาสละ 2, test คลาสละ 2, XAI 2, DAE/CNN 1 epoch และ SHAP nsamples=4; ผลนี้ใช้ตรวจ pipeline เท่านั้น
- การตรวจ history ของ DAE และ CNN ทั้ง 10 เงื่อนไขยืนยันว่าใช้ IDs เดียวกันและ **4 ภาพต่อ epoch ใน smoke** ตามค่า override ไม่ใช้คะแนน smoke เป็นผลวิจัย
- Smoke pipeline ที่รันก่อนปรับจำนวนภาพผ่านครบ: **10 CNN checkpoints, 20 Grad-CAM records และ 20 SHAP records** รวมตรวจ checkpoint round-trip และความครบถ้วนของรายงานแล้ว
- Notebook ทั้งสามผ่าน nbformat validation และตรวจไวยากรณ์ code cells; เซลล์ Colab ล้างโมดูล shared รุ่นเก่าก่อนโหลดโค้ดจาก ZIP
- Cache ภาพหลัง denoising บน Colab อยู่บน runtime disk; ทดสอบ pixel equality, ตำแหน่ง cache และการอ่านซ้ำโดยไม่เขียนไฟล์ใหม่แล้ว

**งานหลัก epoch ละ 320 ภาพ และ XAI ทั้ง 123 ภาพบน Colab GPU ยังไม่ได้รัน** README หลักแสดง not_run ไม่มีตัวเลขจำลอง ผล classification รอบนี้เป็น test subset 100 ภาพ ไม่ใช่ official binary test เต็ม 12,081 ภาพ

ผลรอบใหม่อยู่ใน `runs/pilot320_val80_test100/` แยกจากผลเดิม ส่วนผล smoke ก่อนปรับจำนวนภาพยังอยู่ใน `runs/epoch50_test100/smoke_runs/` และยังไม่ได้ฝึกใหม่ในโฟลเดอร์รอบ 320 ภาพ

อ่าน [RUN_GUIDE.md](RUN_GUIDE.md) แล้วแทนที่ ZIP รุ่นเก่าบน Drive ด้วย ZIP รุ่นใหม่ก่อนเปิด notebook รุ่นใหม่
