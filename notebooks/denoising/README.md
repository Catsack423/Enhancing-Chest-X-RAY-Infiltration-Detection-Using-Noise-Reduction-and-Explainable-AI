# Denoising methods

หมวดนี้เก็บโค้ดเตรียมภาพและผลตรวจว่าการลด noise หรือเพิ่ม contrast เปลี่ยนภาพและแผนที่ XAI อย่างไร. กลุ่มทดลองหลักตาม [guideline](../../guideline.md) คือ raw baseline, Median, CLAHE+DWT และ DAE+CLAHE; Gamma เป็นการทดลองเสริมด้าน contrast

| โมดูล | วิธี/ระดับ | ผลที่มี |
|---|---|---|
| [Median_CLAHE_DWT](Median_CLAHE_DWT/README.md) | raw; Median kernel 3/5/7; CLAHE+DWT preset L1–L3 | ภาพเทียบ 7 สภาวะ และ Grad-CAM |
| [DAE_CLAHE](DAE_CLAHE/README.md) | DAE ที่เทรนเอง + CLAHE clip 2/4/8 | checkpoint, ภาพ pipeline, ตาราง PSNR/SSIM/EPI/CIR 30 แถว |
| [Gamma_Correction](Gamma_Correction/README.md) | gamma 0.5/0.8/1.0/1.2/1.5 | ภาพเทียบและตารางความเข้ม/contrast ของภาพตัวอย่าง |

ใช้ [classification/cnn](../classification/cnn/README.md) สำหรับผลจำแนกที่ฝึก CNN จริง และ [xai/XAI_Evaluation](../xai/XAI_Evaluation/README.md) สำหรับตัวเลขตำแหน่ง Grad-CAM. ค่า strength ของ CLAHE+DWT เปลี่ยนหลายพารามิเตอร์พร้อมกัน จึงยังระบุสาเหตุของผลต่างรายพารามิเตอร์ไม่ได้.
