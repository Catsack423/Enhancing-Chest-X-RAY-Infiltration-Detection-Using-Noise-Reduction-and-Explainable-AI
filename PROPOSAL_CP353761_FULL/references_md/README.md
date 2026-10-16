# สรุปงานวิจัยอ้างอิง — โครงงาน CP353761

บันทึกสรุปจาก PDF ในโฟลเดอร์ `PROPOSAL_CP353761_FULL/` สำหรับใช้อ้างอิงตอนเขียนเค้าโครงและรายงาน ทุกไฟล์อ่านจาก PDF ฉบับเต็ม ตัวเลขทุกตัวมาจากตารางในเปเปอร์ (ระบุหน้า/ตารางไว้)

| # | ไฟล์สรุป | เปเปอร์ | อยู่ในเค้าโครงแล้ว? | ใช้สนับสนุน |
|---|---|---|---|---|
| 1 | [01_Wang2017_ChestX-ray8.md](01_Wang2017_ChestX-ray8.md) | Wang et al. (2017), CVPR | ใช่ (อ้างอิงข้อ 17) | ที่มาชุดข้อมูล, B-Box, เลือก ResNet-50, Infiltration AUC ต่ำ |
| 2 | [02_Chattopadhyay2022_Denoising_CXR.md](02_Chattopadhyay2022_Denoising_CXR.md) | Chattopadhyay (2022), *AI Evolution* | ใช่ (อ้างอิงข้อ 2) | Median filter บน CXR, Poisson noise |
| 3 | [03_Baltruschat2019_ResNet50_ChestXray14.md](03_Baltruschat2019_ResNet50_ChestXray14.md) | Baltruschat et al. (2019), *Sci. Rep.* | **ยังไม่มี** | ResNet-50 fine-tune บน ChestX-ray14, ผลแปรตาม split, Grad-CAM เจอ shortcut |
| 4 | [04_OakdenRayner2020_Public_Dataset_Labels.md](04_OakdenRayner2020_Public_Dataset_Labels.md) | Oakden-Rayner (2020), *Acad. Radiol.* | **ยังไม่มี** | ฉลาก NIH มี noise (Infiltration PPV 66%/36%), ผู้ป่วยถ่ายซ้ำ |
| 5 | [05_Majkowska2020_Adjudicated_CXR.md](05_Majkowska2020_Adjudicated_CXR.md) | Majkowska et al. (2020), *Radiology* | **ยังไม่มี** | ฉลาก adjudicated, NLP sensitivity ~45–50%, bootstrap CI |

**การวิเคราะห์ผลทดลองเทียบวรรณกรรม:** [06_ANALYSIS_v5_results_vs_literature.md](06_ANALYSIS_v5_results_vs_literature.md) — ผล v5 beta เทียบ 5 เปเปอร์ + พบว่า v4 ได้ AUC ≈ 0.5 บนภาพ PA และ metadata อย่างเดียวทำนายได้เท่าโมเดล

## ภาพรวมข้ามเปเปอร์

### 1. Infiltration เป็นโรคที่ยากที่สุดโรคหนึ่งใน ChestX-ray14

| แหล่ง | Setup | Infiltration AUC |
|---|---|---|
| Wang et al. (2017) | ResNet-50, 8 โรค, split สุ่มระดับภาพ | 0.6128 |
| Baltruschat et al. (2019) | ResNet-50 FT, 5-fold ระดับผู้ป่วย | 0.699 ± 0.006 |
| Baltruschat et al. (2019) | ResNet-38/50-large-meta, official split | 0.694 |

ทุกงานเป็น multi-label (14 โรค) งาน binary ของเรา (Infiltration vs No Finding ล้วน) น่าจะได้ AUC สูงกว่านี้เพราะตัดภาพโรคอื่นที่คล้ายกันออก แต่ตัวเลขข้างบนใช้อธิบายได้ว่าทำไมเพดานผลจึงไม่สูงมาก

### 2. ฉลากของ NIH มี noise มาก — เพดานผลของทุกงาน

| แหล่ง | หลักฐาน |
|---|---|
| Wang et al. (2017) | precision ฉลาก Infiltration จาก text mining = 0.74 (วัดบน OpenI) |
| Oakden-Rayner (2020) | รังสีแพทย์ดูภาพจริง: Infiltration PPV 66% (ผ่อนปรน) / 36% (แบบคลินิก); Normal PPV 76% / 62% |
| Majkowska et al. (2020) | ฉลาก NLP ของ ChestX-ray14 มี sensitivity 49.7% (pneumothorax), 45.4% (nodule/mass) เทียบฉลาก adjudicated |
| Baltruschat et al. (2019) | อ้าง label noise ~10% และต้องการ clean test set |

นี่คือเหตุผลที่การประเมิน XAI กับ **Bounding Box ที่รังสีแพทย์วาดจากภาพจริง** มีคุณค่า: เป็นหลักฐานอิสระจากฉลาก NLP

### 3. โมเดลเรียนรู้ shortcut ได้ — XAI ตรวจจับได้

- Baltruschat et al. (2019): Grad-CAM แสดงว่าโมเดล pneumothorax ดูที่ chest drain
- Oakden-Rayner (2020): 80% ของภาพ pneumothorax มี chest drain
- Majkowska et al. (2020): pneumothorax ที่ไม่มี chest tube โมเดลพลาดมากขึ้น (sensitivity −13.2% เทียบรังสีแพทย์บน ChestX-ray14)

### 4. วิธีวิทยาที่ควรยึดตาม

- **Patient-level split**: ผู้ป่วย 43% มีหลายภาพ คิดเป็น 84% ของข้อมูล (Oakden-Rayner); ผลแปรตาม split มาก (Baltruschat) → ยึด official split
- **Fine-tune ดีกว่า freeze**: AUC เฉลี่ย 0.730 → 0.819 (Baltruschat)
- **Bootstrap CI**: nonparametric 1,000 รอบระดับภาพ (Majkowska)
- **Localization หลาย threshold**: Wang ใช้ IoBB/IoU ที่ T = 0.1–0.9; heatmap 32×32 หยาบทำให้กล่องใหญ่กว่า GT

## สิ่งที่ควรแก้ในเค้าโครง (`PROPOSAL_CP353761_FULL.md`)

1. **Sec. 4.3 ข้อ Median Filter** — ประโยค *"Chattopadhyay (2022) ระบุว่าหากใช้ขนาด Kernel ที่ใหญ่เกินไปจะทำให้เกิด Over-smoothing"* **ไม่มีในเปเปอร์** (เปเปอร์ใช้ kernel 5×5 ค่าเดียว ไม่ได้พูดถึง over-smoothing) ควรแก้เป็นข้อค้นพบจริง (Median ได้ PSNR/MSE ดีที่สุด) และเขียนเรื่อง over-smoothing เป็นสมมติฐาน/ช่องว่างงานวิจัยของเราเอง หรือหาแหล่งอื่นมาอ้าง — ดู [02](02_Chattopadhyay2022_Denoising_CXR.md) หัวข้อ 4
2. **เพิ่มเปเปอร์ 3, 4, 5 ในรายการอ้างอิง** (APA อยู่ในแต่ละไฟล์) โดยเฉพาะใช้ใน Sec. 2 (Background) และ Sec. 6 (ข้อจำกัด)
3. **ตัวเลขชุดข้อมูล** — Sec. 5.1 ใช้ 112,120 ภาพ / 30,805 ผู้ป่วย (ChestX-ray14) แต่ Wang et al. (2017) ฉบับ CVPR รายงาน 108,948 ภาพ / 32,717 ผู้ป่วย / 8 โรค ถ้าจะอ้าง Wang สำหรับตัวเลข 112,120 ควรระบุว่าเป็นเวอร์ชันขยาย หรืออ้าง Baltruschat (2019) / Majkowska (2020) ซึ่งรายงานตัวเลข ChestX-ray14 ตรง
4. **คำว่า "ปอดอักเสบ" ในชื่อเรื่อง** — Oakden-Rayner (2020) ชี้ว่าฉลาก Infiltration / Consolidation / Pneumonia ใน NIH แยกกันแบบไม่มีหลักเกณฑ์ชัด และ Infiltration ไม่เท่ากับ pneumonia ควรนิยามในเค้าโครงว่า Infiltration = ฝ้าทึบในเนื้อปอด (airspace opacity) ที่ไม่จำเพาะสาเหตุ
5. **Sec. 6 ข้อจำกัด** — ควรเพิ่ม: label noise (ตัวเลขจากข้อ 2 ข้างบน), ภาพ 8-bit 255 ระดับเทาทำให้รายละเอียดบริเวณทึบหาย (Oakden-Rayner), ฉลาก Normal เองก็มี PPV เพียง 62–76%

## ข้อผิดพลาดในเปเปอร์ที่ไม่ควรอ้างต่อ

- Oakden-Rayner (2020): prevalence Infiltration "almost 10%" — ข้อมูลจริง ≈ 17.7% (19,894 / 112,120)
- Baltruschat et al. (2019) Table 3: หัวคอลัมน์เพศสลับกัน — จริงคือ ชาย 63,340 / หญิง 48,780 (ตรวจกับ `Data_Entry_2017.csv` แล้ว)
- Chattopadhyay (2022): ตัวเลขใน abstract/เนื้อความไม่ตรงกับตาราง และ Table 2 เป็นสถิติของดัชนีพิกเซล (รายละเอียดใน [02](02_Chattopadhyay2022_Denoising_CXR.md))
- Wang et al. (2017): เนื้อความเรียกตาราง AUC ว่า "Table 4" แต่จริงคือ Table 3
