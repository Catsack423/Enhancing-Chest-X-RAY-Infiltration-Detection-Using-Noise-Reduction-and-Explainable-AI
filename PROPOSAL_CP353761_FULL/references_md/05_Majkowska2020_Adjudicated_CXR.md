# Majkowska et al. (2020) — Chest Radiograph Interpretation with Deep Learning Models: Radiologist-adjudicated Reference Standards

> ไฟล์ต้นฉบับ: `majkowska-et-al-2019-chest-radiograph-interpretation-with-deep-learning-models-assessment-with-radiologist-adjudicated.pdf` (10 หน้า)
> สถานะในเค้าโครง: **ยังไม่อยู่ในรายการอ้างอิง** ของ `PROPOSAL_CP353761_FULL.md` — แนะนำให้เพิ่ม
> หมายเหตุชื่อไฟล์: ไฟล์ตั้งชื่อปี 2019 (ปีที่ตีพิมพ์ออนไลน์) แต่ฉบับวารสารคือ *Radiology* ก.พ. 2020 — อ้างเป็น 2020

## ข้อมูลบรรณานุกรม (APA 7)

Majkowska, A., Mittal, S., Steiner, D. F., Reicher, J. J., McKinney, S. M., Duggan, G. E., Eswaran, K., Chen, P.-H. C., Liu, Y., Kalidindi, S. R., Ding, A., Corrado, G. S., Tse, D., & Shetty, S. (2020). Chest radiograph interpretation with deep learning models: Assessment with radiologist-adjudicated reference standards and population-adjusted evaluation. *Radiology*, 294(2), 421–431. https://doi.org/10.1148/radiol.2019191293

- หน่วยงานหลัก: Google Health (ได้ทุนจาก Google; ผู้เขียนหลายคนเป็นพนักงาน/ถือหุ้น Google และมีสิทธิบัตรที่เกี่ยวข้อง)

## สรุปสั้น (TL;DR)

Google Health ฝึกโมเดล 4 ตัว (pneumothorax, **airspace opacity**, nodule/mass, fracture) จากภาพ ~760k ภาพของโรงพยาบาลในอินเดีย (DS1) + ChestX-ray14 แล้วประเมินกับ **ฉลากอ้างอิงที่รังสีแพทย์ 3 คนหารือกันจนได้ข้อยุติ (adjudication)** แทนฉลาก NLP โมเดลทำได้ระดับเดียวกับรังสีแพทย์ (ChestX-ray14: opacity AUC 0.94) จุดเด่นคือแสดงว่า **ฉลาก NLP ของ ChestX-ray14 มี sensitivity เพียง ~45–50%** เมื่อเทียบฉลาก adjudicated และ **เปิดเผยฉลาก adjudicated ของ ChestX-ray14 จำนวน 4,374 ภาพ** (val 2,412 + test 1,962)

## 1. ชุดข้อมูล (Table 1)

| | DS1 (อินเดีย) | ChestX-ray14 |
|---|---|---|
| ผู้ป่วย | 538,390 | 30,805 |
| ภาพ | 759,611 | 112,120 |
| อายุมัธยฐาน | 50 | 49 |
| ชาย | 332,184 (62%) | 16,630 (54.0%) *(นับระดับผู้ป่วย)* |
| AP / PA | 17.6% / 82.4% | 39.9% / 60.0% |
| Test set สุดท้าย | 1,818 | 1,962 |
| Opacity บวกใน test | 444 (24.4%) | **1,135 (57.8%)** |
| Pneumothorax บวก | 88 (4.8%) | 195 (9.9%) |
| Nodule/mass บวก | 322 (17.7%) | 295 (15.0%) |
| Fracture บวก | 257 (14.1%) | 72 (3.7%) |

- DS1: ภาพต่อเนื่องทั้งหมดจาก 5 ศูนย์ในอินเดีย (Bangalore, Bhubaneswar, Chennai, Hyderabad, New Delhi) พ.ย. 2010 – ม.ค. 2018 → สะท้อน prevalence จริงของประชากร
- ChestX-ray14 ถูกระบุว่า **enriched** (มีภาพผิดปกติมากกว่าประชากรทั่วไป)
- ChestX-ray14 split: **คง official test set 25,596 ภาพ / 2,797 ผู้ป่วย** ส่วนที่เหลือ 86,524 ภาพ / 28,008 ผู้ป่วย แบ่งสุ่มเป็น train 68,801 / val 17,723 โดยผู้ป่วยหนึ่งคนอยู่ split เดียว — ตัวเลขตรงกับ `test_list.txt` (25,596) และ `train_val_list.txt` (86,524) ของเรา
- Test set เลือกภาพสุ่มประมาณ 2,000 ภาพจาก official test ของ ChestX-ray14 แล้วตัดภาพที่มีปัญหาทางเทคนิคออก

## 2. ฉลากอ้างอิง (Reference Standard)

- นิยามโรคตาม **Fleischner Society Glossary of Terms for Thoracic Imaging** (nodule < 3 cm, mass ≥ 3 cm; fracture = ซี่โครง กระดูกไหปลาร้า ต้นแขน หรือกระดูกสันหลังที่มองเห็น)
- **Test set**: ภาพละ 3 คน สุ่มจากรังสีแพทย์ board-certified 11 คน (ประสบการณ์ 3–21 ปี ไม่มีผู้เชี่ยวชาญทรวงอกเฉพาะทาง) อ่านอิสระก่อน แล้วหารือแบบไม่ระบุตัวตนแบบ asynchronous สูงสุด 5 รอบ ไม่บังคับให้เห็นตรงกัน ถ้าไม่ได้ข้อยุติใช้ majority vote
- **Validation set**: ภาพละ 3 คนจากกลุ่ม 13 คน (รังสีแพทย์ + แพทย์ประจำบ้าน)
- ผู้อ่านเห็นเฉพาะอายุและ view (PA/AP) ไม่มีข้อมูลทางคลินิกอื่น
- ฉลาก "hedge" (ไม่แน่ใจ) นับเป็นบวก
- Adjudication ทำให้ความเห็นตรงกันเพิ่มจาก **41.8% (หลังอ่านรอบแรก) เป็น 96.8%**
- **Training labels** ของ DS1: ผสมการติดฉลากจากภาพโดยผู้เชี่ยวชาญ + NLP (1D CNN ฝึกจากรายงาน ~35,000 ฉบับ)

### ฉลาก NLP ของ ChestX-ray14 vs ฉลาก adjudicated (Table 2)

| | Pneumothorax | Nodule/Mass |
|---|---|---|
| NLP sensitivity | **49.7%** | **45.4%** |
| NLP specificity | 94.2% | 93.8% |

(airspace opacity และ fracture ไม่มีฉลาก NLP ที่เทียบได้ตรง ๆ จึงไม่ได้เปรียบเทียบ)

ใน DS1 แม้ majority vote รอบแรกก็พลาด: sensitivity pneumothorax 87.5%, nodule/mass 78.6%, fracture 62.6%, opacity 92.0% — ฉลาก adjudicated ไวกว่าวิธีอื่นอย่างสม่ำเสมอ

## 3. โมเดลและสถิติ

- 4 โมเดลแยกโรค (binary) สถาปัตยกรรม **Xception** pretrain บนภาพธรรมชาติ 300 ล้านภาพ (JFT) ฝึกด้วย train ของ DS1 + ChestX-ray14 รวมกัน
- Ensemble หลายโมเดล เลือกด้วย AUPRC บน validation แล้วเฉลี่ยผลทำนาย
- **95% CI ด้วย nonparametric bootstrap 1,000 รอบ ที่ระดับภาพ**
- เทียบกับรังสีแพทย์อีก 4 คน (ไม่ซ้ำกับผู้ทำฉลากอ้างอิง) ด้วยวิธี Obuchowski-Rockette-Hillis, noninferiority margin 5%
- เปรียบเทียบที่ 2 operating point: sensitivity เฉลี่ยของรังสีแพทย์ และ specificity เฉลี่ยของรังสีแพทย์
- **Inverse probability weighting** บน DS1 เพื่อปรับผลจากการ enrich ให้เป็นระดับประชากร (ทำกับ ChestX-ray14 ไม่ได้เพราะไม่รู้ระดับการ enrich)
- XAI: ใช้ **SmoothGrad** แสดงบริเวณที่มีอิทธิพลต่อการทำนาย (Fig. 5)

## 4. ผลลัพธ์

### AUC

| Finding | ChestX-ray14 (95% CI) | DS1 population-adjusted (95% CI) |
|---|---|---|
| Pneumothorax | 0.94 (0.93, 0.96) | 0.95 (0.91, 0.99) |
| Nodule or mass | 0.91 (0.89, 0.93) | 0.72 (0.66, 0.77) |
| **Airspace opacity** | **0.94 (0.93, 0.95)** | 0.91 (0.88, 0.93) |
| Fracture | 0.81 (0.75, 0.86) | 0.86 (0.79, 0.92) |

### Airspace opacity บน ChestX-ray14 (Table 3 — ค่า %, 95% CI)

| ตัวชี้วัด | โมเดล | รังสีแพทย์เฉลี่ย | ต่าง (โมเดล − รังสีแพทย์) |
|---|---|---|---|
| Sensitivity (ที่ specificity เฉลี่ยรังสีแพทย์) | 86.1 (83.3, 89.3) | 81.6 (80.1, 83.1) | +4.4 (−9.7, 18.6) |
| Specificity (ที่ sensitivity เฉลี่ยรังสีแพทย์) | 89.7 (87.3, 92.4) | 85.8 (84.3, 87.4) | +3.9 |
| PPV (ที่ sensitivity เฉลี่ยรังสีแพทย์) | 91.6 (89.8, 93.6) | 88.8 (87.7, 89.9) | +2.8 (1.0, 4.9) |

(หมายเหตุ: ใน PDF เครื่องหมายลบในตารางนี้ถูกแปลงเป็นเลข "2" เวลา extract — ค่าข้างบนตีความแล้ว ควรตรวจกับ PDF ต้นฉบับก่อนอ้าง)

### ข้อค้นพบอื่น

- True positive ที่พบโดยโมเดลหรือรังสีแพทย์เพียงฝ่ายเดียว: 25.1% (ChestX-ray14), 43.7% (DS1) → โมเดลกับคนผิดพลาดคนละแบบ
- มี 42 ภาพที่โมเดลพบรอยโรคแต่รังสีแพทย์ทั้ง 4 คนพลาด
- ข้อผิดพลาดของโมเดลที่พบบ่อย: เข้าใจ granuloma เป็น nodule และภาพก้ำกึ่ง (borderline)
- **Pneumothorax ที่ไม่มี chest tube** (untreated): โมเดลมี sensitivity ต่ำกว่ารังสีแพทย์ 2.5% (DS1) และ **13.2% (ChestX-ray14)** — สอดคล้องกับปัญหา shortcut จาก chest drain
- ฉลากอ้างอิงต่างแบบ ทำให้ AUC ต่างได้ถึง 0.05 (ChestX-ray14) และ 0.04 (DS1)
- แม้รังสีแพทย์มี specificity > 98% ทุก finding แต่ **PPV ระดับประชากรเพียง 42–85%** → ควรรายงานตัวชี้วัดที่ขึ้นกับ prevalence ด้วย

## 5. ข้อจำกัดที่เปเปอร์ยอมรับ

- ไม่มี CT ทรวงอกในช่วงเวลาเดียวกันเพื่อยืนยันผล
- การ enrich ข้อมูลไม่สะท้อน prevalence จริง จึงใช้ IPW ชดเชย (ทำได้เฉพาะ DS1)
- การวิเคราะห์ noninferiority เป็นแบบ exploratory ไม่ได้อ้างนัยสำคัญ

## 6. ความเกี่ยวข้องกับโครงงานเรา

| ประเด็นในเค้าโครง | ใช้เปเปอร์นี้อ้างอย่างไร |
|---|---|
| ฉลาก NIH มี noise | ฉลาก NLP มี sensitivity ~45–50% เทียบฉลาก adjudicated (pneumothorax, nodule/mass) → สนับสนุน Oakden-Rayner (2020) ด้วยข้อมูลเชิงปริมาณ |
| Infiltration ≈ airspace opacity | finding ที่ใกล้ Infiltration ที่สุดคือ "airspace opacity" ตามนิยาม Fleischner — ใช้ประกอบการนิยาม Infiltration ในเค้าโครง |
| Official patient-level split | ยืนยันจำนวน official test 25,596 ภาพ / 2,797 ผู้ป่วย และยึด test set เดิม |
| 95% bootstrap CI | ใช้ nonparametric bootstrap 1,000 รอบระดับภาพ — แนวเดียวกับที่เราจะใช้กับ IoU / Pointing Game |
| Threshold / operating point | เทียบที่ operating point ที่กำหนดจากข้อมูลอื่น ไม่ใช่เลือกให้ผลดีที่สุดบน test — สอดคล้องกับการเลือก threshold จาก validation ของเรา |
| ทางเลือกต่อยอด | ฉลาก adjudicated ของ Google (opacity) สำหรับ test set ของ ChestX-ray14 ใช้เป็น **clean test set เสริม** ได้ (https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest#additional_labels) |
| XAI | ใช้ SmoothGrad อธิบายผลเชิงภาพเท่านั้น (qualitative) ไม่ได้วัด localization เชิงปริมาณ → ช่องว่างที่งานเราเติม |
| Shortcut | pneumothorax ที่ไม่มี chest tube ตรวจยากกว่ามาก (−13.2% sensitivity บน ChestX-ray14) |

## ประโยคที่อ้างตรงได้

- "...because of errors or inconsistencies in the resulting labels, such approaches may lead to overestimation of model performance." (p. 421)
- "Adjudication increased the overall consensus from 41.8% (1580 of 3780 images) after the initial read to 96.8% (3660 of 3780 images)." (p. 428)
- "...although radiologist specificity was greater than 98% across all four findings, population-adjusted positive predictive values ranged from 42% to 85%..." (p. 428)
- "Radiologist-adjudicated labels for 2412 ChestX-ray14 validation set images and 1962 test set images are provided." (Abstract)
