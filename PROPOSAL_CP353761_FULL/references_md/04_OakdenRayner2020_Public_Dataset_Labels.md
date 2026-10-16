# Oakden-Rayner (2020) — Exploring Large-scale Public Medical Image Datasets

> ไฟล์ต้นฉบับ: `Oakden-Rayner (2020) — Exploring Large-scale Public Medical Image Datasets.pdf` (9 หน้า) — **เป็นฉบับ preprint arXiv:1907.12720v1 (30 Jul 2019)**
> สถานะในเค้าโครง: **ยังไม่อยู่ในรายการอ้างอิง** ของ `PROPOSAL_CP353761_FULL.md` — แนะนำให้เพิ่ม (สำคัญมากสำหรับส่วนข้อจำกัด)

## ข้อมูลบรรณานุกรม (APA 7)

ฉบับตีพิมพ์ในวารสาร (ใช้ตัวนี้ในรายการอ้างอิง):

Oakden-Rayner, L. (2020). Exploring large-scale public medical image datasets. *Academic Radiology*, 27(1), 106–112. https://doi.org/10.1016/j.acra.2019.10.006

ฉบับที่อยู่ในโฟลเดอร์ (preprint):

Oakden-Rayner, L. (2019). *Exploring large scale public medical image datasets* (arXiv:1907.12720). arXiv. https://doi.org/10.48550/arXiv.1907.12720

- ผู้เขียน: รังสีแพทย์ board-certified, Australian Institute for Machine Learning / University of Adelaide / Royal Adelaide Hospital
- หมายเหตุ: เลขหน้าที่อ้างใน note นี้เป็นเลขหน้าของ preprint — ก่อนอ้างตรงในรายงานควรตรวจเลขหน้ากับฉบับวารสาร

## สรุปสั้น (TL;DR)

รังสีแพทย์ดูภาพจริงจาก ChestX-ray14 (CXR14) 700 ภาพ (50 ภาพ/คลาส × 14 โรค + Normal) แล้วเทียบกับฉลากที่ได้จาก NLP พบว่า **PPV ของฉลากต่ำกว่าที่เอกสาร NIH อ้างไว้ 10–30%** ในแทบทุกคลาส — **Infiltration PPV เพียง 66% (เกณฑ์ผ่อนปรน) และ 36% (เกณฑ์แบบคลินิก)** พร้อมพบปัญหา hidden stratification (pneumothorax ส่วนใหญ่มี chest drain) และ label disambiguation failure (emphysema ส่วนใหญ่เป็น subcutaneous emphysema) ส่วน MURA ฉลากแม่นกว่า ยกเว้นโรคข้อเสื่อม

## 1. ปัญหาหลักของชุดข้อมูลสาธารณะ (Introduction)

ต้นเหตุคือ **การสร้างข้อมูลแยกขาดจากการใช้ข้อมูล** ผู้ใช้ปลายทางไม่รู้รายละเอียดการติดฉลาก ทำให้เกิด 3 ปัญหา:
1. ผู้ใช้ประเมินความแม่นของฉลากสูงเกินจริง
2. มีกลุ่มย่อยที่มองเห็นได้แต่ไม่ได้ติดฉลาก (**hidden stratification**) ซึ่งเปลี่ยนสิ่งที่โมเดลเรียนรู้
3. ความหมายทางคลินิกของฉลากคลุมเครือ

และถ้าทดสอบบน test set ที่ติดฉลากด้วยวิธีเดียวกัน ปัญหาจะ **"fail silently"** — ผลดูดีเพราะโมเดลเรียนรู้ฉลากที่ผิดเหมือนกัน แต่ใช้จริงทางคลินิกได้ไม่ดี

## 2. ข้อมูล CXR14 ที่เปเปอร์บันทึกไว้

- 112,120 ภาพ จาก 30,805 ผู้ป่วย ศูนย์เดียว (NIH Clinical Center) รวมทั้งผู้ป่วย ICU และไม่ใช่ ICU
- ต้นฉบับ DICOM ~3000×2000 พิกเซล มีระดับเทา ~3000–4000 ระดับ → แปลงเป็น PNG **1024×1024, 255 ระดับเทา**
- ฉลากเริ่มต้น 8 คลาส ขยายเป็น 14 คลาสภายในปี 2017
- "No finding" ประมาณ 75% (~84,000 ภาพ)
- เปเปอร์เขียนว่า Infiltration มี prevalence "almost 10%" — แต่ข้อมูลจริงคือ 19,894/112,120 ≈ **17.7%** (ดู Baltruschat et al., 2019; ตรวจได้จาก `Data_Entry_2017.csv`) ตัวเลขนี้ไม่ควรอ้างต่อ

### ผู้ป่วยถ่ายซ้ำหลายครั้ง (Table 1)

| จำนวนภาพต่อผู้ป่วย | จำนวนผู้ป่วย (% ผู้ป่วย) | จำนวนภาพรวม (% ภาพ) |
|---|---|---|
| > 1 | 13,302 (43%) | 94,617 (84%) |
| > 5 | 4,821 (16%) | 70,081 (63%) |
| > 10 | 2,225 (7%) | 50,468 (45%) |
| > 50 | 151 (0.5%) | 10,812 (10%) |
| > 100 | 18 | 2,310 (2%) |

ผู้ป่วยถ่ายซ้ำ (เช่นผู้ป่วย ICU ถ่ายทุกวัน) ทำให้ความหลากหลายของข้อมูลต่ำ และเป็นเหตุผลหลักที่ **ต้องแบ่งข้อมูลระดับผู้ป่วย** ไม่เช่นนั้นภาพเกือบซ้ำของคนเดียวกันจะรั่วข้าม train/test

## 3. วิธีการตรวจฉลาก (Sec. 2.2–2.3)

- ดูภาพแบบสำรวจราว 100 ภาพต่อคลาสก่อน (แยกจากชุดที่ใช้ relabel) เพื่อเข้าใจ schema
- CXR14: relabel ทั้งชุดตาม prevalence จริงทำไม่ได้ เพราะหลายคลาสหายากและคลาสคลุมเครือ — **pneumonia, consolidation, infiltration ถูกแยกเป็นคนละคลาส แต่ทางคลินิกแทบแยกกันไม่ได้หากไม่มีข้อมูลทางคลินิก**
- จึงสุ่ม 50 ภาพ/คลาส รวม 700 ภาพ และตรวจเฉพาะว่าภาพมีโรคตามฉลากที่ได้รับหรือไม่ (เช่น "cardiomegaly" vs "not cardiomegaly")
- ใช้ 2 เกณฑ์:
  - **Permissive**: ถ้าภาพ "อาจ" แสดงโรคนั้นได้ ให้นับเป็นบวก (เช่น ฝ้าขอบไม่ชัดที่ฐานปอดนับเป็น effusion หรือ consolidation ได้ แต่ไม่นับเป็น mass) — เอียงเข้าหาฉลากเดิมให้มากที่สุด
  - **Clinical**: ใกล้เคียงการรายงานในเวชปฏิบัติจริง
- วัดเฉพาะ **PPV** (ไม่ได้วัด NPV เพราะไม่ได้ relabel ภาพลบทั้งหมด)

## 4. ผลลัพธ์ CXR14 (Table 2)

| คลาส | PPV (ดูภาพ, permissive) | PPV (ดูภาพ, clinical) | PPV text mining (Wang et al.) |
|---|---|---|---|
| Consolidation | 80% | 66% | – |
| Atelectasis | 80% | 50% | 99% |
| **Infiltration** | **66%** | **36%** | 74% |
| Pneumonia | 60% | 50% | 66% |
| Oedema | 76% | 40% | – |
| Nodule | 76% | 64% | 96% |
| Mass | 64% | 46% | 75% |
| Pneumothorax | 90% (60%)* | 90% (60%)* | 90% |
| Pleural effusion | 74% | 70% | 93% |
| Pleural thickening | 84% | 52% | – |
| Emphysema | 14% | 10% | – |
| Cardiomegaly | 70% | 52% | 100% |
| Fibrosis | 46% | 26% | – |
| Hernia | 94% | 78% | – |
| **Normal** | **76%** | **62%** | 87% |

\* ในวงเล็บคือ PPV ของกลุ่มที่ไม่มี chest drain

- PPV จากการดูภาพ **ต่ำกว่าค่าที่ Wang et al. รายงานในทุกคลาส** แม้ใช้เกณฑ์ผ่อนปรน
- PPV ของ Wang et al. วัดบนชุด OpenI ไม่ใช่บน CXR14
- Hidden stratification: **80% ของ pneumothorax มี chest drain** (รักษาแล้ว ปอดไม่แฟบ)
- Label disambiguation failure: **86% ของ emphysema เป็น subcutaneous emphysema** ไม่ใช่ pulmonary emphysema
- Hernia ฉลากบวกเกือบถูกทั้งหมด แต่มี false negative จำนวนมาก

## 5. ผลลัพธ์ MURA (สรุปย่อ)

- relabel 714 ภาพตาม prevalence จริง (abnormal:normal ≈ 45:55) โดยไม่รู้ฉลากเดิม
- ภาพรวม sensitivity 80%, specificity 75%
- Degenerative joint disease: sensitivity **60%**, specificity 82% (fracture 92%/98%, hardware 85%/98%)
- สาเหตุที่น่าจะเป็น: รังสีแพทย์ตีความ "normal" ว่า "normal for age"

## 6. ประเด็นอภิปรายสำคัญ (Sec. 4)

- **รายงานรังสีแพทย์ไม่ใช่คำบรรยายภาพที่ครบถ้วน** — หลายสิ่งไม่ถูกเขียน เช่น รายงานที่เขียนเพียง "no change compared to previous" หรือสิ่งที่แพทย์ผู้ส่งตรวจไม่จำเป็นต้องรู้ ดังนั้น NLP ที่สมบูรณ์แบบก็ยังได้ฉลากที่ไม่ตรงภาพ
- **ฉลาก airspace opacity (infiltration / consolidation / pneumonia) คลุมเครือมาก** จนแทบออกแบบวิธี relabel ไม่ได้ — "the differences between these labels may be mostly arbitrary"
- ฉลาก pneumothorax: "What can we reasonably expect models to learn from these labels, other than the appearance of chest tubes?"
- เอกสาร NIH อ้างว่าฉลากแม่น >90% ซึ่ง **ทำให้เข้าใจผิด**
- ข้อเสนอ: ทำ test set แยกที่ติดฉลากด้วยการดูภาพโดยผู้เชี่ยวชาญ เพื่อป้องกันโมเดล "fail silently"
- CheXNet (Rajpurkar et al., 2017) สร้างฉลากดูภาพเอง และพบว่าฉลากเดิมได้ F1 0.288 เทียบรังสีแพทย์ 0.35–0.44 (แต่ฉลากชุดนั้นไม่เปิดสาธารณะ)

### ข้อจำกัดที่ผู้เขียนยอมรับ

- ฉลากของผู้เขียนเองก็ไม่แม่น 100% (inter-observer variability)
- **การลดระดับเทาเหลือ 255 ระดับทำให้บริเวณทึบหายไป** — เช่น พยาธิสภาพหลังหัวใจ (retrocardiac) มองไม่เห็นเพราะหัวใจขาวทั้งหมด
- ฉลากบางส่วนอาจใช้ข้อมูลทางคลินิกที่ผู้ตรวจภาพไม่มี (เช่น ไข้ → pneumonia)

## 7. ความเกี่ยวข้องกับโครงงานเรา

| ประเด็นในเค้าโครง | ใช้เปเปอร์นี้อ้างอย่างไร |
|---|---|
| ข้อจำกัดด้านฉลาก (Sec. 6) | Infiltration PPV 66% / 36% และ Normal PPV 76% / 62% → ฉลากทั้งสองคลาสในโจทย์ binary ของเรามี noise สูง เป็นเพดานของ AUC/F1 |
| เลือก Infiltration vs No Finding | ฉลาก Infiltration / Consolidation / Pneumonia แยกกันแบบ "arbitrary" → ควรเขียนในเค้าโครงว่า Infiltration ใน NIH เป็นฉลากกลุ่ม airspace opacity ที่ไม่จำเพาะ ไม่ใช่ pneumonia โดยตรง (ระวังการใช้คำ "ปอดอักเสบ" ในชื่อเรื่อง) |
| Patient-level split | ผู้ป่วย 43% มีหลายภาพ คิดเป็น 84% ของข้อมูล → การ split ระดับภาพทำให้ leakage สูง |
| ทำไมต้องประเมิน XAI กับ B-Box | test set ที่ติดฉลากแบบเดียวกับ train อาจ "fail silently"; B-Box จากรังสีแพทย์ที่ดูภาพจริงเป็นหลักฐานอิสระที่ตรวจว่าโมเดลมองถูกที่หรือไม่ |
| Shortcut / confounder | ตัวอย่าง chest drain ใน pneumothorax สนับสนุนการใช้ XAI ตรวจว่าโมเดลดูรอยโรคจริง |
| Image contract 8-bit | ภาพ NIH เหลือ 255 ระดับเทา บริเวณทึบ (หัวใจ) สูญเสียรายละเอียด → ข้อจำกัดของ denoising/CLAHE ที่ทำบน 8-bit |
| ระวังการอ้างตัวเลข | prevalence Infiltration "almost 10%" ในเปเปอร์นี้ไม่ตรงข้อมูลจริง (≈17.7%) |

## ประโยคที่อ้างตรงได้

- "The ChestXray14 labels did not accurately reflect the visual content of the images, with positive predictive values mostly between 10% and 30% lower than the values presented in the original documentation." (Abstract)
- "...the CXR14 label schema considers pneumonia, consolidation, and infiltration as distinct processes, but clinically it is rarely possible to distinguish these processes, at least without clinical information." (p. 4)
- "...the results of testing can look good because the models can learn to reproduce the flawed labels from the training data, but the actual clinical performance of these systems will be poor." (p. 2)
- "...label harvesting with natural language processing may never be able to accurately reflect the image findings on the films..." (p. 7)
- "...it was regularly impossible to identify any retrocardiac pathology in the CXR14 dataset, because the heart appeared purely white." (p. 8)
