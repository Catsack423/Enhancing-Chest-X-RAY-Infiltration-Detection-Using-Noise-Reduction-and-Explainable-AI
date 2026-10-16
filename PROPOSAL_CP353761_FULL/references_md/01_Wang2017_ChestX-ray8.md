# Wang et al. (2017) — ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks

> ไฟล์ต้นฉบับ: `Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf` (CVPR 2017, หน้า 2097–2106)
> สถานะในเค้าโครง: อ้างอิงข้อ 17 ใน `PROPOSAL_CP353761_FULL.md`

## ข้อมูลบรรณานุกรม (APA 7)

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2097–2106).

- หน่วยงาน: NIH Clinical Center + National Library of Medicine (Bethesda, MD)
- ข้อสังเกต: ฉบับ CVPR นี้คือเวอร์ชัน **8 โรค / 108,948 ภาพ / 32,717 ผู้ป่วย** — ชุดข้อมูลที่เราใช้ (ChestX-ray14: 14 โรค, 112,120 ภาพ, 30,805 ผู้ป่วย) เป็นเวอร์ชันขยายที่ NIH ปล่อยภายหลัง (arXiv:1705.02315 ฉบับปรับปรุง) ตัวเลขในเปเปอร์นี้จึง **ไม่ตรง** กับไฟล์ `Data_Entry_2017.csv` ของเรา ถ้าจะอ้างตัวเลข 112,120 ภาพต้องระบุว่าเป็นเวอร์ชันขยาย

## สรุปสั้น (TL;DR)

สร้างฐานข้อมูล CXR ขนาดโรงพยาบาลจาก PACS ของ NIH โดยใช้ NLP ดึงฉลากโรคจากรายงานรังสีแพทย์ (ไม่ใช่การให้รังสีแพทย์ติดฉลากภาพโดยตรง) แล้วทำ benchmark การจำแนกหลายฉลาก (multi-label) และการระบุตำแหน่งแบบ weakly-supervised (ใช้ heatmap จาก CAM สร้าง bounding box) พบว่า ResNet-50 ดีที่สุด แต่ **Infiltration ได้ AUC เพียง 0.6128** ซึ่งต่ำเป็นอันดับต้น ๆ

## 1. การสร้างชุดข้อมูล

- ดึงภาพจาก PACS ช่วงปี 1992–2015
- 8 โรค: Atelectasis, Cardiomegaly, Effusion, **Infiltration**, Mass, Nodule, Pneumonia, Pneumothorax — เลือกตาม feedback ของรังสีแพทย์
- ภาพที่ไม่มีโรคใดเลยติดฉลากเป็น "Normal" (ภายหลังคือ "No Finding")
- ภาพแปลงจาก DICOM และ **resize เป็น 1024×1024** ปรับ intensity ด้วย default window ใน DICOM header (หน้า 2100–2101)
- หน้า 2099: กราฟ co-occurrence ระบุว่า **"Infiltration is often associated with Atelectasis and Effusion"** — ใช้สนับสนุนเหตุผลที่เราตัด negative class ให้เหลือ No Finding ล้วน

### 1.1 การติดฉลากด้วย Text Mining (Sec. 2.1)

- อ่านเฉพาะส่วน Findings และ Impression ของรายงาน (ถ้าไม่มีทั้งสองส่วนจึงใช้ทั้งรายงาน)
- ตรวจหาโรคด้วย **DNorm** (machine learning, map ไปยัง SNOMED-CT) + **MetaMap** (ontology-based, UMLS; semantic type `dsyn`, `fndg`) แล้วรวมผลเพื่อเพิ่ม recall
- จัดการ negation/uncertainty ด้วยกฎบน dependency graph (NLTK, Bllip parser, Stanford "CCProcessed" dependencies) แทน regular expression เช่น "clear of focal airspace disease, pneumothorax, or pleural effusion" จับ negation ได้ครบทั้ง 3 คำ
- เกณฑ์ "Normal": ไม่พบโรคใดเลยในรายงาน (**ทุกโรค ไม่ใช่แค่ 8 โรค**) หรือพบ concept "normal"/"normal size" (CUI C0205307, C0332506)

### 1.2 การตรวจคุณภาพฉลาก (Sec. 2.2, Table 2)

ประเมินกับรายงาน OpenI ที่มนุษย์ติดฉลาก (3,851 รายงาน; ใช้ subset 2,435):

| โรค | MetaMap P / R / F | วิธีของเปเปอร์ P / R / F |
|---|---|---|
| **Infiltration** | 0.25 / 0.98 / 0.39 | **0.74 / 0.87 / 0.80** |
| Pneumonia | 0.58 / 0.93 / 0.71 | 0.66 / 0.93 / 0.77 |
| Normal | 0.93 / 0.90 / 0.91 | 0.87 / 0.99 / 0.93 |
| Total | 0.84 / 0.88 / 0.86 | 0.90 / 0.91 / 0.90 |

ประเด็นสำคัญสำหรับเรา: precision ของฉลาก Infiltration อยู่ที่ **0.74** แปลว่าประมาณ 1 ใน 4 ของฉลาก Infiltration ที่ text-mined อาจผิด (label noise) — ใช้อธิบายเพดาน AUC ของงานเราได้

จำนวนภาพในชุด (Table 1): Infiltration **10,317** ภาพ (overlap กับโรคอื่น 4,698), Normal **84,312** ภาพ

### 1.3 Bounding Box (Sec. 2.4)

- เลือก 200 instance ต่อโรค รวม **1,600 instance ใน 983 ภาพ**
- รังสีแพทย์ board-certified วาด B-Box เฉพาะโรคที่ระบุในแต่ละภาพ เก็บเป็น XML (ภายหลังคือ `BBox_List_2017.csv`)
- ใช้ **เฉพาะการประเมิน localization** ไม่ใช้ฝึก
- (ในชุด ChestX-ray14 ของเรา Infiltration มี 123 กล่อง ทั้งหมดอยู่ใน `test_list.txt`)

## 2. วิธีการ (Sec. 3)

- **Unified DCNN**: เอา backbone ที่ pretrain บน ImageNet (AlexNet, GoogLeNet, VGG-16, ResNet-50) ตัด FC ทิ้ง แล้วใส่ transition layer → global pooling → prediction layer → loss
- Transition layer ทำให้ output มีขนาด S×S×D (S ∈ {8,16,32}; ResNet D=2048)
- Multi-label: เวกเตอร์ฉลาก 8 มิติ; Normal = เวกเตอร์ศูนย์ทั้งหมด
- **Weighted Cross-Entropy (W-CEL)** แก้ class imbalance (สมการ 1):
  `L = β_P Σ_{y_c=1} −ln f(x_c) + β_N Σ_{y_c=0} −ln(1 − f(x_c))`
  โดย `β_P = (|P|+|N|)/|P|`, `β_N = (|P|+|N|)/|N|` คำนวณต่อ batch
- Pooling: เทียบ average / max / **LSE pooling** (สมการ 2–3; r คุมระหว่าง average กับ max)
- Heatmap: คูณ activation ของ transition layer (S×S×D) กับน้ำหนัก prediction layer (D×C) — คือหลักการเดียวกับ **CAM** (Zhou et al., 2016) ซึ่งเป็นต้นแบบของ Grad-CAM
- สร้าง B-Box: normalize heatmap เป็น [0,255] แล้ว threshold ที่ 60 และ 180 ครอบ connected region

## 3. การทดลอง (Sec. 4)

- แบ่งข้อมูล **สุ่มระดับภาพ** 70/10/20 (train/val/test) — ไม่ใช่ patient-level (ข้อควรระวัง: split ทางการของ ChestX-ray14 ที่เราใช้เป็น patient-level ซึ่งออกมาภายหลัง)
- Caffe, batch size × iter size = 80, ResNet-50 ราว 10,000 iterations, 4× Titan X
- ฝึกเฉพาะ transition + prediction layer ส่วน backbone ใช้ weight จาก ImageNet

### 3.1 ผลการจำแนก (Table 3 — AUC)

| Backbone | Atel. | Cardio. | Effus. | **Infil.** | Mass | Nodule | Pneumonia | Pneumothorax |
|---|---|---|---|---|---|---|---|---|
| AlexNet | 0.6458 | 0.6925 | 0.6642 | 0.6041 | 0.5644 | 0.6487 | 0.5493 | 0.7425 |
| GoogLeNet | 0.6307 | 0.7056 | 0.6876 | 0.6088 | 0.5363 | 0.5579 | 0.5990 | 0.7824 |
| VGGNet-16 | 0.6281 | 0.7084 | 0.6502 | 0.5896 | 0.5103 | 0.6556 | 0.5100 | 0.7516 |
| **ResNet-50** | 0.7069 | **0.8141** | 0.7362 | **0.6128** | 0.5609 | 0.7164 | 0.6333 | 0.7891 |
| ResNet-50 + CEL | 0.7064 | 0.7262 | 0.7351 | 0.6084 | 0.5530 | 0.6545 | 0.5164 | 0.7665 |
| ResNet-50 + W-CEL | 0.7069 | 0.8141 | 0.7362 | 0.6128 | 0.5609 | 0.7164 | 0.6333 | 0.7891 |

- ResNet-50 ดีที่สุดทุก backbone → สนับสนุนการเลือก **ResNet-50 เป็น fixed backbone** ของเรา
- W-CEL ช่วยคลาสที่มีตัวอย่างน้อย (Cardiomegaly 0.7262 → 0.8141; Pneumonia 0.5164 → 0.6333) แต่ Infiltration ขยับเพียง 0.6084 → 0.6128
- Pooling: LSE (r = 10) ดีที่สุด; average กับ max ใกล้เคียงกัน (Fig. 6)
- หมายเหตุ: ในเนื้อความเปเปอร์เรียกตาราง AUC ว่า "Table 4" แต่ในตารางจริงเป็น Table 3

### 3.2 ผลการระบุตำแหน่ง (Table 4)

- heatmap ความละเอียด 32×32 เทียบภาพ 1024×1024 → กล่องที่ได้มักใหญ่กว่า GT จึงใช้ **IoBB** (Intersection over detected B-Box area) แทน IoU เป็นหลัก
- ถือว่าถูกเมื่อ IoU > T(IoU) หรือ IoBB > T(IoBB)

ค่า Infiltration (Acc. / AFP = average false positives):

| T(IoBB) | Acc. | AFP |
|---|---|---|
| 0.1 | 0.7886 | 0.0426 |
| 0.25 | 0.5772 | 0.0914 |
| 0.5 | 0.4227 | 0.1422 |
| 0.75 | 0.3252 | 0.1737 |
| 0.9 | 0.2520 | 0.1910 |

เทียบ: Cardiomegaly ที่ T=0.5 ได้ 0.8767 ส่วน Nodule ได้ 0.0126 — โรคที่เป็นวัตถุเล็กระบุตำแหน่งยากมาก

## 4. ข้อจำกัดที่เปเปอร์ยอมรับเอง

- ฉลากเป็น "machine-human annotated" มาจาก NLP ไม่ใช่การอ่านภาพโดยตรง
- Pneumonia มีน้อยกว่า 1% ของภาพ จึงได้ AUC ต่ำ
- จำนวน B-Box (1,600) น้อยเมื่อเทียบกับทั้งชุด แต่ผู้เขียนมองว่าพอใช้ประมาณผล localization ได้
- สรุปท้ายเปเปอร์: การอ่าน CXR อัตโนมัติแบบแม่นยำสูง "remains a strenuous task"

## 5. ความเกี่ยวข้องกับโครงงานเรา

| ประเด็นในเค้าโครง | ใช้เปเปอร์นี้อ้างอย่างไร |
|---|---|
| ที่มาชุดข้อมูล / B-Box GT | แหล่งต้นทางของ NIH ChestX-ray และ bounding box จากรังสีแพทย์ |
| เลือก ResNet-50 | ResNet-50 ชนะ AlexNet/GoogLeNet/VGG ทุกโรค (Table 3) |
| Infiltration ยาก | AUC 0.6128 ต่ำกว่าโรคอื่นส่วนใหญ่; ฉลากมี precision 0.74 |
| Negative = No Finding ล้วน | Infiltration มักเกิดร่วม Atelectasis/Effusion (Fig. 2) |
| XAI localization | ต้นแบบการใช้ CAM heatmap + B-Box ประเมินด้วย IoU/IoBB หลาย threshold — แนวเดียวกับ IoU(τ) ที่ τ = 0.3, 0.5 ของเรา |
| Over-smoothing ของ heatmap | heatmap 32×32 หยาบ ทำให้กล่องใหญ่กว่า GT — ปัจจัยที่ต้องระวังเวลาตีความ IoU ต่ำ |
| ข้อควรระวัง | split ในเปเปอร์เป็นแบบสุ่มระดับภาพ ตัวเลขจึงเทียบกับงานเรา (patient-level split) ตรง ๆ ไม่ได้ |

## ประโยคที่อ้างตรงได้

- "Infiltration is often associated with Atelectasis and Effusion." (p. 2099)
- "...deep convolutional neural network based 'reading chest X-rays' ... remains a strenuous task for fully-automated high precision CAD systems." (Abstract, p. 2097)
- "Due to the relatively low spatial resolution of heatmaps (32 × 32) in contrast to the original image dimensions (1024 × 1024), the computed B-Boxes are often larger than the according GT B-Boxes." (p. 2104)
