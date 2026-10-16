# Baltruschat et al. (2019) — Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification

> ไฟล์ต้นฉบับ: `Baltruschat et al. (2019) — Comparison of Deep Learning Approaches for Multi-Label Chest X-Ray Classification.pdf` (9 หน้า + references)
> สถานะในเค้าโครง: **ยังไม่อยู่ในรายการอ้างอิง** ของ `PROPOSAL_CP353761_FULL.md` — แนะนำให้เพิ่ม

## ข้อมูลบรรณานุกรม (APA 7)

Baltruschat, I. M., Nickisch, H., Grass, M., Knopp, T., & Saalbach, A. (2019). Comparison of deep learning approaches for multi-label chest X-ray classification. *Scientific Reports*, 9, 6381. https://doi.org/10.1038/s41598-019-42294-8

- หน่วยงาน: Hamburg University of Technology, University Medical Center Hamburg-Eppendorf, Philips Research (เยอรมนี)
- Open access (Nature Scientific Reports)

## สรุปสั้น (TL;DR)

ศึกษา **ResNet-50 บน ChestX-ray14** อย่างเป็นระบบ: off-the-shelf (OTS) vs fine-tuning (FT) vs train from scratch, ขยาย input เป็น 448×448, เพิ่ม non-image features (อายุ เพศ view position) และเทียบความลึก (ResNet-38/50/101) ใช้ 5-fold re-sampling ระดับผู้ป่วย พบว่า **ผลแปรปรวนมากตาม data split**, fine-tuning ดีกว่า OTS มาก (0.730 → 0.819 AUC เฉลี่ย), และ **Infiltration ยังได้ AUC ต่ำสุดในทุก setup (~0.70)** นอกจากนี้ใช้ Grad-CAM พบว่าโมเดลเรียนรู้ "สายระบายทรวงอก (chest drain)" แทน pneumothorax จริง

## 1. ชุดข้อมูล (ChestX-ray14)

- 112,120 ภาพ frontal จาก 30,805 ผู้ป่วย; 14 โรค + "No Finding"
- NIH ไม่ได้ให้ DICOM ต้นฉบับ แต่ทำ preprocessing ตาม display setting แล้ว **ลดเป็น 8-bit** และ **resize เป็น 1024×1024 โดยไม่รักษา aspect ratio** (หน้า 4–5) — ตรงกับ image contract `uint8 [0,255]` ของเรา
- Prevalence ของโรค (Table 2) อยู่ระหว่าง 0.20% (Hernia) ถึง **17.74% (Infiltration, 19,894 ภาพ)** — Infiltration เป็นโรคที่มีภาพมากที่สุด
- อายุเฉลี่ย 46.87 ± 16.60 ปี; Table 3 ในเปเปอร์เขียนหัวคอลัมน์ Female 63,340 / Male 48,780 แต่ตรวจกับ `Data_Entry_2017.csv` แล้ว **สลับกัน: จริงคือ M = 63,340, F = 48,780** (อัตราส่วน 1.30); PA:AP = 67,310:44,810

## 2. วิธีการ

| ประเด็น | รายละเอียด |
|---|---|
| ปัญหา | Multi-label, M = 15 ฉลาก (14 โรค + No Finding เป็นฉลากแยก) |
| Loss | class-averaged Binary Cross-Entropy — ทดลอง positive/negative balancing (แบบ Wang et al.) แล้ว **ไม่ต่างอย่างมีนัยสำคัญ** |
| Backbone | ResNet-50 แทน dense layer สุดท้ายด้วย 15-d + sigmoid |
| OTS | ใช้ ImageNet weights เป็น feature extractor ฝึกเฉพาะ classifier |
| FT | ฝึก conv layers ใหม่ทั้งหมด (ยกเว้น pooling) จาก ImageNet weights |
| From scratch | ResNet-50-1channel (input 1 ช่อง) และ ResNet-50-large (448×448, เพิ่ม max-pool หลัง bottleneck แรก) |
| Meta | ต่อเวกเตอร์ 3 มิติ (VP, gender เป็น {0,1}, อายุ scale [0,1]) เข้ากับ image feature ก่อน dense |
| Split | **5× re-sampling** 70/10/20 โดยผู้ป่วยหนึ่งคนอยู่ subset เดียว (patient-level) + ทดสอบบน official split |
| Augmentation | random crop 8–100% ของพื้นที่, aspect 3:4–4:3, หมุน ±7°, horizontal flip |
| Test | resize 256×256 (หรือ 480×480 สำหรับ large) แล้ว center crop |
| Optimizer | Adam (β1 = 0.9, β2 = 0.999), lr = 0.001 (transfer) / 0.01 (scratch), ลด lr ครึ่งหนึ่งเมื่อ val loss ไม่ดีขึ้น, ไม่ใช้ dropout |
| Batch | 16 (transfer) / 8 (scratch-large) |
| Framework | CNTK, GTX 1080, ~10 ms/ภาพ |

## 3. ผลลัพธ์

### 3.1 5-fold re-sampling (Table 4, AUC × 100, mean ± std)

| Setup | **Infiltration** | Pneumonia | Average (14 โรค) | No Finding |
|---|---|---|---|---|
| OTS | 65.9 ± 0.2 | 66.4 ± 2.7 | 73.0 | 71.6 |
| **FT** | **69.9 ± 0.6** | 74.4 ± 1.6 | 81.7 | 76.9 |
| Scratch 1channel | 70.2 ± 0.3 | 74.3 ± 1.5 | 81.9 | 77.3 |
| Scratch large | 70.2 ± 0.5 | 75.3 ± 2.2 | 82.1 | 77.1 |
| OTS + meta | 67.0 ± 0.4 | 68.3 ± 2.3 | 74.8 | 72.5 |
| FT + meta | 70.2 ± 0.2 | 73.3 ± 1.3 | 82.0 | 76.8 |
| 1channel + meta | 70.1 ± 0.5 | 74.8 ± 1.5 | 82.0 | 77.1 |
| large + meta | 70.0 ± 0.7 | 76.7 ± 1.5 | **82.2** | 77.1 |

- Fine-tuning ยก AUC เฉลี่ยจาก 0.730 → 0.819
- การฝึกจากศูนย์บน CXR ได้ผลใกล้เคียง FT; ภาพใหญ่ (448) ช่วยโรควัตถุเล็ก (Nodule +0.018, Mass +0.006)
- Non-image features เพิ่ม AUC เพียงเล็กน้อย — เพราะ image feature **ทำนาย view position ได้ AUC 0.9983**, เพศ AUC 0.9435, อายุ MAE 9.13 ± 7.05 ปี (ภาพ "รู้" ข้อมูลเหล่านี้อยู่แล้ว)
- Std สูงสุดถึง 0.05 ใน Hernia → การเทียบงานที่ใช้ split ต่างกันอาจไม่สรุปผล

### 3.2 Official split (Table 6, AUC)

| โรค | Wang et al. | Yao et al. | Guendel et al.* | ResNet-38-large-meta | ResNet-50-large-meta | ResNet-101-large-meta |
|---|---|---|---|---|---|---|
| **Infiltration** | 0.661 | 0.673 | 0.709 | 0.694 | 0.694 | 0.686 |
| Pneumonia | 0.658 | 0.684 | 0.731 | 0.714 | 0.703 | 0.694 |
| Average | 0.745 | 0.761 | 0.807 | **0.806** | 0.795 | 0.785 |
| No Finding | – | – | – | 0.727 | 0.725 | 0.720 |

\* Guendel et al. ใช้ข้อมูล PLCO เพิ่มอีก 185,000 ภาพ

- โมเดลตื้นกว่าดีกว่าบน ChestX-ray14 (101 → 50 → 38: 0.785 → 0.795 → 0.806)
- Abstract สรุปว่า "X-ray-specific ResNet-38 integrating non-image data yields the best overall results" (อิง official split); ใน 5-fold ตัวที่ดีที่สุดคือ ResNet-50-large-meta

### 3.3 Grad-CAM และ label noise (หน้า 7, Fig. 4)

- ใช้ Grad-CAM ตรวจโมเดล pneumothorax: บางภาพ activation สูงสุดอยู่ที่ **chest drain** ไม่ใช่รอยโรค → โมเดลเรียนรู้ shortcut จากผู้ป่วยที่รักษาแล้ว
- ยืนยันข้อกังวลของ Oakden-Rayner เรื่องคุณภาพฉลาก NIH
- Label noise ของ ChestX-ray14 ประมาณ **10%** จึงต้องการ clean test set สำหรับประเมินผลทางคลินิก

## 4. ความเกี่ยวข้องกับโครงงานเรา

| ประเด็นในเค้าโครง | ใช้เปเปอร์นี้อ้างอย่างไร |
|---|---|
| ResNet-50 + ImageNet + fine-tune | หลักฐานตรงว่า FT ดีกว่า OTS มากบน ChestX-ray14 (0.730 → 0.819) — สนับสนุนการ fine-tune แทนการ freeze ทั้งหมด |
| Fixed backbone / fair comparison | ผลแปรปรวนตาม split สูง → สนับสนุนการล็อก official split และใช้ backbone/hyperparameter เดียวกันทุก condition; ควรรายงานหลาย seed + CI |
| Patient-level split | ใช้ re-sampling ระดับผู้ป่วย และอธิบายว่าผู้ป่วยมี follow-up หลายภาพ |
| Infiltration ยาก | AUC ~0.69–0.70 เป็นค่าต่ำสุดใน 14 โรคทุก setup → ตั้งความคาดหวัง baseline ได้ (แม้งานเราเป็น binary vs No Finding ซึ่งง่ายกว่า multi-label) |
| Image contract uint8 | NIH ลดเป็น 8-bit และ resize 1024×1024 ไม่รักษา aspect ratio |
| Augmentation | ตัวอย่าง geometric augmentation (crop, rotation ±7°, flip) สำหรับ CXR |
| XAI ตรวจ shortcut | ตัวอย่างใช้ Grad-CAM พบ confounder (chest drain) — สนับสนุนข้ออ้างใน Background ว่าโมเดลอาจตัดสินจาก artifact ไม่ใช่รอยโรค |
| Loss weighting | balancing loss ไม่ได้ช่วยอย่างมีนัยสำคัญในงานนี้ |
| ข้อควรระวัง | horizontal flip บน CXR เปลี่ยนตำแหน่งหัวใจ — ถ้าใช้ต้องอธิบาย; label noise ~10% จำกัดเพดานผล |

## ประโยคที่อ้างตรงได้

- "Our experiments (Sec. 3) demonstrate that performance of a network depends significantly on the selected split." (p. 2)
- "Using fine-tuning, the results are improved considerably, from 0.730 to 0.819 AUC on average." (p. 6)
- "This indicates that the network learned not only to detect an acute pneumothorax but also the presence of chest drains." (p. 7)
- "...for the ChestX-ray14 datasets, the rather high label noise of 10% makes an assessment of the true network performance difficult. Therefore, a clean test set without label noise is needed for clinical impact evaluation." (p. 9)
