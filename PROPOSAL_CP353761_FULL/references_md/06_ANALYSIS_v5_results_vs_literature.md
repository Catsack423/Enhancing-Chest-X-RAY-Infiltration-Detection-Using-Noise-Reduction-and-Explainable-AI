# วิเคราะห์ผลทดลอง v5 beta เทียบกับงานวิจัยอ้างอิง

> อินพุต: [`Notebook_5 beta/results/SUMMARY.md`](../../Notebook_5%20beta/results/SUMMARY.md) (รัน 2026-10-14), prediction ของ v4 ใน `comparison/epochs/*/epoch_7/test_predictions.csv`, metadata จาก `Data_Entry_2017.csv` / `BBox_List_2017.csv`
> เปเปอร์ที่ใช้: [01 Wang](01_Wang2017_ChestX-ray8.md), [03 Baltruschat](03_Baltruschat2019_ResNet50_ChestXray14.md), [04 Oakden-Rayner](04_OakdenRayner2020_Public_Dataset_Labels.md), [05 Majkowska](05_Majkowska2020_Adjudicated_CXR.md)
> ส่วนที่ 3 เป็นการวิเคราะห์ใหม่ที่ทำในรอบนี้ (ยังไม่มีใน SUMMARY.md) โค้ดอยู่ในภาคผนวก

## สรุปสั้น

> **ผลล่าสุด (ส่วนที่ 7, `best.pt` seed 42):** `best.pt` ทุก condition มาจากช่วง fine-tune layer4 จึงมีพฤติกรรมแบบ epoch 7 — PA ≈ เดาสุ่ม (0.465) และ denoising ไม่ได้ดีกว่า baseline อย่างเป็นระบบ (หลังปรับ multiple comparisons เหลือเพียง DAE+CLAHE L1 +0.036) แต่ DAE+CLAHE และ CLAHE+DWT แย่ลงเมื่อเพิ่มความแรง สอดคล้องกับ over-smoothing

1. **AUC ของ v4 (~0.67–0.68) อยู่ในช่วงที่งานวิจัยรายงานไว้สำหรับ Infiltration บน NIH (0.61–0.71)** จึงไม่ได้ผิดปกติ ส่วนตัวเลข 0.92 ของ notebook อ้างอิงสอดคล้องกับงานที่ใช้ฉลากจากรังสีแพทย์ (Majkowska: airspace opacity AUC 0.94 บนภาพ NIH ชุดเดียวกัน แต่ใช้ฉลาก adjudicated)
2. **ข้อค้นพบใหม่ที่สำคัญที่สุด: บนภาพ PA โมเดล v4 ได้ AUC 0.508 [0.482, 0.535] ซึ่งเท่ากับการเดาสุ่ม** ความสามารถแยกโรคทั้งหมดมาจากภาพ AP (0.679) และเป็นแบบนี้ **ทุก condition ของ denoising** (PA 0.46–0.51)
3. **ข้อมูล metadata อย่างเดียว (view position, follow-up #, อายุ, เพศ) ทำนายได้ AUC 0.685 ≈ โมเดลภาพ (0.680)** เพราะ official test set มีภาพ Infiltration จากผู้ป่วยที่ถ่ายซ้ำหลายครั้งและถ่ายแบบ AP มากกว่า No Finding มาก ตรงกับปัญหา hidden stratification ที่ Oakden-Rayner (2020) เตือนไว้
4. **⚠ ส่วนที่ 6 แก้ข้อสรุปของข้อ 2 และ 5: ผลเหล่านั้นวัดที่ epoch 7 (หลัง fine-tune layer4) เท่านั้น ที่ epoch 2 baseline ได้ AUC กับฉลากรังสีแพทย์ 0.842 (PA 0.741) และชนะ denoising เกือบทุกตัว — ยังสรุปไม่ได้ว่า denoising ดีกว่า baseline**
5. **(ส่วนที่ 5, snapshot epoch 7) วัดกับฉลากรังสีแพทย์ของ Google แล้ว**: กลุ่ม AP ขึ้นจาก 0.68 เป็น 0.77 แต่ PA ยัง ≈ 0.47 และ denoising 9/11 condition ดีกว่า baseline อย่างมีนัย (ซึ่งมองไม่เห็นเมื่อวัดกับฉลาก NIH) พร้อมรูปแบบ over-smoothing L1 < L2 > L3 ใน CLAHE+DWT และ DAE+CLAHE
6. ผลต่อโครงงาน: การเทียบ denoising ด้วย AUC รวมกำลังวัดผลในกลุ่ม AP เป็นหลัก และชุด BBox 123 ภาพมี AP 81 / PA 42 และ **มีเพียง 19 ภาพที่เป็น Infiltration-only** ตรงนิยาม positive ตอนฝึก ต้องรายงานแยกกลุ่มย่อย

---

## 1. ข้อสรุปใน SUMMARY.md เทียบกับวรรณกรรม

### ข้อ 1–2: "ไม่พบ bug ช่องว่างมาจาก dataset/label" → วรรณกรรมสนับสนุนเต็มที่

| แหล่ง | Setup | Infiltration AUC |
|---|---|---|
| Wang et al. (2017) | ResNet-50, multi-label, split สุ่มระดับภาพ | 0.613 |
| Baltruschat et al. (2019) | ResNet-50 FT, 5-fold ระดับผู้ป่วย | 0.699 ± 0.006 |
| Baltruschat et al. (2019) | ResNet-50-large-meta, official split | 0.694 |
| **v5 E5 (เรา)** | full FT, 2,000+2,000, Infiltration-only, official test | **0.613 [0.600, 0.625]** |
| **v4 release (เรา)** | 46,108 ภาพ, official test | **0.668** (baseline epoch 7 ในไฟล์ prediction = 0.680) |

- ผลของเราอยู่ในช่วงเดียวกับงานตีพิมพ์ที่ใช้ ResNet-50 บน NIH ทั้งที่งานเหล่านั้นเป็น multi-label
- **หลักฐานที่ชี้ว่าเป็นปัญหาฉลาก ไม่ใช่ปัญหาภาพ**: Majkowska et al. (2020) ใช้ภาพจาก NIH test set เดียวกัน แต่ให้รังสีแพทย์ 3 คน adjudicate ฉลากใหม่ ได้ **airspace opacity AUC 0.94** — ช่องว่างจาก ~0.69 เป็น 0.94 บนภาพชุดเดียวกันจึงมาจากฉลากเป็นหลัก ซึ่งอธิบายช่องว่าง 0.67 กับ 0.92 (VinBigData ใช้ฉลากจากรังสีแพทย์) ได้ตรง
- ขนาดของ label noise:
  - Wang (2017): precision ของฉลาก Infiltration จาก text mining = 0.74 (วัดบน OpenI ไม่ใช่ NIH)
  - Oakden-Rayner (2020): รังสีแพทย์ดูภาพจริง Infiltration PPV **66% (ผ่อนปรน) / 36% (แบบคลินิก)**, No Finding PPV 76% / 62%
  - Majkowska (2020): ฉลาก NLP ของ NIH มี sensitivity เพียง 45–50% (pneumothorax, nodule/mass)
  - ถ้าภาพบวกจริงเพียง 36–66% และภาพลบจริงเพียง 62–76% เพดาน AUC ที่วัดกับฉลาก NIH จะต่ำกว่าความสามารถจริงของโมเดลมาก

### ข้อ 3: "นิยาม positive แบบ any เพิ่ม AUC ≈ +0.06" → สอดคล้อง

- Wang (2017): Infiltration มักเกิดร่วมกับ Atelectasis และ Effusion
- ข้อมูลเรา: ภาพ any-Infiltration มีโรคร่วมบ่อยสุดคือ Effusion 4,000, Atelectasis 3,264, Nodule 1,546, Consolidation 1,221 ภาพ — โรคเหล่านี้ทำให้ภาพผิดปกติชัดขึ้น โจทย์จึงเอียงไปทาง "ผิดปกติ vs ปกติ"
- Oakden-Rayner: ฉลาก Infiltration / Consolidation / Pneumonia แยกกันแบบ "mostly arbitrary" → การใช้ Infiltration-only ไม่ได้ทำให้ได้ภาพที่ "บริสุทธิ์" กว่า แค่ตัดภาพที่ชัดออก
- ข้อเสนอสำหรับรายงาน: คง Infiltration-only ตาม protocol แต่เขียนชัดว่าเป็นโจทย์ที่ยากกว่าที่งานส่วนใหญ่ใช้ และเทียบตัวเลขกับงานที่ใช้ any ตรง ๆ ไม่ได้

### ข้อ 4: "test 100 ภาพแกว่งมาก" → สอดคล้อง

- Baltruschat (2019): AUC เปลี่ยนตาม split อย่างมีนัย (std สูงถึง 0.05 ในคลาสเล็ก) และเตือนว่าการเทียบงานที่ใช้ split ต่างกัน "might be non-conclusive"
- Majkowska (2020): ใช้ bootstrap 1,000 รอบรายงาน 95% CI
- ข้อเสนอเพิ่ม: ใช้ **patient-level (cluster) bootstrap** แทน image-level เพราะ test set มีภาพซ้ำจากผู้ป่วยเดียวกันมาก (ผู้ป่วย 20 คนแรกมีภาพรวมกัน 9.9% ของ test, สูงสุด 88 ภาพจากผู้ป่วยคนเดียว) image-level bootstrap จะให้ CI แคบเกินจริง

### ข้อ 5: "freeze backbone ไม่ได้แย่ลงอย่างมีนัย (200 ภาพ)" → ระวังการสรุป

- Baltruschat (2019) ที่ข้อมูลเต็ม: fine-tune ดีกว่า off-the-shelf มาก (AUC เฉลี่ย 0.730 → 0.819; Infiltration 0.659 → 0.699)
- E4 ใช้ 200 ภาพ ทุก setup ยังใกล้ 0.5 (control E3 = 0.497) จึง **ไม่มี power พอ** จะเห็นความต่าง ควรเขียนว่า "ไม่พบความต่างที่ข้อมูลขนาดเล็ก" ไม่ใช่ "freeze ไม่ได้แย่กว่า"

---

## 2. ข้อสังเกตต่อ notebook อ้างอิง (VinBigData) เทียบวรรณกรรม

| ข้อสังเกตใน SUMMARY | วรรณกรรมที่เกี่ยวข้อง |
|---|---|
| `drop_duplicates` เลือก annotation แถวแรกของรังสีแพทย์หลายคน | Majkowska: ความเห็นตรงกันหลังอ่านรอบแรกเพียง 41.8% → การเลือกแถวแรกเท่ากับสุ่มผู้อ่านหนึ่งคน ฉลากจะผันผวนมากในภาพก้ำกึ่ง |
| split ระดับภาพแบบสุ่ม ไม่มี patient ID | Oakden-Rayner: ผู้ป่วย 43% มีหลายภาพ (84% ของข้อมูล NIH); ถ้า VinBigData มีลักษณะคล้ายกัน การสุ่มระดับภาพอาจรั่ว |
| test 100 ภาพ seed เดียว ไม่มี CI | E0 ของเรา: สุ่ม 50+50 จาก prediction ของ v4 ได้ช่วง 0.58–0.78 สูงสุด 0.81–0.86 — ตัวเลขจาก test 100 ภาพเดียวจึงไม่น่าเชื่อถือ |

---

## 3. การวิเคราะห์ใหม่: hidden stratification ใน official test set

### 3.1 Train กับ test มีการกระจายต่างกันมาก

| Split / คลาส | n | AP % | Follow-up # มัธยฐาน | ภาพแรกของผู้ป่วย (FU = 0) % |
|---|---|---|---|---|
| train Infiltration-only | 7,327 | 35.6 | 2 | 33.6 |
| train No Finding | 50,500 | 31.7 | 1 | 38.0 |
| **test Infiltration-only** | **2,220** | **75.3** | **20** | **4.9** |
| test No Finding | 9,861 | 51.3 | 8 | 20.1 |

- ใน train สองคลาสมีสัดส่วน AP ใกล้กัน (35.6% vs 31.7%) แต่ใน test ต่างกันมาก (75.3% vs 51.3%)
- ภาพ Infiltration ใน test มาจากผู้ป่วยเพียง 565 คน และส่วนใหญ่เป็นภาพครั้งหลัง ๆ (มัธยฐานครั้งที่ 20) — ลักษณะผู้ป่วยใน/ICU ที่ถ่ายซ้ำทุกวัน ตรงกับที่ Oakden-Rayner อธิบาย
- อายุไม่ต่างกัน (45.1 vs 46.1 ปี)

### 3.2 Metadata อย่างเดียวทำนายได้เท่าโมเดล

| ตัวทำนาย (บน official test, Infiltration-only vs No Finding) | AUC |
|---|---|
| Follow-up # อย่างเดียว | 0.668 |
| View = AP อย่างเดียว | 0.620 |
| อายุอย่างเดียว | 0.485 |
| Logistic regression (view, log follow-up, อายุ, เพศ), patient-grouped 5-fold CV | **0.685** |
| **โมเดลภาพ v4 baseline (epoch 7)** | **0.680** |
| Logistic regression (metadata + score ของโมเดล) | 0.723 |

- โมเดลภาพไม่ได้ดีกว่าการใช้ metadata 4 ตัว แต่เมื่อรวมกันได้ 0.723 แปลว่าโมเดลมีข้อมูลบางส่วนที่ metadata ไม่มี
- (logistic regression นี้ฝึกบน test โดยใช้ cross-validation แยกผู้ป่วย จึงเป็นการวัดว่ามีข้อมูลอยู่เท่าไร ไม่ใช่โมเดลที่ใช้ได้จริง)

### 3.3 แยก AUC ของ v4 ตาม view position

| Condition | AUC รวม | **AUC เฉพาะ PA** | AUC เฉพาะ AP | AUC เฉพาะภาพแรก (FU = 0) | AUC ของ score ในการทำนาย AP vs PA |
|---|---|---|---|---|---|
| baseline | 0.680 | **0.508** | 0.679 | 0.420 | 0.680 |
| median_L1 | 0.684 | 0.512 | 0.682 | 0.442 | 0.695 |
| median_L2 | 0.672 | 0.500 | 0.669 | 0.422 | 0.700 |
| median_L3 | 0.672 | 0.496 | 0.670 | 0.410 | 0.692 |
| clahe_dwt_strength_L1 | 0.669 | 0.504 | 0.674 | 0.427 | 0.648 |
| clahe_dwt_strength_L2 | 0.667 | 0.494 | 0.670 | 0.433 | 0.671 |
| clahe_dwt_strength_L3 | 0.649 | 0.465 | 0.660 | 0.414 | 0.651 |
| dae_clahe_strength_L1 | 0.674 | 0.487 | 0.678 | 0.425 | 0.694 |
| dae_clahe_strength_L2 | 0.674 | 0.494 | 0.675 | 0.458 | 0.699 |
| dae_clahe_strength_L3 | 0.636 | 0.460 | 0.648 | 0.408 | 0.637 |
| dft_low_mid_high_fusion | 0.677 | 0.511 | 0.665 | 0.502 | 0.723 |
| dwt_L1_L4_fusion | 0.658 | 0.465 | 0.647 | 0.433 | 0.758 |

Baseline พร้อม 95% CI แบบ patient-level bootstrap (1,000 รอบ):

| กลุ่ม | n | บวก | AUC [95% CI] |
|---|---|---|---|
| ทั้งหมด | 12,081 | 2,220 | 0.680 [0.664, 0.696] |
| **PA** | 5,353 | 548 | **0.508 [0.482, 0.535]** |
| AP | 6,728 | 1,672 | 0.679 [0.660, 0.697] |

ใน AP แยกตาม follow-up: ครั้งที่ 0–5 = 0.712, 6–20 = 0.679, > 20 = 0.615

**สิ่งที่ตารางบอก**

1. **บนภาพ PA (ผู้ป่วยนอก ยืนถ่าย คุณภาพดีกว่า) ทุก condition แยก Infiltration ไม่ได้เลย** CI ของ baseline คร่อม 0.5
2. **บนภาพแรกของผู้ป่วย (FU = 0) AUC ต่ำกว่า 0.5** (มีภาพบวกเพียง 108 ภาพ ตัวเลขแกว่งได้มาก แต่ไม่มี condition ใดเกิน 0.50)
3. Score ของโมเดลแยก AP กับ PA ได้เองที่ AUC 0.64–0.76 และภาพ No Finding ที่เป็น AP ได้ score เฉลี่ย 0.667 เทียบ PA 0.612 → โมเดลตอบสนองต่อ "ความเป็นภาพ AP"
4. ความต่างระหว่าง condition (0.636–0.684) เกิดในกลุ่ม AP ทั้งหมด เช่น dae_clahe L3 ตกจาก 0.679 → 0.648 ใน AP ขณะที่ PA ทุกตัวใกล้ 0.5 อยู่แล้ว

### 3.4 ตีความ — สองสมมติฐานที่ยังแยกกันไม่ได้

| สมมติฐาน | หลักฐานสนับสนุน | วิธีทดสอบ |
|---|---|---|
| **A. Shortcut**: โมเดลเรียนลักษณะของภาพ AP/ผู้ป่วยหนัก (สายระโยง ท่อ เครื่องหมาย portable, ท่าถ่าย) ที่สัมพันธ์กับ Infiltration ใน test | Baltruschat: CNN ทำนาย view position จากภาพได้ AUC 0.998 และ Grad-CAM เจอโมเดลดู chest drain; Oakden-Rayner: hidden stratification; score ของเราแยก AP/PA ได้ | Grad-CAM บนภาพ No Finding AP ที่ได้ score สูง ดูว่าชี้ที่สาย/ท่อหรือไม่; สัดส่วน heatmap energy ที่อยู่นอกปอด |
| **B. ภาพ PA ยากจริง + ฉลากผิดมากกว่า**: Infiltration ในผู้ป่วยนอกบางกว่า และฉลากจาก report ผิดบ่อยกว่า | Oakden-Rayner: PPV แบบคลินิก 36%; Majkowska: ฉลาก NLP sensitivity ต่ำ | ประเมินด้วยฉลาก adjudicated ของ Google (ข้อ 4.4) แยก PA/AP |

ข้อสังเกตที่ทำให้ A ไม่ใช่ทั้งหมด: ใน train สัดส่วน AP สองคลาสใกล้กัน (35.6% vs 31.7%) โมเดลจึงไม่ได้ถูกสอนให้ใช้ view ตรง ๆ และในกลุ่ม AP ภาพครั้งแรก ๆ ยังได้ 0.71 แต่ไม่ว่าสาเหตุจะเป็นอะไร **ตัวเลข AUC รวมไม่ได้สะท้อนความสามารถตรวจ Infiltration ในภาพ PA เลย**

---

## 4. ผลกระทบต่อโครงงานและข้อเสนอ

### 4.1 การเทียบ denoising (คำถามหลักของงาน)

- รายงาน **AUC แยก PA / AP เป็นตารางหลัก** คู่กับ AUC รวม ไม่เช่นนั้นการบอกว่า method ไหนดีกว่าคือการบอกว่าใครใช้ลักษณะของภาพ AP ได้ดีกว่า
- แนวโน้ม over-smoothing (L3 ของ CLAHE+DWT และ DAE+CLAHE ตกชัด) **ยังเห็นได้ในกลุ่ม AP** (0.679 → 0.660 / 0.648) จึงยังใช้สนับสนุนสมมติฐาน over-smoothing ได้ แต่ต้องบอกว่าวัดในกลุ่มที่โมเดลมีสัญญาณเท่านั้น
- ใช้ patient-level bootstrap CI และทดสอบความต่างแบบ paired (ภาพชุดเดียวกันทุก condition) ก่อนสรุปว่า method ใดดีกว่า — ความต่าง 0.01–0.02 ส่วนใหญ่อยู่ใน CI

### 4.2 การประเมิน XAI กับ Bounding Box

| ลักษณะชุด BBox Infiltration (123 ภาพ) | ค่า |
|---|---|
| AP / PA | 81 / 42 |
| ฉลาก Infiltration-only (ตรงนิยาม positive ตอนฝึก) | **19** |
| มีโรคร่วม (อยู่นอกนิยามที่ใช้ฝึก/ทดสอบ classification) | 104 |

- ต้องเขียนชัดใน methodology ว่าชุด XAI ใช้นิยาม any-Infiltration ต่างจากชุด classification
- รายงาน Pointing Game / IoU **แยก PA vs AP** และแยก only vs มีโรคร่วม (กลุ่ม PA 42 ภาพ CI จะกว้าง)
- เพิ่มตัวชี้วัดที่ตรวจ shortcut: สัดส่วน heatmap energy ที่อยู่นอกบริเวณปอด (ใช้ lung mask ที่มีในแผน Sec. 5.2) — ถ้าสูงในภาพ AP สนับสนุนสมมติฐาน A

### 4.3 Control ที่ควรใส่ในรายงาน

- **Metadata baseline (AUC 0.685)** เป็นเส้นอ้างอิงคู่กับ shuffled-label control (E3 = 0.497) — โมเดลภาพที่ไม่ชนะ metadata baseline ยังไม่ได้แสดงว่าเรียนรู้จากภาพ
- รายงานสัดส่วน AP และ follow-up ของแต่ละคลาสใน train/test (ตาราง 3.1) ในบท dataset

### 4.4 การทดลองต่อที่ให้ข้อมูลมากที่สุด

1. **ประเมินกับฉลาก adjudicated ของ Google** (Majkowska et al., 2020; airspace opacity สำหรับภาพ NIH test 1,962 ภาพ — https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest#additional_labels) ไม่ต้องฝึกใหม่ ใช้ prediction เดิมได้เลย ถ้า AUC กระโดดขึ้นมาก (โดยเฉพาะใน PA) แปลว่าเพดานมาจากฉลาก NIH ถ้ายังต่ำใน PA แปลว่าโมเดลมีปัญหาจริง (หมายเหตุ: ต้องเช็คว่าภาพในชุดนั้นซ้อนกับ test ของเรากี่ภาพ และ "airspace opacity" กว้างกว่า Infiltration)
2. Grad-CAM บน No Finding AP ที่ score สูงสุด 20 ภาพ เทียบ PA — ตรวจสมมติฐาน shortcut เชิงคุณภาพ
3. ฝึกโดย balance view position ต่อคลาส หรือเพิ่ม view เป็นตัวแปร stratify แล้วดูว่า PA AUC ขยับไหม
4. E5 หลาย seed + learning curve (ตามที่ SUMMARY เสนอ) และรายงานแยก PA/AP ด้วย

### 4.5 สิ่งที่ควรเขียนเพิ่มในเค้าโครง/รายงาน

- Sec. 6 ข้อจำกัด: label noise (ตัวเลขจากข้อ 1), การกระจาย view/follow-up ที่ต่างกันระหว่าง train กับ official test, และการที่ชุด BBox มีโรคร่วม 104/123 ภาพ
- ข้อค้นพบว่า AUC บน PA ≈ 0.5 เป็นผลที่มีคุณค่าเอง: แสดงว่าการวัดแค่ AUC รวมบน NIH ปิดบังปัญหา และเป็นเหตุผลโดยตรงที่ต้องมี XAI เชิงปริมาณ — สอดคล้องกับ Background ของเค้าโครงที่เขียนเรื่อง "โมเดลตัดสินจาก artifact/confounder"

---

## 5. ผลวัดกับฉลากรังสีแพทย์ของ Google (เพิ่ม 2026-10-14)

> สคริปต์: [`Notebook_5 beta/eval_google_labels.py`](../../Notebook_5%20beta/eval_google_labels.py) · ผลเต็ม: [`Notebook_5 beta/results/google_labels_eval.md`](../../Notebook_5%20beta/results/google_labels_eval.md)
> ฉลาก: สำเนา bucket `gcs-public-data--healthcare-nih-chest-xray-labels` ที่ root ของ repo — ตรวจแล้วตรงกับเปเปอร์ (test 1,962 ภาพ; opacity YES 1,135, pneumothorax 195, nodule/mass 295, fracture 72 ตรง Table 1 ของ Majkowska ทุกตัว; ภาพทั้งหมดอยู่ใน `test_list.txt`; metadata ตรง `Data_Entry_2017.csv` 100%; ฉลาก all-findings = majority ของผู้อ่าน ground-truth 3 คนตรง 100%)
> ใช้ prediction เดิมของ v4 (epoch 7) ไม่ได้ฝึกใหม่

### 5.1 ฉลาก NIH ผิดมากแค่ไหน (วัดบนภาพของเราเอง)

Four findings (ภาพที่ซ้อนกับ prediction ของเรา 942 ภาพ):

| | Airspace opacity = YES | NO |
|---|---|---|
| NIH Infiltration | 125 (73%) | 47 (27%) |
| NIH No Finding | **290 (38%)** | 480 (62%) |

All findings (PA ล้วน, 382 ภาพ, ฉลาก Infiltration โดยตรงจากรังสีแพทย์ 5 คน):

| | รังสีแพทย์: Infiltration = YES | NO | รังสีแพทย์: Abnormal = YES |
|---|---|---|---|
| NIH Infiltration (34) | **6 (18%)** | 28 (82%) | 22 (65%) |
| NIH No Finding (348) | 11 (3%) | 337 (97%) | **169 (49%)** |

- บนภาพ PA ฉลาก NIH "Infiltration" ตรงกับที่รังสีแพทย์เรียก Infiltration เพียง 18% — ต่ำกว่า PPV แบบคลินิก 36% ของ Oakden-Rayner
- ภาพ NIH "No Finding" ที่เป็น PA รังสีแพทย์บอกว่าผิดปกติเกือบครึ่ง (49%) และเห็น airspace opacity 38% (ทุก view)

### 5.2 AUC ของ baseline เมื่อเปลี่ยนเป็นฉลากรังสีแพทย์ (airspace opacity)

| กลุ่ม | n | AUC vs ฉลาก NIH | AUC vs ฉลากรังสีแพทย์ [95% CI patient bootstrap] |
|---|---|---|---|
| ทั้งหมด | 942 | 0.693 | 0.725 [0.684, 0.763] |
| **AP** | 545 | 0.678 | **0.774 [0.731, 0.815]** |
| **PA** | 397 | 0.515 | **0.465 [0.394, 0.539]** |

**ตีความ**
- ในกลุ่ม AP ฉลาก NIH กด AUC ไว้ราว 0.10 → **label noise เป็นเพดานจริง** ตามที่ SUMMARY ข้อ 2 สรุป
- ในกลุ่ม PA เปลี่ยนเป็นฉลากรังสีแพทย์แล้ว **ยังเท่าการเดาสุ่ม** (ฉลาก Infiltration ของรังสีแพทย์บน PA: 0.596 [0.437, 0.741] แต่มีภาพบวกเพียง 17 ภาพ) → **ตัดคำอธิบาย "PA ต่ำเพราะฉลากผิด" ออกได้** โมเดลไม่มีความสามารถตรวจ opacity ในภาพ PA จริง สมมติฐาน A (shortcut จากลักษณะภาพ AP) หรือ "PA ยากโดยธรรมชาติ" เหลืออยู่ ต้องใช้ XAI แยก

### 5.3 การเทียบ denoising เปลี่ยนไปเมื่อใช้ฉลากรังสีแพทย์ ← สำคัญต่อคำถามหลักของงาน

ความต่างของ AUC จาก baseline (paired patient-level bootstrap 1,000 รอบ บนภาพชุดเดียวกัน, `*` = CI ไม่คร่อม 0):

| Condition | ทั้งหมด vs NIH | ทั้งหมด vs รังสีแพทย์ | AP vs NIH | AP vs รังสีแพทย์ |
|---|---|---|---|---|
| median_L1 | +0.008 | **+0.024 \*** | +0.004 | **+0.030 \*** |
| median_L2 | −0.002 | **+0.025 \*** | −0.004 | **+0.031 \*** |
| median_L3 | −0.008 | **+0.028 \*** | −0.008 | **+0.050 \*** |
| clahe_dwt L1 | +0.003 | **+0.020 \*** | +0.012 | **+0.034 \*** |
| clahe_dwt L2 | +0.010 | **+0.041 \*** | +0.002 | **+0.054 \*** |
| clahe_dwt L3 | −0.025 | −0.001 | −0.024 | **+0.031 \*** |
| dae_clahe L1 | **+0.026 \*** | **+0.038 \*** | +0.028 | **+0.038 \*** |
| dae_clahe L2 | +0.004 | **+0.059 \*** | +0.001 | **+0.053 \*** |
| dae_clahe L3 | **−0.044 \*** | +0.006 | −0.039 | +0.034 |
| dft fusion | +0.018 | **+0.055 \*** | −0.008 | +0.027 |
| dwt fusion | −0.005 | **+0.060 \*** | −0.040 | **+0.037 \*** |

(CI เต็มอยู่ในไฟล์ผล)

**ตีความ**
1. **วัดกับฉลาก NIH: denoising แทบไม่ต่างจาก baseline** (มีนัยเพียง 2 จาก 11 และมีทิศทางตรงข้ามกัน) **วัดกับฉลากรังสีแพทย์: 9 จาก 11 condition ดีกว่า baseline อย่างมีนัย** (+0.02 ถึง +0.06) → label noise ของ NIH **บังผลของ denoising** ถ้าวัดแค่ฉลาก NIH จะสรุปผิดว่า denoising ไม่ช่วย
2. **รูปแบบ over-smoothing ชัดขึ้นเมื่อใช้ฉลากสะอาด** สำหรับ CLAHE+DWT และ DAE+CLAHE: L1 < **L2 (ดีสุด)** > L3 (ทั้งหมด: +0.020 → +0.041 → −0.001 และ +0.038 → +0.059 → +0.006) — L3 กลับไปเท่า baseline ซึ่งสอดคล้องกับสมมติฐานว่าลด noise แรงเกินทำลายรายละเอียดฝ้า ส่วน Median ไม่เห็นรูปแบบนี้ในช่วง kernel 3–7
3. ผลทั้งหมดมาจากกลุ่ม AP เท่านั้น บน PA ไม่มี condition ใดเกิน ~0.53

**ข้อควรระวังก่อนเขียนลงรายงาน**
- เทียบ 11 คู่ ไม่ได้ปรับ multiple comparisons (ถ้าใช้ Bonferroni α = 0.05/11 CI ต้องกว้างขึ้น — ผลของ L2 และ fusion มีโอกาสรอดสูงสุด)
- โมเดลแต่ละ condition ฝึก **seed เดียว** ความต่าง 0.02–0.06 อาจรวมความแปรปรวนจากการฝึก ควรยืนยันด้วยหลาย seed
- "Airspace opacity" กว้างกว่า Infiltration (รวม consolidation/pneumonia) และภาพที่ซ้อนกันมี 942 ภาพ (PA 397) ไม่ใช่ test ทั้งชุด
- ฉลาก all-findings มี Infiltration บวกเพียง 17 ภาพ และ Pneumonia 2 ภาพ ไม่พอสรุป

### 5.4 ข้อเสนอที่เปลี่ยนไปหลังผลนี้

- **ใช้ฉลากของ Google เป็นชุดประเมินหลักคู่กับฉลาก NIH** ในบทผลการทดลอง: รายงานทั้งสองแบบ และอธิบายว่าทำไมผลต่างกัน — เป็นข้อค้นพบที่มีคุณค่าในตัวเอง
- อัปเดต SUMMARY ข้อ 2 ได้ว่า "label noise อธิบายช่องว่างในภาพ AP ได้ (0.68 → 0.77) แต่ไม่ได้อธิบาย PA"
- อ้างอิงที่ต้องใส่ (ตาม README ของชุดฉลาก): Majkowska et al. (2020) และ Nabulsi, Z., Sellergren, A., Jamshy, S., et al. (2021). Deep learning for distinguishing normal versus abnormal chest radiographs and generalization to two unseen diseases tuberculosis and COVID-19. *Scientific Reports*, 11, 15523. https://doi.org/10.1038/s41598-021-93967-2 — และอ้าง NIH (Wang et al., 2017)

---

## 6. ตรวจซ้ำด้วย snapshot epoch 2 — ผลในส่วนที่ 3 และ 5 ใช้ได้เฉพาะ epoch 7 (เพิ่ม 2026-10-14)

`comparison/epochs/` มี snapshot 2 จุด: **epoch 2 = หลัง head warm-up (backbone ยัง freeze)** และ **epoch 7 = หลัง fine-tune layer4** (README ของโฟลเดอร์ระบุว่าทั้งสองไม่ใช่ `best.pt` ที่เลือกด้วย validation) เมื่อวัดทั้งสองจุด:

| Condition | NIH AUC ep2 → ep7 | NIH PA ep2 → ep7 | **ฉลากรังสีแพทย์ ep2 → ep7** | **รังสีแพทย์ PA ep2 → ep7** |
|---|---|---|---|---|
| baseline | 0.652 → 0.680 | 0.525 → 0.508 | **0.842 → 0.725** | **0.741 → 0.465** |
| median_L1 | 0.656 → 0.684 | 0.533 → 0.512 | 0.826 → 0.749 | 0.696 → 0.472 |
| median_L2 | 0.657 → 0.672 | 0.539 → 0.500 | 0.828 → 0.750 | 0.689 → 0.453 |
| median_L3 | 0.653 → 0.672 | 0.537 → 0.496 | 0.811 → 0.753 | 0.662 → 0.424 |
| clahe_dwt L1 | 0.651 → 0.669 | 0.537 → 0.504 | 0.841 → 0.745 | 0.771 → 0.471 |
| clahe_dwt L2 | 0.653 → 0.667 | 0.534 → 0.494 | 0.831 → 0.766 | 0.703 → 0.480 |
| clahe_dwt L3 | 0.638 → 0.649 | 0.508 → 0.465 | 0.786 → 0.724 | 0.640 → 0.403 |
| dae_clahe L1 | 0.637 → 0.674 | 0.503 → 0.487 | 0.812 → 0.763 | 0.676 → 0.508 |
| dae_clahe L2 | 0.634 → 0.674 | 0.489 → 0.494 | 0.794 → 0.784 | 0.630 → 0.517 |
| dae_clahe L3 | 0.619 → 0.636 | 0.477 → 0.460 | 0.745 → 0.731 | 0.586 → 0.441 |
| dft fusion | 0.615 → 0.677 | 0.480 → 0.511 | 0.746 → 0.780 | 0.592 → 0.527 |
| dwt fusion | 0.663 → 0.658 | 0.558 → 0.465 | 0.865 → 0.785 | 0.786 → 0.525 |

ความต่างจาก baseline เทียบฉลากรังสีแพทย์ (ทั้งหมด 942 ภาพ, CI แบบ Bonferroni 99.55% สำหรับ 11 คู่):

| Condition | epoch 2 | epoch 7 |
|---|---|---|
| median_L1 / L2 / L3 | −0.016 / −0.014 / **−0.031 \*** | **+0.024 \*** / +0.025 / +0.028 |
| clahe_dwt L1 / L2 / L3 | −0.001 / −0.011 / **−0.056 \*** | +0.020 / **+0.041 \*** / −0.001 |
| dae_clahe L1 / L2 / L3 | **−0.030 \*** / **−0.048 \*** / **−0.097 \*** | **+0.038 \*** / **+0.059 \*** / +0.006 |
| dft fusion | **−0.096 \*** | **+0.055 \*** |
| dwt fusion | +0.023 | **+0.060 \*** |

**ตีความ**
1. **Fine-tune layer4 บนฉลาก NIH ทำให้ AUC กับฉลาก NIH ขึ้น แต่ AUC กับฉลากรังสีแพทย์ลง** (baseline 0.842 → 0.725; PA 0.741 → 0.465) โมเดลกำลังเรียนรู้โครงสร้างของ label noise และลักษณะที่สัมพันธ์กับ AP แทน opacity จริง — สอดคล้องกับคำเตือน "fail silently" ของ Oakden-Rayner
2. **ข้อสรุป "PA ≈ เดาสุ่ม" (ส่วนที่ 3, 5.2) เป็นจริงเฉพาะหลัง fine-tune** ที่ epoch 2 โมเดลแยก opacity บน PA ได้ 0.64–0.79
3. **อันดับ denoising vs baseline กลับข้างระหว่าง epoch 2 และ 7**: ที่ epoch 2 baseline ดีที่สุดหรือเสมอ (แย่กว่าอย่างมีนัย 6 จาก 11) ที่ epoch 7 denoising ดีกว่า เพราะ baseline เสื่อมลงมากกว่าตอน fine-tune ไม่ใช่เพราะ denoising ทำให้โมเดลดีกว่าในตัวเอง
4. รูปแบบ over-smoothing (L3 แย่สุดในกลุ่มเดียวกัน) **คงที่ทั้งสอง epoch** สำหรับ CLAHE+DWT และ DAE+CLAHE — เป็นผลที่ทนทานที่สุดในชุดนี้

**ข้อสรุปที่ถูกต้อง ณ ตอนนี้: หลักฐานยังไม่พอจะบอกว่า denoising ดีกว่า baseline** (ผลขึ้นกับ checkpoint, seed เดียว, ยังไม่ได้วัด `best.pt`) สิ่งที่บอกได้คือ (a) ลด noise แรงเกิน (L3) แย่กว่าระดับเบากว่าอย่างสม่ำเสมอ และ (b) การ fine-tune บนฉลาก NIH ทำลายความสามารถตรวจ opacity จริงในภาพ PA

---

## 7. ผลของ `best.pt` (checkpoint ทางการของ v4, seed 42) — รันบน Colab 2026-10-14

> ผลเต็ม: [`Notebook_5 beta/results/google_labels_eval_best_seed_42.md`](../../Notebook_5%20beta/results/google_labels_eval_best_seed_42.md) · notebook: `Notebook_5 beta/eval_google_labels_colab.ipynb`
> seed 43 / 44 ยังไม่มีผลบน Drive

### 7.1 `best.pt` มาจาก epoch ไหน

ทุก condition มาจาก **ช่วง fine-tune layer4** (stage `layer4_head`) ไม่มีตัวใดมาจากช่วงฝึก head: epoch 6 สำหรับ 9 condition, epoch 9 สำหรับ clahe_dwt L2/L3, epoch 13 สำหรับ dwt fusion (early stopping หยุดหลัง best 3 epoch) → **ผลหลักของงานเป็นแบบ epoch 7 ไม่ใช่แบบ epoch 2** และจำนวน epoch ที่ฝึกไม่เท่ากันระหว่าง condition เป็นตัวแปรกวนอีกตัว

### 7.2 Baseline

| กลุ่ม | AUC vs ฉลาก NIH | AUC vs ฉลากรังสีแพทย์ [95% CI] |
|---|---|---|
| ทั้งหมด (942) | 0.691 | 0.737 [0.697, 0.775] |
| AP (545) | 0.681 | 0.815 [0.777, 0.854] |
| **PA (397)** | 0.542 | **0.465 [0.392, 0.540]** |

เหมือน snapshot epoch 7: AP ดีขึ้นมากเมื่อใช้ฉลากสะอาด แต่ **PA ยังเท่าเดาสุ่ม** (snapshot epoch 2 ได้ PA 0.741)

### 7.3 Denoising เทียบ baseline (ฉลากรังสีแพทย์, paired patient bootstrap)

| Condition | ทั้งหมด | AP |
|---|---|---|
| median L1 / L2 / L3 | +0.005 / +0.009 / **+0.030 \*** | +0.014 / +0.017 / **+0.025 \*** |
| clahe_dwt L1 / L2 / L3 | **+0.020 \*** / **−0.031 \*** / −0.019 | −0.000 / **−0.029 \*** / −0.008 |
| dae_clahe L1 / L2 / L3 | **+0.036 \*** / +0.021 / −0.024 | +0.012 / +0.009 / +0.000 |
| dft fusion | **+0.033 \*** | +0.017 |
| dwt fusion | +0.009 | −0.011 |

(`*` = 95% CI ไม่คร่อม 0 ยังไม่ปรับ multiple comparisons — ถ้าประมาณ Bonferroni สำหรับ 11 คู่ โดยขยาย CI ×1.45 จะเหลือเพียง **dae_clahe L1** ที่ยังไม่คร่อม 0)

### 7.4 สรุปสำหรับคำถาม "denoising ดีกว่า baseline ไหม"

1. **ไม่มีหลักฐานว่า denoising ช่วยอย่างเป็นระบบ**: ดีกว่าอย่างมีนัย 4 จาก 11, แย่กว่า 1, ที่เหลือไม่ต่าง; ทิศทางไม่คงที่ข้ามวิธีและข้ามกลุ่ม AP/PA; หลังปรับ multiple comparisons เหลือเพียง DAE+CLAHE L1 (+0.036) และยังเป็น seed เดียว
2. **ความแรงของ denoising มีผล — แนวโน้มที่สอดคล้องกับ over-smoothing**: DAE+CLAHE ลดลงตามความแรง (+0.036 → +0.021 → −0.024) และ CLAHE+DWT ระดับ L1 ดีกว่า L2/L3 — สอดคล้องกับ snapshot epoch 2 และ 7 ที่ L3 แย่สุดในกลุ่มเดียวกัน **ยกเว้น Median** ที่ L3 กลับดีสุด (kernel 3–7 บนภาพ 224 px อาจยังไม่แรงพอจะ over-smooth)
3. **ปัญหาใหญ่กว่า denoising คือโมเดลหลัง fine-tune ตรวจภาพ PA ไม่ได้** ทุก condition (PA 0.40–0.53 vs ฉลากรังสีแพทย์) — ข้อนี้ควรอยู่ในบทอภิปรายและเป็นเหตุผลหลักของส่วน XAI
4. ประโยคที่เขียนในรายงานได้: *"เมื่อประเมินด้วยฉลากรังสีแพทย์ (Majkowska et al., 2020) การลดสัญญาณรบกวนไม่ได้ให้ประสิทธิภาพที่ดีกว่าภาพดิบอย่างสม่ำเสมอ มีเพียง DAE+CLAHE ระดับเบาที่ดีกว่าอย่างมีนัยสำคัญ และสำหรับ DAE+CLAHE กับ CLAHE+DWT ประสิทธิภาพลดลงเมื่อเพิ่มความแรงของการลดสัญญาณรบกวน ซึ่งสอดคล้องกับสมมติฐาน over-smoothing (ผลจาก seed เดียว)"*

### 7.5 สิ่งที่ต้องทำต่อ

- **ฝึก seed 43 / 44** (config มีอยู่แล้ว) แล้วรัน notebook เดิมซ้ำ — notebook ข้าม seed ที่ยังไม่มีผลให้อัตโนมัติ ถ้า DAE+CLAHE L1 ยังดีกว่าและแนวโน้มตามความแรงยังอยู่ จึงสรุปได้
- ตัดสินใจเรื่องเกณฑ์เลือก checkpoint (ส่วนที่ 6): `best.pt` เลือกด้วย loss บนฉลาก NIH และเลือกช่วง fine-tune เสมอ ซึ่งเป็นช่วงที่ PA พัง
- XAI: Grad-CAM บน `best.pt` เทียบ snapshot epoch 2 แยก PA/AP

---

## ภาคผนวก: โค้ดที่ใช้ (รันจาก root ของ repo ด้วย `.gpu/Scripts/python.exe`)

```python
import pandas as pd, numpy as np, glob
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

meta = pd.read_csv("data/versions/3/Data_Entry_2017.csv").rename(columns={"Image Index": "image"})

# AUC แยก view / follow-up สำหรับทุก condition
for f in sorted(glob.glob("comparison/epochs/*/epoch_7/test_predictions.csv")):
    p = pd.read_csv(f).merge(meta, on="image")
    y, s, ap, fu = p.label, p.prob_infiltration, p["View Position"] == "AP", p["Follow-up #"]
    print(f, roc_auc_score(y, s), roc_auc_score(y[~ap], s[~ap]), roc_auc_score(y[ap], s[ap]),
          roc_auc_score(y[fu == 0], s[fu == 0]), roc_auc_score(ap, s))

# metadata-only baseline (patient-grouped CV บน test)
p = pd.read_csv("comparison/epochs/baseline/epoch_7/test_predictions.csv").merge(meta, on="image")
X = np.c_[p["View Position"] == "AP", np.log1p(p["Follow-up #"]), p["Patient Age"], p["Patient Gender"] == "M"]
prob = cross_val_predict(LogisticRegression(max_iter=1000), X, p.label, groups=p["Patient ID"],
                         cv=GroupKFold(5), method="predict_proba")[:, 1]
print(roc_auc_score(p.label, prob))

# patient-level bootstrap CI
def patient_bootstrap_auc(y, s, g, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s, g = map(np.asarray, (y, s, g))
    groups = {k: np.where(g == k)[0] for k in np.unique(g)}
    keys = np.array(list(groups))
    aucs = []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(aucs, [2.5, 97.5])
```
