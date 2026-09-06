# Guideline: การเพิ่มประสิทธิภาพการตรวจจับภาวะปอดอักเสบ (Infiltration) จากภาพเอกซเรย์
### ด้วยการลดสัญญาณรบกวนและปัญญาประดิษฐ์ที่อธิบายได้ (Denoising + Explainable AI)

**ผู้จัดทำ:** นายปิยะพล ตุ่นป่า (673380280-2), นายสัพพัญญู คำตุ้ม (673380066-4)
**อาจารย์ที่ปรึกษา:** อ. ดร.พบพร ด่านวิรุทัย
**สถานะเอกสาร:** ต่อยอดจากบทที่ 1-3 (Literature Review เสร็จสมบูรณ์) → เริ่มเฟส Implementation
**วัตถุประสงค์ของไฟล์นี้:** ใช้เป็น single source of truth เพื่อส่งต่อได้ทันทีหากแชทหลุด/หมด quota โดยไม่ต้องอธิบายบริบทซ้ำ

---

## 0. สรุปบริบทงานวิจัย (จากบทที่ 1-3)

- **Research Gap หลัก**: ยังไม่มีงานวิจัยที่ศึกษา Denoising + XAI แบบเจาะจงภาวะ Infiltration (ลักษณะฝ้ากระจาย ขอบเขตไม่ชัด/Diffuse-Hazy Opacity) มีเพียง Sheu et al. (2023) ที่แตะภาวะนี้โดยตรง แต่ใช้วิธีคัดภาพทิ้งแทนการกู้คืนภาพ
- **Gap 6 ข้อที่งานนี้จะเติมเต็ม**:
  1. ไม่มีเทคนิคลดสัญญาณรบกวนที่ออกแบบเฉพาะสำหรับฝ้ากระจาย
  2. ไม่มีการทดลอง Fusion Denoising กับ Infiltration โดยตรง
  3. ไม่รู้ว่า Over-smoothing ทำลายรายละเอียดฝ้าบางหรือไม่
  4. ความน่าเชื่อถือของ XAI (Grad-CAM/SHAP) กับรอยโรคกระจายตัวยังไม่ชัดเจน
  5. ปัญหา Label Noise รุนแรงเป็นพิเศษเพราะ Infiltration ประเมินแบบ Subjective
  6. ไม่มี Fair Comparison ระหว่าง Traditional Filter กับ Deep Learning Denoising บนงานเดียวกัน
- **หลักการสำคัญที่ต้องยึดตลอดโปรเจกต์**: ห้ามทำแค่ "ลองสูตรสำเร็จรูปแล้วดูว่าตัวไหนชนะ" (นั่นคือสิ่งที่ Rahman et al. 2021 ทำไปแล้ว) — ต้องมีตัวแปรควบคุมความแรงของการลด noise เพื่อตอบคำถาม Over-smoothing โดยเฉพาะ ซึ่งเป็นส่วนที่ทำให้งานนี้ต่างจากงานเดิม

---

## 1. Dataset — การตัดสินใจที่ล็อกไว้แล้ว

**แหล่งข้อมูล:** NIH ChestX-ray14 (ChestX-ray8 extended)
- 112,120 ภาพ frontal-view, 30,805 patients, 1024×1024 PNG
- Label mining ด้วย NLP จากรายงานรังสีแพทย์ ความแม่นยำโดยประมาณ >90% (**มี Label Noise จริง ไม่ใช่แค่สมมติฐาน**)
- Bounding box มีเฉพาะ ~984 กล่อง จาก 880 ภาพ (รวมทุกโรค) — เป็น subset แยกต่างหาก

### 1.1 Data Split
- **ใช้ train_val_list.txt / test_list.txt ตามที่ NIH ให้มา** (patient-level split ป้องกัน patient leakage)
- ห้าม re-split เอง

### 1.2 Task Type
- **Binary Classification: Infiltration vs Normal only**
- Negative class = เฉพาะภาพที่ "ไม่มีโรคใดเลย" (ไม่ใช่ "ไม่มี Infiltration" เฉย ๆ)
- เหตุผล: Infiltration มี co-occurrence สูงกับ Effusion / Atelectasis / Consolidation / Edema (ดู co-occurrence matrix ใน README) หากรวมภาพที่มีโรคอื่นเป็น negative จะทำให้โมเดลเรียนรู้สัญญาณของโรคอื่นปนเข้ามา ไม่ตรงกับโจทย์ที่ต้องการศึกษาเฉพาะลักษณะฝ้ากระจาย
- **Trade-off ที่ต้องยอมรับ**: ขนาดข้อมูลเทรนจะเล็กลงมากจากทั้งหมด 112,120 ภาพ ต้องเช็คจำนวนจริงหลังกรองก่อนเริ่มเทรน (ถ้าน้อยเกินไปอาจต้องพิจารณา class imbalance handling เช่น weighted loss หรือ oversampling)

### 1.3 XAI Evaluation Set — ยืนยันแล้วว่าใช้ได้จริง
- Bbox ของ Infiltrate: **123 กล่อง จาก 123 ภาพ**
- ยืนยันแล้วว่า **100% อยู่ใน test_list.txt** → ไม่มี data leakage สำหรับการประเมิน XAI
- n=123 มากพอสำหรับรายงานเชิงปริมาณ (ไม่ใช่แค่ qualitative case study) แต่ **ต้องรายงานคู่กับ 95% CI (bootstrap resampling)** ไม่ใช่ค่าเฉลี่ยเดี่ยว ๆ เพราะ sample size ยังไม่ใหญ่มาก

### 1.4 Action Items ก่อนเริ่มเทรน
- [ ] Query metadata (Data_Entry_2017.csv) กรองภาพ Infiltration-only vs Normal-only ตาม Finding Labels
- [ ] นับจำนวนภาพจริงหลังกรอง แยกตาม train_val / test
- [ ] เช็ค class imbalance ratio (Infiltration : Normal)
- [ ] เตรียม loader ที่ join กับ BBox_List_2017.csv สำหรับ subset ที่มี bbox (ใช้เฉพาะตอนประเมิน XAI ไม่ใช้เทรน)

---

## 2. Model Architecture

- **Backbone**: CNN pretrained (ResNet50 หรือเทียบเท่า — ยืนยันชื่อโมเดลที่แน่นอนก่อนเริ่ม coding เพราะมีความกำกวมจากที่พิมพ์ว่า "sp50" ในตอนแรก)
- ใช้ **backbone และ hyperparameter เดียวกันทุก experiment** (fix ทุกอย่างยกเว้นตัวแปร denoising) เพื่อให้เป็น Fair Comparison ตาม Gap ข้อ 6
- Fine-tuning strategy: Transfer Learning จาก ImageNet pretrained weights, freeze early layers, fine-tune late layers + classification head

---

## 3. Denoising Technique — Shortlist (ตกลงแล้ว)

ไม่ทดสอบทุกเทคนิคที่มีในวรรณกรรม (สิ้นเปลืองและซ้ำกับ Rahman et al. 2021) แต่เลือก 4 กลุ่มที่ครอบคลุมทุกแนวทางสำคัญ:

| กลุ่ม | เทคนิค | อ้างอิงและเหตุผล |
|---|---|---|
| Baseline | ภาพดิบ (ไม่ denoise) | ใช้เป็นเส้นฐานเทียบว่า denoising ช่วยจริงหรือไม่ |
| Traditional เดี่ยว | Median Filter | Chattopadhyay (2022): ดีที่สุดในกลุ่ม traditional filter สำหรับ Poisson noise แบบ X-ray, PSNR 34.52 dB |
| Fusion แบบที่ 1 | CLAHE + DWT | Chutia et al. (2023/2024): Accuracy 97.77% บน Atelectasis/Pneumothorax — ทดสอบว่า transfer มาที่ Infiltration ได้ไหม |
| Fusion แบบที่ 2 | DAE + CLAHE | Thamilarasi et al. (2025): ผลดีสุดในรีวิว — ต้องเทรน DAE เองด้วยชุดข้อมูลนี้ (ไม่ใช่ pretrained model สำเร็จรูป) |

### 3.1 ตัวแปรเสริมที่เป็น Contribution ใหม่ของงานนี้ (สำคัญที่สุด — ห้ามข้าม)
- เพิ่ม **noise-reduction strength เป็นตัวแปรควบคุม** ในแต่ละเทคนิค เช่น:
  - CLAHE: ปรับค่า Clip Limit (เช่น 2.0, 4.0, 8.0)
  - DWT: ปรับ threshold level ของการตัดความถี่สูง
  - Median Filter: ปรับขนาด kernel
- วัดความสัมพันธ์ระหว่างความแรงของการลด noise กับ:
  1. Classification accuracy/F1
  2. IoU ของ XAI heatmap เทียบกับ bbox ground truth
- จุดที่คาดว่าจะพบ (สมมติฐานที่ต้องพิสูจน์): ที่ระดับความแรงสูงเกินจุดหนึ่ง accuracy อาจเพิ่มขึ้นต่อ แต่ IoU ของ XAI จะเริ่มลดลง (สัญญาณของ Over-smoothing ที่กลบรายละเอียดฝ้าบาง) — นี่คือคำตอบของ Gap ข้อ 3

---

## 4. Experiment Design

### 4.1 Matrix การทดลอง
- 4 denoising groups × 2-3 ระดับความแรง (strength level) = 8-12 experiment runs
- ทุก run ใช้ backbone, hyperparameter, train/test split เดียวกัน (fix ตาม Section 2)

### 4.2 Evaluation Metrics
**ฝั่ง Classification:**
- Accuracy, Precision, Recall, F1-score (ตาม test_list.txt แบบเต็ม)
- รายงานพร้อม confusion matrix

**ฝั่ง XAI (บน subset n=123 ที่มี bbox เท่านั้น):**
- ใช้ Grad-CAM และ SHAP (ตามแนวทาง Sheu et al. 2023)
- คำนวณ IoU หรือ Pointing Game accuracy ระหว่าง heatmap กับ bbox ground truth
- รายงานค่าเฉลี่ย + 95% CI (bootstrap, เพราะ n=123)
- เปรียบเทียบ heatmap ก่อน/หลัง denoising แต่ละระดับความแรง

### 4.3 Fair Comparison Checklist (ตอบ Gap ข้อ 6)
- [ ] Train/val/test split เหมือนกันทุก run
- [ ] Backbone architecture เหมือนกันทุก run
- [ ] Hyperparameter (learning rate, batch size, epoch) เหมือนกันทุก run
- [ ] Random seed fix เพื่อ reproducibility
- [ ] รายงานเวลา/ทรัพยากรที่ใช้ต่อเทคนิค (เพื่อเปรียบเทียบ cost-effectiveness ด้วย ไม่ใช่แค่ accuracy)

---

## 5. Label Noise Mitigation (ตอบ Gap ข้อ 5)

- ยอมรับข้อจำกัดว่า label เป็น NLP-mined accuracy >90% ไม่สามารถแก้ไขได้ทั้งหมดในระยะเวลาโปรเจกต์นี้
- แนวทางบรรเทา:
  - รายงานข้อจำกัดนี้อย่างชัดเจนในบทวิเคราะห์ผล (ไม่ปิดบัง)
  - พิจารณา cross-check ผลลัพธ์ของโมเดลกับ subset ที่มี bbox (n=123) ซึ่งน่าเชื่อถือกว่า label แบบ NLP-mined ทั้งหมด เป็นตัวช่วยยืนยันคุณภาพ

---

## 6. Timeline (แนวทาง — ปรับตามความคืบหน้าจริง)

| ระยะ | งาน |
|---|---|
| Phase 1 | เตรียมข้อมูล: กรอง Infiltration-only vs Normal-only, join bbox, เตรียม data loader |
| Phase 2 | Implement denoising pipeline ทั้ง 4 กลุ่ม × strength levels |
| Phase 3 | Train baseline model (no denoise) ยืนยัน pipeline ทำงานถูกต้องก่อน |
| Phase 4 | รัน experiment matrix เต็มรูปแบบ (8-12 runs) |
| Phase 5 | ประเมิน Classification metrics ทุก run |
| Phase 6 | ประเมิน XAI (Grad-CAM/SHAP) บน subset n=123 พร้อม bootstrap CI |
Phase 6 (XAI Evaluation) 

การรายงาน IoU/Pointing-game accuracy ของ XAI eval set (n=123) ให้แบ่งเป็น 3 ระดับเสมอ:
1. Overall (n=123) — ตัวเลขหลักสำหรับเทียบระหว่าง denoising technique
2. Pure-only (n=19) — รายงานเป็น exploratory/secondary เพื่อดูแนวโน้มบนกรณีที่ไม่มี confound เลย (ไม่ต้องใช้ CI แคบเพราะ sample เล็ก แต่ต้อง report น้ำหนักตัวเลขตามจริง)
3. Mixed แยกตามกลุ่มความเสี่ยง confound:
   - High-risk overlap (co-occur กับ Effusion หรือ Atelectasis) n≈70-79 ภาพ (มีซ้อนกันบางภาพ)
   - Low-risk overlap (co-occur กับโรคอื่นที่ไม่ใช่สองตัวนี้)
เพิ่มคอลัมน์ risk_flag ใน xai_eval_set.csv ระบุว่าภาพนั้นเป็น pure / mixed-high-risk / mixed-low-risk เพื่อให้ filter ได้ตอนวิเคราะห์ผลจริงใน Phase 6

| Phase 7 | วิเคราะห์ความสัมพันธ์ strength vs accuracy vs XAI-IoU หา over-smoothing point |
| Phase 8 | สรุปผล เขียนรายงานบทที่ 4-5 |

---

## 7. Open Questions / จุดที่ต้องตัดสินใจต่อ

- [ ] ยืนยันชื่อ backbone model ที่แน่นอน (ResNet50 หรืออื่น)
- [ ] ตัดสินใจ negative-class filtering เชิงปฏิบัติ: กรองยังไงถ้าภาพมี label ว่าง/ไม่ชัดเจน
- [ ] เลือก framework implementation (PyTorch/TensorFlow) และ environment (Colab/local GPU)
- [ ] วางแผน compute budget (DAE ต้องเทรนเอง ใช้เวลา/ทรัพยากรมากกว่าเทคนิคอื่น)

---

## 8. Reference สรุป (เทคนิคหลักที่ใช้อ้างอิงในแผนนี้)

- Chattopadhyay, S. (2022). *A study on various common denoising methods on chest X-ray images.*
- Chutia, U., Tewari, A. S., & Singh, J. P. (2024). *Collapsed lung disease classification by coupling denoising algorithms and deep learning techniques.*
- Thamilarasi, V., Asaithambi, A., & Roselin, R. (2025). *Enhanced ensemble segmentation of lung chest X-ray images by denoising autoencoder and CLAHE.*
- Sheu, R. K., et al. (2023). *Interpretable classification of pneumonia infection using eXplainable AI (XAI-ICP).*
- Wang, X., et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks.* (ที่มาของ dataset)

(รายการอ้างอิงฉบับเต็มอยู่ในเอกสารโครงงานบทที่ 2-3)
