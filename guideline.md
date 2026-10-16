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

## 9. v3 release — สเปกที่ผู้ใช้ให้สร้าง (12 ตุลาคม 2026)

Implementation อยู่ `Notebook_3 release/` บน branch `codex/notebook-v3-release` รายละเอียดใน README/RUN_GUIDE/configs ของ release

- Dataset Colab: `/content/drive/MyDrive/nih -xray/data/3` ที่มีอยู่บน Google Drive; ไม่สร้าง/อัปโหลด ZIP ภาพใหม่
- ใช้ pure Infiltration vs strictly No Finding ตามเดิม: Train46,108 (40,371/5,737), Validation11,719 (10,129/1,590), fullofficialfilteredTest12,081 (9,861/2,220)
- Reuse validation-patient partition เดิม seed42/20% reserved4,815patients; ไม่ re-split official lists เก็บ400trainเดิมครบ
- XAI123/115patients: pure19/mixedhigh64/mixedlow40, background32ภาพเดิมจากtrain; patient-bootstrap95%CI2,000draws
- หลัก12conditions: baseline, Median3/5/7, controlledDWT→CLAHEthreshold0.5/1/2 (db1/depth3/clip2คงที่), controlledDAE→CLAHEα0.25/0.5/1(clip2คงที่), DWT5viewfusion และDFT3viewfusion
- เก็บ v2 parameterpreset เป็นlegacyแยก Controlledstrengthไม่ได้ใช้CLAHEclipเป็นตัวแทนdenoisingstrengthโดยลำพัง
- ResNet50ImageNetV2ทุกmethod: headwarmup3epochs จากนั้นlayer4+head สูงสุด20epochs พร้อม early stopping patience3 (นับหลัง warmup และหลัง best epoch; ใช้กฎเดียวกันทุก method), freezeearlylayers/BatchNormstatisticsทั้งหมด; AdamW/dropout0.3/weightedBCEtrain-onlypos_weight≈7.037/effectivebatch16
- Train-only geometric augmentation ทุก method: หมุน ±7°, เลื่อน ±5%, scale 0.95–1.05, reflection padding, ใช้ transform เดียวกันทุก view ของภาพ; ไม่มี brightness/contrast/noise/flip เพื่อไม่ปนกับผล denoising/CLAHE; validation/test/XAI ไม่ augment
- 3trainingseeds42/43/44 แต่manifestsคงที่ รวม36CNN+3DAE; bestเลือกvalidationbalancedBCE ไม่ใช้test/XAIเลือกcheckpointหรือthreshold
- Classification threshold ของแต่ละ model เลือกจาก Youden J บน validation predictions ของ model นั้น; รายงาน metrics ที่ threshold 0.5 คู่กันเป็น reference และ paired delta ใช้ threshold ของแต่ละ model
- เหตุผลที่เปลี่ยน protocol (13 ตุลาคม 2026): run แรก (`runs/release_full_dae_b064_gpu`, ไม่มี augmentation, train ครบ20epochs, threshold0.5) baseline seed42 overfit หลังเปิด layer4: validation balanced BCE ต่ำสุดที่ epoch4 (0.640) แล้วขึ้นถึง 3.5–6.6 ขณะ train-eval ลงถึง 0.01; test ROC-AUC 0.655, accuracy 0.377 (threshold0.5 ทาย positive 76%) จึงยกเลิก run นั้น ผลใช้เป็นหลักฐานใน Methods เท่านั้น ไม่รวมกับ run ใหม่ `runs/release_v4` ยกเว้น DAE ที่ train ครบแล้ว (เช่น seed42) ซึ่งนำเข้าได้ด้วย `tools/import_dae.py` เพราะ DAE config/code/data/versions เหมือนเดิมทุกอย่าง; checkpoint ที่นำเข้ามี `imported_from` บันทึก hash ของ run เดิม
- Training DAE/CNN ใช้ non-deterministic cuDNN (benchmark) + channels_last เพื่อความเร็ว (~5× บน L4); seed ยังคุมลำดับข้อมูล, noise และ weight เริ่มต้น แต่ weight ที่ได้อาจต่างกันเล็กน้อยระดับทศนิยมเมื่อรันซ้ำ ความแปรปรวนนี้รวมอยู่ใน mean±SD ข้าม 3 seeds; DAE denoising inference, test evaluation และ Grad-CAM/SHAP ยัง deterministic
- Runtimecacheuint8shardsอยู่ColabVM, results/checkpointsอยู่Drive; durablemetadata/manifest/sourcepixel/code/versionhashesใช้ตรวจresume
- ผลvalidationและข้อจำกัดอยู่ `Notebook_3 release/VALIDATION.md` การทดสอบชุดเล็กไม่ใช่ผลclassificationงานวิจัยเต็ม

## 10. v7 beta — noise-controlled class160 pilot (16 October 2026)

- โค้ดแยก `Notebook_7 beta/`, branch `codex/notebook-v7-beta`; ไม่เปลี่ยนผลหรือโค้ดของ v4 เดิม
- ทำงานจริงบน Colab: อ่าน `/content/drive/MyDrive/nih -xray/data/3`, เก็บ audit/ชุดภาพที่คัด/checkpoints/results ที่ `/content/drive/MyDrive/nih -xray/Notebook_7_beta`
- วิเคราะห์ 112,120 ภาพด้วย db2 wavelet MAD บน patch64 ในกลาง80%, ตัด patch สีคงที่/อิ่มตัว, ใช้ patch gradient ต่ำสุด25%; ค่าเป็น estimated noise ไม่ใช่หลักฐานว่าภาพไม่มี noise
- คัดคะแนนต่ำสุดต่อ class/partition (tie=image ID): train160+160, validation40+40, test50+50; reuse frozen patient partition v4.3, Infiltration-only vs strictly No Finding; ไม่ลดเกณฑ์เมื่อภาพไม่พอ
- ใช้ 12 methods v4.3 ตรงสูตรเดิม ไม่มี strength L4; DWT fusion ยังคง decomposition4ระดับ และ DFT3views ตามเดิม
- Gaussian, Poisson, Poisson–Gaussian, salt-and-pepper, speckle อย่างละ3ระดับ + ภาพก่อนเติม รวม16สภาวะ; trainสุ่มหนึ่งสภาวะต่อภาพต่อepochเหมือนกันทุกmethod, validation/testใช้corruptionคงที่; parametersอยู่ใน `v7/noise.py` และบันทึกในexperiment.json
- seed42รอบเดียวสำหรับ beta pilot: 1DAEใหม่จากtrainเท่านั้น +12CNN; ResNet50ImageNetV2/frozenearlylayers/BN/warmup3/รวมสูงสุด20epochs/patience3/geometryaugmentation ตามv4.3, pos_weight=1เพราะtrainสมดุล
- DAE effective optimizer batch64 ใช้ microbatch8พร้อมaccumulationเพื่อรองรับGPUเล็ก; DAE BNเห็นbatch8ตามที่บันทึกในprotocol; targetsคือภาพก่อนเติมnoise ไม่ใช่clean detector truth
- bestเลือกbalanced validation BCE รวม16states และเลือกYouden thresholdหนึ่งค่าต่อmodelใช้ทุกteststate; รายงานthreshold0.5คู่กัน, patient-bootstrap95%CI2,000draws และpaired deltaเทียบbaseline
- PSNR/SSIMใช้ภาพก่อนเติมเป็นreference proxy; frequency fusionไม่มีPSNR/SSIMเพราะเป็นrepresentation; ผลใช้กับcohortที่คัดnoiseต่ำเท่านั้น ไม่ใช่ผลแทนNIHทั้งหมด
- ผู้ใช้ให้ทำCNNก่อน ยังไม่รันXAI; เก็บnative bbox123ภาพ, SHAPbackground16+16จากtrain, checkpoint/model loader/input gradients/geometry/view order เพื่อใช้ต่อได้
- ปิดTF32สำหรับmatmul/cuDNNเพื่อให้ผลbatchเดี่ยวของXAIสอดคล้องกับCNNevaluation; AMP trainingยังใช้ตามconfig และเก็บpackage/code/datahashเพื่อปฏิเสธresumeที่ไม่เข้ากัน
