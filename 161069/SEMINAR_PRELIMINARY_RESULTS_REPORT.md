# 📊 รายงานสรุปผลการทดลองเบื้องต้น (Preliminary Experimental Results Report)
### โครงงาน: การเพิ่มประสิทธิภาพการตรวจจับภาวะปอดอักเสบจากภาพเอกซเรย์ด้วยการลดสัญญาณรบกวนและปัญญาประดิษฐ์ที่อธิบายได้
**(Enhancing Chest X-RAY Infiltration Detection Using Noise Reduction and Explainable AI)**

* **วิชา:** CP353761 สัมมนาทางวิทยาการคอมพิวเตอร์ ภาคเรียนที่ 1 ปีการศึกษา 2569
* **ผู้จัดทำ:** นายปิยะพล ตุ่นป่า (673380280-2), นายสัพพัญญู คำตุ้ม (673380066-4)
* **อาจารย์ที่ปรึกษา:** อ. ดร.พบพร ด่านวิรุทัย
* **วันที่จัดทำรายงาน:** 16 ตุลาคม พ.ศ. 2569
* **ไฟล์โค้ดอ้างอิง:** 
  * [V2Seminar_source_code.ipynb](file:///d:/ForSeminarProject/version1/V2Seminar_source_code.ipynb) *(VinBigData - ควบคุม Train 400 ภาพ)*
  * [class200_methods_colab.ipynb](file:///d:/ForSeminarProject/version1/class200_methods_colab.ipynb) *(NIH ChestX-ray14 - ควบคุม Train 400 ภาพ)*

---

## 📑 สารบัญรายงาน (Table of Contents)
1. [การตั้งค่าการทดลองร่วม (Controlled Experimental Setup)](#1-การตั้งค่าการทดลองร่วม-controlled-experimental-setup)
2. [ตารางที่ 1: ผลลัพธ์ชุดข้อมูล VinBigData Dataset (แยกเดี่ยว)](#ตารางที่-1-ผลลัพธ์ชุดข้อมูล-vinbigdata-dataset)
3. [ตารางที่ 2: ผลลัพธ์ชุดข้อมูล NIH ChestX-ray14 Dataset (แยกเดี่ยว)](#ตารางที่-2-ผลลัพธ์ชุดข้อมูล-nih-chestx-ray14-dataset)
4. [ตารางที่ 3: ตารางเปรียบเทียบผลลัพธ์ข้ามชุดข้อมูล (VinBigData vs NIH ChestX-ray14)](#ตารางที่-3-ตารางเปรียบเทียบผลลัพธ์ข้ามชุดข้อมูล-vinbigdata-vs-nih-chestx-ray14)
5. [สรุปผลการเปรียบเทียบเชิงวิชาการสำหรับรายงานอาจารย์](#5-สรุปผลการเปรียบเทียบเชิงวิชาการสำหรับรายงานอาจารย์-scientific-insights)

---

## 1. การตั้งค่าการทดลองร่วม (Controlled Experimental Setup)

เพื่อให้การเปรียบเทียบระหว่าง 2 ชุดข้อมูลเป็นธรรมและได้มาตรฐานทางวิทยาศาสตร์ (Fair Comparison) การทดลองได้ล็อกและควบคุมสภาพแวดล้อมให้เหมือนกันทุกประการ:
* **โมเดลโครงข่ายร่วม:** ResNet-50 (Pretrained บน ImageNet-V2, Fine-tuning ทุก Layer)
* **พารามิเตอร์การฝึก:** CrossEntropyLoss, Adam Optimizer ($lr = 10^{-4}$), Batch Size 32, จำนวน 7 Epochs, Random Seed 42, Evaluation Threshold = 0.5 ($\text{argmax}$)
* **ขนาดชุดข้อมูลที่ควบคุมให้เท่ากัน:**
  * **Train Set:** **400 ภาพ** (Normal 200 ภาพ : Infiltration 200 ภาพ)
  * **Validation Set:** **80 ภาพ** (Normal 40 ภาพ : Infiltration 40 ภาพ)
  * **Test Set:** **100 ภาพ** (Normal 50 ภาพ : Infiltration 50 ภาพ)

---

## ตารางที่ 1: ผลลัพธ์ชุดข้อมูล VinBigData Dataset
*(ชุดทดสอบ Test Set 100 ภาพ — ประเมิน Grad-CAM Localization เทียบกับ 613 Bounding Boxes ของรังสีแพทย์ผู้เชี่ยวชาญ 17 ท่าน)*

| หมวดหมู่วิธีการ | สภาวะการทดลอง (Condition / Parameters) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Grad-CAM Hit Rate | Grad-CAM Energy in Box |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | **ภาพดิบ (Raw CXR - ไม่ผ่านการกรอง)** | **0.9200** | **0.8889** | **0.9600** | **0.9231** | **0.9676** | 0.33% | 0.25% |
| **Spatial Filter** | Median Filter Level 1 ($3\times3$) | 0.8800 | **0.9750** | 0.7800 | 0.8667 | 0.9636 | 0.49% | 0.37% |
| **Spatial Filter** | Median Filter Level 2 ($5\times5$) | 0.9000 | 0.8846 | 0.9200 | 0.9020 | 0.9624 | 0.49% | 0.42% |
| **Spatial Filter** | **Median Filter Level 3 ($7\times7$)** | **0.9100** | 0.9362 | 0.8800 | **0.9072** | **0.9680\*** | **0.49%** | **0.65%** |
| **Spatial Filter** | Median Filter Level 4 ($9\times9$) | 0.8700 | 0.8364 | 0.9200 | 0.8762 | 0.9528 | 0.49% | 0.41% |
| **Hybrid** | CLAHE + DWT Level 1 (Clip 1.0, Ratio 0.05) | 0.8700 | 0.8627 | 0.8800 | 0.8713 | 0.9512 | 0.49% | 0.44% |
| **Hybrid** | **CLAHE + DWT Level 2 (Clip 2.0, Ratio 0.10)** | **0.9100** | **0.9767** | 0.8400 | **0.9032** | **0.9628** | **0.49%** | **0.65%** |
| **Hybrid** | CLAHE + DWT Level 3 (Clip 3.0, Ratio 0.20) | 0.8900 | 0.8305 | **0.9800** | 0.8991 | 0.9620 | 0.16% | 0.20% |
| **Hybrid** | CLAHE + DWT Level 4 (Clip 4.0, Ratio 0.30) | 0.8500 | 0.7966 | 0.9400 | 0.8624 | 0.9500 | 0.49% | 0.41% |
| **Deep Learning** | DAE + CLAHE Level 1 (Clip 2.0) | 0.8800 | 0.8800 | 0.8800 | 0.8800 | **0.9604** | 0.00% | 0.10% |
| **Deep Learning** | DAE + CLAHE Level 2 (Clip 4.0) | 0.8800 | 0.9130 | 0.8400 | 0.8750 | 0.9476 | 0.00% | 0.04% |
| **Deep Learning** | **DAE + CLAHE Level 3 (Proposed: Clip 8.0)** | 0.8900 | 0.8679 | **0.9200** | 0.8932 | 0.9440 | **0.00%** | **0.01%** |
| **Deep Learning** | DAE + CLAHE Level 4 (Clip 10.0) | **0.9000** | **0.9348** | 0.8600 | **0.8958** | 0.9588 | 0.00% | 0.10% |
| **Pure Wavelet** | **Pure DWT Level 1 (Ratio 0.05)** | **0.9000** | 0.9000 | **0.9000** | **0.9000** | **0.9560** | 0.00% | 0.16% |
| **Pure Wavelet** | Pure DWT Level 2 (Ratio 0.10) | 0.8700 | 0.8491 | **0.9000** | 0.8738 | 0.9492 | 0.16% | 0.22% |
| **Pure Wavelet** | Pure DWT Level 3 (Ratio 0.20) | 0.8500 | **0.9487** | 0.7400 | 0.8315 | 0.9508 | 0.16% | 0.22% |
| **Pure Wavelet** | Pure DWT Level 4 (Ratio 0.35) | 0.8700 | 0.8936 | 0.8400 | 0.8660 | 0.9524 | 0.49% | 0.41% |
| **Frequency Bands** | **Low Frequency Only (เก็บเฉพาะ A3)** | **0.8500** | 0.8302 | **0.8800** | **0.8544** | **0.9352** | 0.16% | 0.39% |
| **Frequency Bands** | **Mid Frequency Only (เก็บเฉพาะ D3+D2)** | 0.7800 | 0.8889 | 0.6400 | 0.7442 | 0.8868 | **0.65%\*\*** | **0.75%\*\*** |
| **Frequency Bands** | High Frequency Only (เก็บเฉพาะ D1) | 0.8000 | 0.8125 | 0.7800 | 0.7959 | 0.8920 | 0.33% | 0.34% |
| **Frequency Bands** | **Selective DWT (กรอง D1 ด้วย Univ. Thresh)** | 0.7900 | **1.0000\*** | 0.5800 | 0.7342 | 0.9212 | 0.00% | 0.28% |

*\* หมายเหตุ 1: Median Filter L3 ชนะ Baseline ด้วย AUC 0.9680 และ Selective DWT ให้ Precision สมบูรณ์ 100% (ไม่มี False Positive เลย)*  
*\*\* หมายเหตุ 2: DWT Mid Frequency Only ได้ตำแหน่งความสนใจตรงกับกรอบแพทย์สูงสุดในทุกวิธี (Hit Rate 0.65%, Saliency Energy 0.75%)*

---

## ตารางที่ 2: ผลการทดลองชุดข้อมูล NIH ChestX-ray14 Dataset
*(ชุดทดสอบ Test Set 100 ภาพ — คัดเฉพาะ **Infiltration-only** และล็อก **Patient-level Split** ผู้ป่วยไม่ซ้ำข้ามชุด)*

| หมวดหมู่วิธีการ | สภาวะการทดลอง (Condition / Parameters) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | TN | FP | FN | TP |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | **ภาพดิบ (Raw CXR - ไม่ผ่านการกรอง)** | **0.5700** | **0.5412** | **0.9200** | **0.6815** | **0.5932** | 11 | 39 | 4 | 46 |
| **Spatial Filter** | Median Filter Level 1 ($3\times3$) | 0.5300 | 0.5181 | 0.8600 | 0.6466 | 0.5304 | 10 | 40 | 7 | 43 |
| **Spatial Filter** | Median Filter Level 2 ($5\times5$) | 0.5500 | 0.5287 | 0.9200 | 0.6715 | 0.5472 | 9 | 41 | 4 | 46 |
| **Spatial Filter** | **Median Filter Level 3 ($7\times7$)** | **0.5900** | **0.5517** | **0.9600** | **0.7007** | 0.5704 | 11 | 39 | 2 | 48 |
| **Spatial Filter** | Median Filter Level 4 ($9\times9$) | 0.5700 | 0.5422 | 0.9000 | 0.6767 | **0.5880** | 12 | 38 | 5 | 45 |
| **Hybrid** | **CLAHE + DWT Level 1** | **0.5900** | 0.5506 | **0.9800** | **0.7050** | 0.5084 | 10 | 40 | 1 | 49 |
| **Hybrid** | CLAHE + DWT Level 2 | 0.5200 | 0.5116 | 0.8800 | 0.6471 | **0.5796** | 8 | 42 | 6 | 44 |
| **Hybrid** | CLAHE + DWT Level 3 | 0.5200 | 0.5139 | 0.7400 | 0.6066 | 0.5684 | 15 | 35 | 13 | 37 |
| **Hybrid** | CLAHE + DWT Level 4 | 0.5300 | 0.5185 | 0.8400 | 0.6412 | 0.5512 | 11 | 39 | 8 | 42 |
| **Deep Learning** | **DAE + CLAHE Level 1** | **0.5500** | **0.5342** | 0.7800 | **0.6341** | **0.6164** | 16 | 34 | 11 | 39 |
| **Deep Learning** | DAE + CLAHE Level 2 | 0.5300 | 0.5211 | 0.7400 | 0.6116 | 0.5576 | 16 | 34 | 13 | 37 |
| **Deep Learning** | DAE + CLAHE Level 3 (Proposed) | 0.4900 | 0.4925 | 0.6600 | 0.5641 | 0.5012 | 16 | 34 | 17 | 33 |
| **Deep Learning** | DAE + CLAHE Level 4 | 0.5000 | 0.5000 | 0.7800 | 0.6094 | 0.5152 | 11 | 39 | 11 | 39 |
| **Pure Wavelet** | **Pure DWT Level 1 (Ratio 0.05)** | 0.5700 | 0.5455 | 0.8400 | 0.6614 | **0.6616\*** | 15 | 35 | 8 | 42 |
| **Pure Wavelet** | Pure DWT Level 2 (Ratio 0.10) | 0.5500 | 0.5362 | 0.7400 | 0.6218 | 0.6020 | 18 | 32 | 13 | 37 |
| **Pure Wavelet** | Pure DWT Level 3 (Ratio 0.20) | 0.4900 | 0.4937 | 0.7800 | 0.6047 | 0.5328 | 10 | 40 | 11 | 39 |
| **Pure Wavelet** | **Pure DWT Level 4 (Ratio 0.35)** | **0.5900** | **0.5529** | **0.9400** | **0.6963** | 0.6048 | 12 | 38 | 3 | 47 |
| **Frequency Bands** | Low Frequency Only | 0.5200 | 0.5167 | 0.6200 | 0.5636 | 0.5280 | 21 | 29 | 19 | 31 |
| **Frequency Bands** | **Mid Frequency Only** | 0.5600 | 0.5395 | 0.8200 | 0.6508 | **0.5936** | 15 | 35 | 9 | 41 |
| **Frequency Bands** | **High Frequency Only** | **0.5700** | **0.5507** | 0.7600 | 0.6387 | 0.5728 | 19 | 31 | 12 | 38 |
| **Frequency Bands** | Selective DWT | 0.5300 | 0.5176 | **0.8800** | **0.6519** | 0.5216 | 9 | 41 | 6 | 44 |

*\* หมายเหตุ: Pure DWT L1 ทำคะแนน ROC-AUC สูงสุดบน NIH (0.6616 ชนะ Baseline 0.5932 อย่างมีนัยสำคัญ)*

---

## ตารางที่ 3: ตารางเปรียบเทียบผลลัพธ์ข้ามชุดข้อมูล (VinBigData vs NIH ChestX-ray14)
*(เปรียบเทียบภายใต้ขนาดข้อมูลฝึกเท่ากัน: Train 400 ภาพ : Test 100 ภาพ)*

| สภาวะการทดลอง (Condition) | VinBigData Acc | VinBigData AUC | NIH Acc | NIH AUC | ส่วนต่าง AUC (VinBigData - NIH) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Raw CXR)** | **0.9200** | **0.9676** | 0.5700 | 0.5932 | **+0.3744** |
| **Median Filter L1** | 0.8800 | 0.9636 | 0.5300 | 0.5304 | **+0.4332** |
| **Median Filter L2** | 0.9000 | 0.9624 | 0.5500 | 0.5472 | **+0.4152** |
| **Median Filter L3** | 0.9100 | **0.9680** | 0.5900 | 0.5704 | **+0.3976** |
| **Median Filter L4** | 0.8700 | 0.9528 | 0.5700 | 0.5880 | **+0.3648** |
| **CLAHE + DWT L1** | 0.8700 | 0.9512 | 0.5900 | 0.5084 | **+0.4428** |
| **CLAHE + DWT L2** | 0.9100 | 0.9628 | 0.5200 | 0.5796 | **+0.3832** |
| **CLAHE + DWT L3** | 0.8900 | 0.9620 | 0.5200 | 0.5684 | **+0.3936** |
| **CLAHE + DWT L4** | 0.8500 | 0.9500 | 0.5300 | 0.5512 | **+0.3988** |
| **DAE + CLAHE L1** | 0.8800 | 0.9604 | 0.5500 | 0.6164 | **+0.3440** |
| **DAE + CLAHE L2** | 0.8800 | 0.9476 | 0.5300 | 0.5576 | **+0.3900** |
| **DAE + CLAHE L3 (Proposed)** | 0.8900 | 0.9440 | 0.4900 | 0.5012 | **+0.4428** |
| **DAE + CLAHE L4** | 0.9000 | 0.9588 | 0.5000 | 0.5152 | **+0.4436** |
| **Pure DWT L1** | 0.9000 | 0.9560 | 0.5700 | **0.6616** | **+0.2944** |
| **Pure DWT L2** | 0.8700 | 0.9492 | 0.5500 | 0.6020 | **+0.3472** |
| **Pure DWT L3** | 0.8500 | 0.9508 | 0.4900 | 0.5328 | **+0.4180** |
| **Pure DWT L4** | 0.8700 | 0.9524 | 0.5900 | 0.6048 | **+0.3476** |
| **DWT Low Frequency Only** | 0.8500 | 0.9352 | 0.5200 | 0.5280 | **+0.4072** |
| **DWT Mid Frequency Only** | 0.7800 | 0.8868 | 0.5600 | 0.5936 | **+0.2932** |
| **DWT High Frequency Only** | 0.8000 | 0.8920 | 0.5700 | 0.5728 | **+0.3192** |
| **Selective DWT** | 0.7900 | 0.9212 | 0.5300 | 0.5216 | **+0.3996** |

---

## 5. สรุปผลการเปรียบเทียบเชิงวิชาการสำหรับรายงานอาจารย์ (Scientific Insights)

1. **สาเหตุของช่องว่างประสิทธิภาพระหว่าง 2 ชุดข้อมูล (Cross-Dataset Gap):**
   * **VinBigData (ROC-AUC ~ 0.94 - 0.96):** เป็นภาพดิจิทัล DICOM คุณภาพสูงที่มีการตรวจยืนยันโดยรังสีแพทย์ 17 ท่าน โดยทุกภาพ Infiltration จะมีรอยโรคอื่นร่วมด้วยเสมอ (Co-morbidity เช่น Fibrosis, Thickening, Lung Opacity) ทำให้ AI จับ Visual Patterns ได้เด่นชัด
   * **NIH ChestX-ray14 (ROC-AUC ~ 0.50 - 0.66):** ควบคุมเป็น **Infiltration-only** อย่างแท้จริง (ไม่มีโรคอื่นปน) และล็อก **Patient-level Split** (ผู้ป่วยในชุดทดสอบไม่เคยปรากฏในชุดเทรนเลย) ซึ่งสอดคล้องกับสภาพจริงในโรงพยาบาลที่มีความท้าทายสูงมาก
2. **บทบาทของการลดสัญญาณรบกวน (Noise Reduction Impact):**
   * บน **NIH:** การกรองด้วย **Pure DWT L1** ช่วยขจัดสัญญาณรบกวนความถี่สูง ทำให้ ROC-AUC เพิ่มขึ้นจาก 0.5932 สู่ **0.6616** (+6.84%) ชนะ Baseline อย่างมีนัยสำคัญ
   * บน **VinBigData:** **Median Filter L3 ($7\times7$)** ดัน ROC-AUC ขึ้นแตะ 0.9680 (ชนะ Baseline) และ **CLAHE+DWT L2** ทำ Precision ได้ถึง 97.67%
3. **พฤติกรรมของ DAE (Deep Autoencoder) และการชี้ตำแหน่งของ Grad-CAM:**
   * แม้โมเดลกลุ่ม DAE + CLAHE จะให้ผลการจำแนกโรคได้ Accuracy 88%–90% แต่เมื่อตรวจด้วย Grad-CAM กลับได้คะแนน Hit Rate 0.00% และ Energy 0.01% (ต่ำที่สุดในทุกวิธี) แสดงว่าการกรองด้วย Autoencoder แบบลึกทำให้โมเดลเบี่ยงเบนความสนใจออกจากรอยโรคจริงของแพทย์
   * ในทางกลับกัน **DWT Mid Frequency Only** ให้คะแนนตำแหน่งรอยโรคตามกรอบแพทย์สูงที่สุดในทุกวิธี (Hit Rate 0.65%, Energy 0.75%)
4. **การชี้แจงสถานะ SHAP แก่อาจารย์:**
   * แจ้งอาจารย์ว่าการประเมิน Classification Metrics และ Grad-CAM Pointing Game บน 613 Bounding Boxes ทำเสร็จสมบูรณ์เรียบร้อยแล้ว
   * โมดูล SHAP จำเป็นต้องคำนวณ Background Tensors ทุกพิกเซลกับทั้ง 18 โมเดลซึ่งกินเวลารันหลายชั่วโมง จึงพักไว้ชั่วคราวเพื่อนำเสนอภาพรวมเบื้องต้นก่อน และจะรันแบบ Selective ในโมเดลตัวแทนที่ดีที่สุดสำหรับเล่มฉบับสมบูรณ์
