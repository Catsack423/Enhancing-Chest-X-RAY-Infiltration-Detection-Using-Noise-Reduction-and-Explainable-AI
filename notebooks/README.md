# แผนผังและคู่มือโครงสร้างโฟลเดอร์การวิจัย (Research Notebooks Architecture Guide)
**Project Title:** *Enhancing Chest X-RAY Infiltration Detection Using Noise Reduction and Explainable AI*  
**โครงงานสัมมนาวิชาการ:** การเพิ่มประสิทธิภาพการตรวจจับรอยโรคฝ้าในปอด (Infiltration) บนภาพเอกซเรย์ทรวงอกด้วยการลดสัญญาณรบกวนและปัญญาประดิษฐ์ที่อธิบายได้

---

## 🗺️ 1. แผนผังโครงสร้างภาพรวม (Directory Tree)

```text
notebooks/
│
├── 📖 README.md                                # เอกสารสรุปภาพรวมและสถาปัตยกรรมโฟลเดอร์นี้
├── 📓 bbox_visualization.ipynb                 # การสำรวจ Bounding Box ของแพทย์รังสี 8 กลุ่มโรค
│
├── 📁 Gradcam/                                 # [ระยะที่ 1] พื้นฐาน Grad-CAM บนรอยโรค Infiltration
│   └── gradcam_normal_vs_infiltration.ipynb
│
├── 📁 Denoise_Gradcam/                         # [ระยะที่ 2] การประเมิน Denoising 7 สภาวะ (Median & DWT)
│   └── denoise_gradcam_comparison.ipynb
│
├── 📁 SHAP/                                    # [ระยะที่ 3] การอธิบายระดับพิกเซลด้วย Game Theory (SHAP)
│   └── shap_normal_vs_infiltration.ipynb
│
├── 📁 DAE_CLAHE/                               # [ระยะที่ 4] Deep Denoising Autoencoder (ตอบ Research Gap 6)
│   └── dae_clahe_pipeline.ipynb
│
├── 📁 XAI_Evaluation/                          # [ระยะที่ 5] การวัดผลเชิงสถิติ (Pointing Game, Energy, IoU 100 เคส)
│   └── xai_quantitative_metrics.ipynb
│
├── 📁 Score_CAM/                               # [ส่วนเสริม B.1] Gradient-Free CAM แก้ปัญหา Gradient Saturation
│   └── score_cam_pipeline.ipynb
│
├── 📁 Gamma_Correction/                        # [ส่วนเสริม B.2] การปรับความสว่างแบบไม่ใช่เชิงเส้น (gamma 0.5 - 1.5)
│   └── gamma_correction_pipeline.ipynb
│
├── 📁 Lung_Segmentation/                       # [ส่วนเสริม B.3] การตัดขอบเขตปอดเพื่อขจัด False Positive ภายนอก
│   └── lung_segmentation_pipeline.ipynb
│
└── 📁 DenseNet_ChexNet/                        # [ส่วนเสริม B.4] สถาปัตยกรรม ChexNet SOTA (DenseNet121 vs ResNet50)
    └── densenet_vs_resnet_pipeline.ipynb
```

---

## 📚 2. คำอธิบายรายละเอียดของแต่ละโฟลเดอร์ (Detailed Module Breakdown)

### 1. `Gradcam/` — พื้นฐาน Grad-CAM และการตีกรอบ Ground Truth
* **โน้ตบุ๊กหลัก:** [gradcam_normal_vs_infiltration.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/Gradcam/gradcam_normal_vs_infiltration.ipynb)
* **วัตถุประสงค์:** 
  * สร้างชุดข้อมูลทดสอบมาตรฐาน [sample_manifest_200.csv](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/Gradcam/sample_manifest_200.csv) ประกอบด้วยกลุ่ม **Normal (100 เคส จากผู้ป่วยไม่ซ้ำกัน)** และกลุ่ม **Infiltration (100 เคส ที่มี Bounding Box จากรังสีแพทย์สถาบัน NIH)**
  * สกัด Activation Map ของโมเดล ResNet50 (`layer4`) เทียบกับ Bounding Box เพื่อดูว่าภาพดิบเดิมโมเดลจับรอยโรคได้ตรงจุดหรือไม่

---

### 2. `Denoise_Gradcam/` — การเปรียบเทียบการลดสัญญาณรบกวน 7 สภาวะ
* **โน้ตบุ๊กหลัก:** [denoise_gradcam_comparison.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/Denoise_Gradcam/denoise_gradcam_comparison.ipynb)
* **เทคนิคที่ทดสอบ:**
  1. Baseline (Raw Image)
  2. Median Filter Level 1 (3x3), Level 2 (5x5), Level 3 (7x7)
  3. CLAHE + DWT (Discrete Wavelet Transform: db1) Level 1, Level 2, Level 3
* **จุดเด่น:** แสดงภาพเปรียบเทียบทั้ง 7 สภาวะพร้อมกัน และจัดทำ **Confusion Matrix ทั้ง 2 รูปแบบ** (Classification Matrix & XAI Localization Matrix)

---

### 3. `SHAP/` — การอธิบายผลระดับพิกเซลด้วย Game Theory
* **โน้ตบุ๊กหลัก:** [shap_normal_vs_infiltration.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/SHAP/shap_normal_vs_infiltration.ipynb)
* **วัตถุประสงค์:** 
  * ก้าวข้ามข้อจำกัดของ Grad-CAM ด้วยการใช้ **SHAP (SHapley Additive exPlanations)** เพื่อระบุว่าแต่ละพิกเซลมีอิทธิพลเชิงบวก (สนับสนุนการเป็น Infiltration) หรือเชิงลบ (สนับสนุนการเป็น Normal)
  * มีรายงานสรุป [summary_report.md](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/SHAP/reports/summary_report.md) และตารางเปรียบเทียบค่าความสำคัญของพิกเซล

---

### 4. `DAE_CLAHE/` — Deep Denoising Autoencoder ร่วมกับ CLAHE (ตอบ Research Gap 6)
* **โน้ตบุ๊กหลัก:** [dae_clahe_pipeline.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/DAE_CLAHE/dae_clahe_pipeline.ipynb)
* **อ้างอิงวิจัย:** *Thamilarasi, Asaithambi & Roselin (2025)*
* **ปัญหาที่แก้ไข (Gap 6 ในเล่ม 3 บท):** ฟิลเตอร์ดั้งเดิมอย่าง Median Filter ก่อให้เกิดปัญหา **Over-smoothing (ภาพเบลอเกินไป)** จนขอบเขตฝ้าจางๆ เลือนหาย 
* **ผลลัพธ์:** โครงข่าย Autoencoder พร้อม **Skip Connections** ช่วยดูดซับ Poisson/Gaussian Noise โดยไม่ทำลาย Texture ฝ้าในปอด มีโมเดลที่ผ่านการเทรนสมบูรณ์ที่ [checkpoints/dae_trained.pth](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/DAE_CLAHE/checkpoints/dae_trained.pth)

---

### 5. `XAI_Evaluation/` — ชุดวัดผลเชิงสถิติ (Pointing Game, Energy, IoU 100 เคส)
* **โน้ตบุ๊กหลัก:** [xai_quantitative_metrics.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/XAI_Evaluation/xai_quantitative_metrics.ipynb)
* **วัตถุประสงค์:** ป้องกันการเกิดอคติในการเลือกภาพ (Cherry-picking bias) โดยคำนวณตัวเลขสถิติบน 100 เคส Infiltration:
  1. **Pointing Game (Hit Rate %):** ตรวจว่าจุด Peak Activation อยู่ในกรอบแพทย์หรือไม่ (**DAE+CLAHE ชนะที่ 30.0%** เทียบกับ Baseline 25.0%)
  2. **Energy Inside BBox (%):** สัดส่วนความสนใจของโมเดลที่อยู่ในรอยโรคจริง (**DAE+CLAHE ชนะที่ 17.50%**)
  3. **Intersection over Union (IoU):** วัดการทับซ้อนเชิงพื้นที่ (**DAE+CLAHE ชนะที่ 0.1528**)
* **ไฟล์สำคัญ:** [reports/xai_quantitative_summary.csv](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/XAI_Evaluation/reports/xai_quantitative_summary.csv) พร้อมนำไปใส่ในเล่มสัมมนาบทที่ 4 ทันที

---

### 6. `Score_CAM/` — การอธิบายผลแบบ Gradient-Free
* **โน้ตบุ๊กหลัก:** [score_cam_pipeline.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/Score_CAM/score_cam_pipeline.ipynb)
* **อ้างอิงวิจัย:** *Rahman et al. (2021) / Wang et al. (2020)*
* **วัตถุประสงค์:** แก้ปัญหา **Gradient Saturation** และสัญญาณรบกวนของ Grad-CAM โดยนำ Activation Map มา Mask บนภาพจริงแล้วส่ง Forward Pass เพื่อวัดคะแนนความมั่นใจตรงๆ

---

### 7. `Gamma_Correction/` — การปรับความสว่างแบบไม่ใช่เชิงเส้น
* **โน้ตบุ๊กหลัก:** [gamma_correction_pipeline.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/Gamma_Correction/gamma_correction_pipeline.ipynb)
* **อ้างอิงวิจัย:** *Rahman et al. (2021)* (ระบุว่า Gamma Correction ให้ความแม่นยำสูงถึง 96.29% บน ChexNet)
* **วัตถุประสงค์:** ทดสอบค่า $\gamma \in [0.5, 0.8, 1.0, 1.2, 1.5]$ เพื่อขยายรายละเอียดเนื้อปอดส่วนที่มืดและเปรียบเทียบกับ CLAHE

---

### 8. `Lung_Segmentation/` — การตัดขอบเขตเนื้อปอด
* **โน้ตบุ๊กหลัก:** [lung_segmentation_pipeline.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/Lung_Segmentation/lung_segmentation_pipeline.ipynb)
* **อ้างอิงวิจัย:** หัวข้อ 2.2.4 ในเล่ม 3 บท
* **วัตถุประสงค์:** ตัดเฉพาะโพรงปอดซ้ายและขวา เพื่อกำจัดสัญญาณรบกวนภายนอก เช่น กระดูกไหปลาร้า, หัวไหล่, ลมในกระเพาะ และสายยางแพทย์ บังคับให้โมเดลโฟกัสเฉพาะในเนื้อปอด 100%

---

### 9. `DenseNet_ChexNet/` — สถาปัตยกรรม ChexNet SOTA
* **โน้ตบุ๊กหลัก:** [densenet_vs_resnet_pipeline.ipynb](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3/notebooks/DenseNet_ChexNet/densenet_vs_resnet_pipeline.ipynb)
* **อ้างอิงวิจัย:** *Rajpurkar et al. (2017) / Parra-Cabrera et al. (2026)*
* **วัตถุประสงค์:** ประชันสถาปัตยกรรมระหว่าง **ResNet50** (Additive Residual) กับ **DenseNet121** (Dense Concatenation) พิสูจน์ว่าโครงสร้าง Dense Connectivity สามารถดึงฟีเจอร์ฝ้า Infiltration จากชั้นแรกๆ ข้ามไปยังชั้นลึกสุดได้มีประสิทธิภาพกว่า

---

## 🧭 3. ลำดับการเปิดอ่านและนำเสนอที่แนะนำ (Recommended Reading Sequence)

หากต้องการเปิดอ่านหรือนำเสนอให้อาจารย์ที่ปรึกษาตรวจ แนะนำให้เรียงลำดับตามขั้นตอนดังนี้:
1. 🥇 **จุดเริ่มต้น:** `Gradcam/gradcam_normal_vs_infiltration.ipynb` (เข้าใจข้อมูลและ Ground Truth)
2. 🥈 **การทดลองลด Noise ดั้งเดิม:** `Denoise_Gradcam/denoise_gradcam_comparison.ipynb`
3. 🥉 **การทดลอง Deep Denoising (จุดเด่นงานวิจัย):** `DAE_CLAHE/dae_clahe_pipeline.ipynb`
4. 🏅 **บทพิสูจน์ทางสถิติ (ใส่เล่มบทที่ 4):** `XAI_Evaluation/xai_quantitative_metrics.ipynb`
5. ⭐ **เทคนิคขั้นสูงต่อยอด:** `Score_CAM/` $\rightarrow$ `Lung_Segmentation/` $\rightarrow$ `DenseNet_ChexNet/`

---

## ⚙️ 4. สัญญาข้อตกลงและสภาพแวดล้อม (Contract & Environment)
* **สัญญาภาพ:** ภาพประมวลผลทุกภาพถูกรับประกันให้อยู่ในรูปแบบ 2D NumPy Array, Data Type `uint8`, ค่าความสว่างอยู่ในช่วง `[0, 255]` เสมอ
* **โมเดลที่รองรับ:** PyTorch (รองรับทั้ง CPU และ CUDA GPU)
* **Dataset:** อิงตามสถาปัตยกรรมและดัชนีของ [NIH ChestX-ray14](file:///d:/ForSeminarProject/datasets/nih-chest-xrays/data/versions/3)
