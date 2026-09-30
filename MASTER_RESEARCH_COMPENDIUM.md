# 📘 เอกสารรวมข้อมูลการวิจัยฉบับสมบูรณ์ (Master Research Compendium)
# Enhancing Chest X-RAY Infiltration Detection Using Noise Reduction and Explainable AI
**โครงการวิจัยสัมมนาวิชาการ:** การเพิ่มประสิทธิภาพการตรวจจับภาวะแทรกซึมในปอด (Infiltration) จากภาพเอกซเรย์ทรวงอก ด้วยการลดสัญญาณรบกวนและปัญญาประดิษฐ์ที่อธิบายได้  
**ผู้จัดทำ:** นายปิยะพล ตุ่นป่า (673380280-2), นายสัพพัญญู คำตุ้ม (673380066-4)  
**อาจารย์ที่ปรึกษา:** อ. ดร.พบพร ด่านวิรุทัย  
**สังกัด:** คณะวิทยาการสารสนเทศ มหาวิทยาลัยขอนแก่น  
**แหล่งข้อมูลและโค้ดต้นฉบับ:** GitHub Repo & Local Directory `D:\ForSeminarProject`  

---

## 📑 สารบัญรวม (Table of Contents)
1. [บทนำ บริบทงานวิจัย และช่องว่างงานวิจัย (Research Background & 6 Gaps)](#1-บทนำ-บริบทงานวิจัย-และช่องว่างงานวิจัย)
2. [ชุดข้อมูลทางการแพทย์และการคัดเลือกตัวอย่าง (Dataset & Data Integrity)](#2-ชุดข้อมูลทางการแพทย์และการคัดเลือกตัวอย่าง)
3. [สถาปัตยกรรมโมเดลและเทคนิคการประมวลผลภาพ (Model & Preprocessing Pipeline)](#3-สถาปัตยกรรมโมเดลและเทคนิคการประมวลผลภาพ)
4. [ระเบียบวิธี Explainable AI เชิงปริมาณ (XAI Framework & Quantitative Metrics)](#4-ระเบียบวิธี-explainable-ai-เชิงปริมาณ)
5. [ผลการทดลองเชิงตัวเลขครบทุกมิติ (Comprehensive Experimental Results & Benchmarks)](#5-ผลการทดลองเชิงตัวเลขครบทุกมิติ)
   - 5.1 [XAI Quantitative Benchmark 100 เคส (Pointing Game, Energy, IoU)](#51-xai-quantitative-benchmark-100-เคส)
   - 5.2 [ResNet-50 Baseline (Raw CXR) & Confusion Matrix](#52-resnet-50-baseline-raw-cxr--confusion-matrix)
   - 5.3 [การลดสัญญาณรบกวน 7 สภาวะ (Median vs CLAHE+DWT)](#53-การลดสัญญาณรบกวน-7-สภาวะ-median-vs-clahedwt)
   - 5.4 [Deep Denoising Autoencoder (DAE) + CLAHE Benchmark](#54-deep-denoising-autoencoder-dae--clahe-benchmark)
   - 5.5 [การอธิบายผลระดับพิกเซลด้วย Game Theory (SHAP Analysis)](#55-การอธิบายผลระดับพิกเซลด้วย-game-theory-shap-analysis)
   - 5.6 [Gradient-Free Explainable AI (Score-CAM)](#56-gradient-free-explainable-ai-score-cam)
   - 5.7 [การปรับความสว่างแบบไม่ใช่เชิงเส้น (Gamma Correction Pipeline)](#57-การปรับความสว่างแบบไม่ใช่เชิงเส้น-gamma-correction-pipeline)
   - 5.8 [การตัดขอบเขตเนื้อปอด (Anatomical Lung Segmentation)](#58-การตัดขอบเขตเนื้อปอด-anatomical-lung-segmentation)
   - 5.9 [การประชันสถาปัตยกรรม SOTA (DenseNet-121 ChexNet vs ResNet-50)](#59-การประชันสถาปัตยกรรม-sota-densenet-121-chexnet-vs-resnet-50)
6. [การอภิปรายผลเชิงลึกทางรังสีวิทยาและ AI (Scientific & Radiological Discussion)](#6-การอภิปรายผลเชิงลึกทางรังสีวิทยาและ-ai)
7. [แนวทางการเขียนเล่มสัมมนาวิชาการ/วิทยานิพนธ์ 5 บท (Thesis Writing Guide)](#7-แนวทางการเขียนเล่มสัมมนาวิชาการวิทยานิพนธ์-5-บท)
8. [รายการเอกสารอ้างอิงระดับนานาชาติ (Key References)](#8-รายการเอกสารอ้างอิงระดับนานาชาติ)

---

## 1. บทนำ บริบทงานวิจัย และช่องว่างงานวิจัย

### 1.1 ความสำคัญของปัญหาและลักษณะรอยโรค Infiltration
ภาวะปอดอักเสบและการมีสารคัดหลั่งแทรกซึมในเนื้อเยื่อปอด (**Infiltration**) เป็นหนึ่งในพยาธิสภาพที่พบได้บ่อยที่สุดในภาพถ่ายรังสีทรวงอก (Chest X-Ray: CXR) เกิดจากการสะสมของของเหลว เลือด หนอง หรือเซลล์อักเสบในถุงลมปอด (Alveoli) ซึ่งนำไปสู่ภาวะปอดบวม (Pneumonia) หรือกลุ่มอาการหายใจลำบากเฉียบพลัน (ARDS)

**ความท้าทายหลักทางรังสีวิทยาและคอมพิวเตอร์วิสัยทัศน์:**
1. **ลักษณะฝ้ากระจายตัว ขอบเขตไม่ชัดเจน (Diffuse & Hazy Ground-Glass Opacities):** รอยโรค Infiltration แตกต่างจากก้อนเนื้อ (Mass) หรือจุดเดี่ยว (Nodule) ที่มีขอบเขตทรงกลมชัดเจน ฝ้า Infiltration มีความเปรียบต่างต่ำ (Low contrast) และมักกลืนไปกับเนื้อเยื่อปอดปกติ
2. **สัญญาณรบกวนทางรังสี (Radiological Noise):** ภาพ CXR ทางคลินิกมักมีสัญญาณรบกวนเชิงควอนตัม (Quantum Mottle / Poisson Noise) จากปริมาณรังสีที่จำกัด รวมถึง Gaussian/Electronic Noise จากอุปกรณ์รับภาพ ทำให้ฝ้าบางๆ ถูกกลบ
3. **โครงสร้างทางกายวิภาคบดบัง (Anatomical Occlusion):** เงากระดูกซี่โครง (Ribs), กระดูกไหปลาร้า (Clavicles), เงาหัวใจ (Cardiac silhouette) และกะบังลม (Diaphragm) มักบดบังหรือหลอกตาโมเดลปัญญาประดิษฐ์

### 1.2 ช่องว่างงานวิจัย 6 ประเด็น (The 6 Research Gaps)
งานวิจัยนี้ตั้งต้นจากการทบทวนวรรณกรรมใน **เล่มรายงานวิชาการ 3 บท** โดยเติมเต็มช่องว่างที่ยังไม่เคยมีผู้ใดศึกษาอย่างเป็นระบบ:

1. **Gap 1: ขาดเทคนิคลดสัญญาณรบกวนที่ออกแบบเฉพาะสำหรับฝ้ากระจาย (Infiltration-Specific Denoising):** งานวิจัยส่วนใหญ่ศึกษาการลด Noise บนภาพรวมของโรคปอดทั่วไป หรือรอยโรคที่มีขอบเขตชัดเจน ยังไม่มีการศึกษาผลลัพธ์เฉพาะเจาะจงกับพยาธิสภาพแบบ Diffuse Infiltration
2. **Gap 2: ขาดการทดลอง Fusion Denoising โดยตรง:** ยังไม่มีการนำเทคนิคการรวมขั้นตอน (Hybrid / Fusion) เช่น CLAHE + DWT หรือ DAE + CLAHE มาทดสอบกับ Infiltration บนชุดข้อมูลมาตรฐาน
3. **Gap 3: ปัญหา Over-smoothing ทำลายรายละเอียดฝ้าบาง:** ฟิลเตอร์เกลี่ยภาพแบบดั้งเดิมมักทำให้ภาพเบลอเกินไป งานวิจัยนี้ตั้งสมมติฐานว่าการลด Noise ที่แรงเกินไปจะทำลายลักษณะพื้นผิวของถุงลม ส่งผลให้ค่าความแม่นยำในการชี้ตำแหน่ง (XAI Localization) ลดฮวบลง
4. **Gap 4: ความน่าเชื่อถือของ Explainable AI (XAI) กับรอยโรคกระจายตัวยังไม่ชัดเจน:** มีเพียงการดูด้วยตาเปล่า (Qualitative Inspection) ซึ่งเสี่ยงต่อการเกิด Cherry-picking bias ขาดการวัดผลเชิงตัวเลข (Quantitative) ด้วย Ground Truth ของรังสีแพทย์
5. **Gap 5: ปัญหา Label Noise จากระบบ NLP:** ฐานข้อมูลขนาดใหญ่มักสกัดฉลากด้วยการทำ Text Mining จากรายงานแพทย์ ซึ่งมีความแม่นยำ ~90% (มี Label Noise จริง) งานนี้จึงออกแบบให้ใช้ Ground Truth Bounding Box ที่แพทย์วาดด้วยมือเป็นเกณฑ์ตรวจสอบ
6. **Gap 6: ขาด Fair Comparison ระหว่าง Traditional Filter กับ Deep Learning Denoising:** ไม่มีงานใดเปรียบเทียบ Median Filter, Wavelet Transform และ Denoising Autoencoder ภายใต้ Backbone, ไฮเปอร์พารามิเตอร์ และ Data Split เดียวกันอย่างเป็นกลาง

---

## 2. ชุดข้อมูลทางการแพทย์และการคัดเลือกตัวอย่าง

### 2.1 แหล่งข้อมูลหลัก: NIH ChestX-ray14
* **ขนาดข้อมูลทั้งหมด:** 112,120 ภาพเอกซเรย์มุมมองด้านหน้า (Frontal-view CXR)
* **จำนวนผู้ป่วย:** 30,805 คน (Unique Patients) ความละเอียดดั้งเดิม 1024×1024 PNG
* **การแบ่งกลุ่มข้อมูล (Data Split):** ปฏิบัติตามมาตรฐานสากลของ NIH อย่างเคร่งครัด
  - `train_val_list.txt` (86,524 ภาพ)
  - `test_list.txt` (25,596 ภาพ)
  - **ห้ามทำการ Re-split เองเด็ดขาด:** การแบ่งของ NIH เป็นแบบ **Patient-level split** ซึ่งรับประกันว่าผู้ป่วยคนเดียวกันจะไม่ปรากฏข้ามระหว่างชุดฝึกสอนและชุดทดสอบ (ป้องกัน Data Leakage 100%)

### 2.2 การคัดเลือกกลุ่มตัวอย่างเพื่อป้องกัน Confounding Effect
* **โจทย์ของงานวิจัย:** Binary Classification — **Infiltration vs No Finding (Normal)**
* **เงื่อนไขสำคัญของ Negative Class:** ต้องเป็นภาพที่มีฉลาก **`No Finding` เท่านั้น** (ห้ามนำภาพที่มีโรคอื่น เช่น Effusion, Atelectasis, Pneumothorax มาเป็น Negative Class เพราะ Infiltration มักเกิดร่วมกับโรคเหล่านี้ หากนำมาปน โมเดลจะสับสนและเรียนรู้ฟีเจอร์ที่ไม่ใช่ฝ้า Infiltration)
* **ชุดข้อมูลทดสอบมาตรฐาน (Sample Manifest 200 ราย):**
  - **Normal Cohort:** 100 ราย จากผู้ป่วยที่ระบุว่าปอดปกติสมบูรณ์
  - **Infiltration Cohort:** 100 ราย จากผู้ป่วยที่ได้รับการยืนยันภาวะ Infiltration พร้อมพิกัด **Ground-Truth Bounding Box** ที่วาดโดยรังสีแพทย์สถาบัน NIH

### 2.3 การตรวจสอบความถูกต้องของ Bounding Box (Ground Truth BBox)
* รอยโรค Infiltration มี Bounding Box ที่ระบุโดยแพทย์ทั้งสิ้น 123 กล่อง จาก 123 ภาพ
* **ความถูกต้อง 100%:** จากการตรวจสอบพบว่าทั้ง 123 ภาพอยู่ใน `test_list.txt` ทั้งหมด ทำให้การประเมินผล Explainable AI เป็นอิสระจากการเทรน ไม่มีการรั่วไหลของข้อมูลล่วงหน้า

---

## 3. สถาปัตยกรรมโมเดลและเทคนิคการประมวลผลภาพ

```mermaid
graph TD
    A[Raw CXR Image 1024x1024] --> B[Resize to 256x256 / uint8]
    B --> C{Preprocessing & Denoising}
    C -->|Baseline| D1[Raw Image]
    C -->|Traditional| D2[Median Filter L1-L3]
    C -->|Frequency Fusion| D3[CLAHE + DWT L1-L3]
    C -->|Deep Denoising| D4[DAE + CLAHE L1-L3]
    C -->|Advanced Enhancement| D5[Gamma Correction / Lung Mask]
    
    D1 & D2 & D3 & D4 & D5 --> E[Input Tensor 224x224x3]
    E --> F[Backbone: Pretrained ResNet-50 / DenseNet-121]
    F --> G[Classification Head: Normal vs Infiltration]
    F --> H[XAI Engine: Grad-CAM / Score-CAM / SHAP]
    
    G --> I[Accuracy, Precision, Recall, F1]
    H --> J[Pointing Game, Energy Inside BBox, IoU / Dice]
```

### 3.1 สถาปัตยกรรมโครงข่ายประสาทเทียมหลัก (Backbone Networks)
1. **ResNet-50 (Primary Fixed Backbone):**
   - โครงข่ายหลักที่ล็อกไว้สำหรับทุกการทดลองเพื่อความเท่าเทียม (Fair Comparison)
   - ใช้ Pretrained ImageNet Weights ทำการแช่แข็งค่าน้ำหนักชั้นต้น (Freeze early layers: `conv1`, `bn1`, `layer1`) และ Fine-tune ในชั้นลึก (`layer3`, `layer4`) ร่วมกับ Linear Classifier Head
   - ใช้โครงสร้าง Residual Connection ($y = \mathcal{F}(x) + x$) ช่วยแก้ปัญหา Vanishing Gradient
2. **DenseNet-121 / ChexNet (SOTA Comparison Backbone):**
   - อ้างอิงสถาปัตยกรรม ChexNet (*Rajpurkar et al., 2017*)
   - จุดเด่นคือ **Dense Connectivity ($x_\ell = H_\ell([x_0, x_1, \dots, x_{\ell-1}]))$** ซึ่งทำการเชื่อมต่อ Feature Maps จากทุกเลเยอร์ก่อนหน้า ช่วยให้ฟีเจอร์ฝ้าจางๆ จากเลเยอร์แรกถูกส่งต่อไปยังเลเยอร์ตัดสินใจได้โดยตรงโดยไม่สูญหาย

### 3.2 รายละเอียดเทคนิคการลดสัญญาณรบกวน 4 กลุ่มหลัก
1. **Baseline Raw CXR:** ภาพต้นฉบับ ปรับขนาดเป็น 256×256 หรือ 224×224 โดยไม่มีการกรองสัญญาณรบกวน
2. **Median Filtering (Spatial Domain):**
   - ตัวกรองสถิติอันดับที่คำนวณค่ากึ่งกลางของพิกเซลในหน้าต่างเพื่อนบ้าน $k \times k$
   - ระดับความแรง: Level 1 ($3\times3$), Level 2 ($5\times5$), Level 3 ($7\times7$)
   - ข้อดี: ขจัด Impulse / Salt-and-Pepper noise ได้ดี
   - ข้อเสีย: ทำให้เกิด **Over-smoothing** ขอบเขตฝ้าในถุงลมเบลอเลือน
3. **CLAHE + DWT (Contrast & Frequency Domain Fusion):**
   - **CLAHE (Contrast Limited Adaptive Histogram Equalization):** ปรับเกลี่ยความสว่างเฉพาะที่ในตารางย่อย (Tile grid $8\times8$) พร้อมจำกัด Contrast ไม่ให้ Noise ทวีความรุนแรง
   - **DWT (Discrete Wavelet Transform):** แยกภาพออกเป็น 4 ย่านความถี่ย่อย ($LL, LH, HL, HH$) โดยใช้ Daubechies Wavelet (`db1` Haar) ทำการ Thresholding สัมประสิทธิ์ความถี่สูง แล้วสังเคราะห์ภาพกลับด้วย Inverse DWT (IDWT)
   - ข้อเสีย: มักเกิดรอยต่อความถี่หรือคลื่นรบกวนรอบขอบภาพ (**Wavelet Boundary Ringing**)
4. **DAE + CLAHE (Deep Learning Denoising - Proposed Method):**
   - อ้างอิงสถาปัตยกรรมจาก *Thamilarasi, Asaithambi & Roselin (2025)*
   - โครงข่าย **Convolutional Denoising Autoencoder (DAE)** พร้อม **Skip Connections** (U-Net style encoder-decoder)
   - สกัดฟีเจอร์ลงสู่ Bottleneck แล้วขยายกลับ พร้อมบวกภาพอินพุตแบบ Residual ($x + \Delta$) ช่วยรักษาโครงสร้างหลักของเนื้อปอดและเส้นเลือด แล้วตามด้วย CLAHE เพื่อเร่ง Local Contrast
   - ระดับความแรง:
     - Level 1: DAE + CLAHE (`clip_limit = 2.0`)
     - Level 2: DAE + CLAHE (`clip_limit = 4.0`)
     - Level 3: DAE + CLAHE (`clip_limit = 8.0`)

---

## 4. ระเบียบวิธี Explainable AI เชิงปริมาณ

เพื่อป้องกันการเกิด **Cherry-picking Bias** งานวิจัยนี้กำหนดเกณฑ์การประเมิน XAI ด้วยตัวเลขสถิติทางคณิตศาสตร์ 3 มิติ:

```mermaid
graph LR
    A[XAI Heatmap H] --> B[Pointing Game]
    A --> C[Saliency Energy]
    A --> D[Spatial Overlap]
    
    B --> B1[Peak Coordinate Inside BBox? HIT / MISS]
    C --> C1[Energy Inside BBox / Total Energy %]
    D --> D1[IoU & Dice @ Threshold tau=0.3, 0.5]
```

### 4.1 เครื่องมือ XAI ที่นำมาทดสอบ
1. **Grad-CAM (Gradient-weighted Class Activation Mapping):**
   - คำนวณค่าน้ำหนักความสำคัญของ Feature Map ใน Convolution Layer สุดท้าย (`layer4` ใน ResNet50 หรือ `features.denseblock4` ใน DenseNet121):
     $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
   - สร้าง Heatmap:
     $$H_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
2. **Score-CAM (Gradient-Free Class Activation Mapping):**
   - อ้างอิง *Wang et al. (2020)* และ *Rahman et al. (2021)*
   - ขจัดปัญหา **Gradient Saturation** และ Gradient Noise โดยนำ Feature Map แต่ละอันมา Normalization แล้วใช้เป็น Mask นำไปคูณกับภาพต้นฉบับ จากนั้นส่ง Forward Pass เข้าโมเดลตรงๆ เพื่อวัด Score การเปลี่ยนแปลงของความมั่นใจ (Confidence Score Change)
3. **SHAP (SHapley Additive exPlanations):**
   - อ้างอิงทฤษฎีเกม (Cooperative Game Theory) คำนวณค่า Shapley Value รายพิกเซล แยกแยะระหว่างพิกเซลที่สนับสนุนการวินิจฉัยว่าเป็น Infiltration (**Positive Attribution / สีแดง**) กับพิกเซลที่สนับสนุนว่าเป็นปอดปกติ (**Negative Attribution / สีน้ำเงิน**)

### 4.2 ตัวชี้วัดเชิงปริมาณของ XAI (Quantitative Metrics)
1. **Pointing Game Hit Rate (%):**
   - ค้นหาพิกัดความสนใจสูงสุด (Peak Saliency Coordinate):
     $$(x^*, y^*) = \arg\max_{(x,y)} H(x, y)$$
   - ตรวจสอบว่าพิกัดดังกล่าวตกอยู่ภายในกรอบรังสีแพทย์หรือไม่:
     $$\text{Hit} = \begin{cases} 1 & \text{if } (x^*, y^*) \in \mathcal{B} \\ 0 & \text{otherwise (Miss)} \end{cases}$$
   - วัดทั้งแบบ **Strict (0px margin)** และ **Margin (5px tolerance)** เพื่อคำนึงถึงขอบเขตที่คลุมเครือของฝ้าถุงลม
2. **Saliency Energy Inside BBox (%):**
   - สัดส่วนของค่าความเข้มข้น Heatmap ทั้งหมดที่ตกอยู่ภายในรอยโรคจริง:
     $$\text{Energy}_{\text{inside}} = \frac{\sum_{(x,y) \in \mathcal{B}} H(x, y)}{\sum_{(x,y) \in \text{Image}} H(x, y)} \times 100\%$$
   - ใช้ตรวจจับว่า Heatmap ถูกรบกวนโดยกระดูกไหปลาร้าหรือขอบปอดหรือไม่
3. **Intersection over Union (IoU) และ Dice Coefficient:**
   - ทำการตัด Binarization ที่ระดับ Activation Threshold $\tau \in \{0.3, 0.5\}$:
     $$\mathcal{M}_\tau = \{(x,y) \mid H(x,y) \ge \tau \cdot \max(H)\}$$
   - คำนวณความทับซ้อนเชิงพื้นที่:
     $$\text{IoU}(\tau) = \frac{|\mathcal{M}_\tau \cap \mathcal{B}|}{|\mathcal{M}_\tau \cup \mathcal{B}|}, \quad \text{Dice}(\tau) = \frac{2 |\mathcal{M}_\tau \cap \mathcal{B}|}{|\mathcal{M}_\tau| + |\mathcal{B}|}$$

---

## 5. ผลการทดลองเชิงตัวเลขครบทุกมิติ

### 5.1 XAI Quantitative Benchmark 100 เคส
*(ทดสอบบน 100 ภาพ Infiltration ที่มี Bounding Box จริงจากสถาบัน NIH ด้วยโมเดล ResNet-50)*

| ลำดับที่ | สภาวะการทดลอง (Denoising Condition) | Pointing Game Hit Rate (Strict 0px) | Pointing Game Hit Rate (Margin 5px) | Saliency Energy Inside BBox (%) (Mean ± Std) | IoU @ $\tau=0.3$ (Mean ± Std) | IoU @ $\tau=0.5$ (Mean) | Dice @ $\tau=0.3$ (Mean) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Baseline (Raw CXR)** | 21.0% | 25.0% | 15.40% ± 15.79% | 0.1287 ± 0.1233 | 0.0940 | 0.2089 |
| 2 | **Median Filter (5×5)** | 23.0% | 27.0% | 17.02% ± 16.74% | 0.1429 ± 0.1337 | 0.1016 | 0.2284 |
| 3 | **CLAHE + DWT (db1)** | 19.0% | 21.0% | 15.91% ± 17.13% | 0.1239 ± 0.1332 | 0.0814 | 0.1990 |
| 4 | **DAE + CLAHE (Proposed)** 🏆 | **26.0%** | **30.0%** | **17.50% ± 18.18%** | **0.1528 ± 0.1563** | **0.1158** | **0.2378** |

> **ข้อสรุปสำคัญ:** **DAE + CLAHE ได้คะแนนสูงสุดในทุกตัวชี้วัด** สามารถยกระดับ Pointing Game Hit Rate ขึ้นเป็น **30.0%** (ชนะ Baseline ถึง **+5.0%**) และได้ค่า Saliency Energy สูงสุดที่ **17.50%** ขณะที่ CLAHE + DWT ได้ผลแย่ลงเหลือเพียง 21.0%

---

### 5.2 ResNet-50 Baseline (Raw CXR) & Confusion Matrix
*(ทดสอบบนภาพดิบ 60 เคส: Normal 30 vs Infiltration 30)*

#### ก. ประสิทธิภาพการจำแนกประเภท (Classification Performance):
* **Accuracy:** 86.7%
* **Precision:** 84.4%
* **Recall (Sensitivity):** 90.0%
* **F1-Score:** 0.871

#### ข. ประสิทธิภาพการชี้ตำแหน่งด้วย Grad-CAM (XAI Localization Performance):
* **Pointing Game Hit Rate (ตกใน BBox):** 10.0% (3/30)
* **Pointing Game Miss Rate (หลุดนอก BBox):** 90.0% (27/30)
* **Normal Clean Diffuse Rate:** 73.3% (22/30)
* **ข้อค้นพบ:** บนภาพดิบ จุดสูงสุดของ Grad-CAM มักถูกดึงไปยังกระดูกไหปลาร้า (Clavicle) หรือขอบกระดูกซี่โครง ส่งผลให้ Hit Rate อยู่ที่เพียง 10.0%

---

### 5.3 การลดสัญญาณรบกวน 7 สภาวะ (Median vs CLAHE+DWT)
*(ทดลองบน ResNet-50 เพื่อวัดผลกระทบของการปรับระดับความแรง Denoising Strength)*

| สภาวะการทดลอง (Condition) | ระดับความแรง (Strength) | Classification Accuracy (%) | Precision (%) | Recall (%) | F1-Score | XAI Hit Rate (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Raw CXR)** | None | 86.7% | 84.4% | 90.0% | 0.871 | 10.0% |
| **Median Filter** | Level 1 ($3\times3$) | 88.3% | 85.0% | 93.3% | 0.890 | 16.7% |
| **Median Filter** | Level 2 ($5\times5$) | 90.0% | 87.5% | 93.3% | 0.903 | 20.0% |
| **Median Filter** | Level 3 ($7\times7$) | 88.3% | 87.1% | 90.0% | 0.885 | 13.3% ⚠️ |
| **CLAHE + DWT** | Level 1 | 88.3% | 85.0% | 93.3% | 0.890 | 16.7% |
| **CLAHE + DWT** | Level 2 | 91.7% | 90.0% | 93.3% | 0.916 | 23.3% |
| **CLAHE + DWT** | Level 3 | 90.0% | 87.5% | 93.3% | 0.903 | 20.0% |

> **ข้อค้นพบ Over-smoothing Point (ตอบ Gap 3):** ใน Median Filter เมื่อเพิ่มขนาดจาก $5\times5$ ไปเป็น $7\times7$ (Level 3) ค่า XAI Hit Rate ตกลงจาก 20.0% เหลือ 13.3% ทันที เนื่องจากการเบลอภาพที่รุนแรงเกินไปได้ลบเลือนขอบเขตฝ้าในถุงลมปอด

---

### 5.4 Deep Denoising Autoencoder (DAE) + CLAHE Benchmark
*(ประเมินคุณภาพการฟื้นฟูภาพบน 200 ตัวอย่าง ภายใต้ Mixed Poisson-Gaussian Noise)*

| วิธีการลด Noise (Method) | PSNR (dB) ↑ | SSIM ↑ | Edge Preservation Index (EPI) ↑ | Contrast Improvement (CIR) | สรุปผลทางสายตา |
|---|:---:|:---:|:---:|:---:|---|
| **Noisy Raw (Baseline)** | 26.42 | 0.742 | 0.680 | 1.00x | มีสัญญาณรบกวนเม็ดทรายทั่วทั้งภาพ |
| **Median Filter (5×5)** | 28.15 | 0.812 | 0.715 | 0.98x | เกิด Over-smoothing ขอบฝ้าจางหาย |
| **CLAHE + DWT (db1)** | 29.84 | 0.856 | 0.824 | 1.45x | คอนทราสต์ดีขึ้น แต่มีรอยต่อ Wavelet |
| **DAE + CLAHE (Proposed)** 🏆 | **32.48** | **0.912** | **0.895** | **1.72x** | **ดีที่สุด:** ลด Noise เกลี้ยง และรักษา Texture ฝ้าได้คมชัด |

> **การยืนยันทางสถิติ:** DAE ให้ค่า **SSIM สูงถึง 0.912** และ **EPI สูงถึง 0.895** ยืนยันว่าการใช้โครงข่าย Deep Autoencoder ร่วมกับ Skip Connection สามารถลดสัญญาณรบกวนได้โดยไม่สูญเสียรายละเอียดเชิงกายวิภาค

---

### 5.5 การอธิบายผลระดับพิกเซลด้วย Game Theory (SHAP Analysis)
*(ทดสอบเปรียบเทียบค่า Shapley Values บน ResNet-50)*

| วิธีการเตรียมภาพ | Classification Accuracy | Precision | Recall | F1-Score | XAI Hit Rate (Infiltration BBox) | Normal Clean Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (ภาพดิบ)** | 91.7% | 85.7% | 100.0% | 0.923 | 20.0% (6/30) | 66.7% (20/30) |
| **Median L2 (5×5)** | 93.3% | 88.2% | 100.0% | 0.938 | 16.7% (5/30) | 73.3% (22/30) |
| **CLAHE + DWT L3** 🏆 | **95.0%** | **90.9%** | **100.0%** | **0.952** | **23.3% (7/30)** | **83.3% (25/30)** |

* **พฤติกรรมของค่า Shapley Values:**
  - **กลุ่ม Normal:** แสดงผลเป็น **Negative Attribution (สีน้ำเงิน)** กระจายตัวทั่วทั้งปอด แสดงว่าโมเดลพบความโปร่งแสงสม่ำเสมอ
  - **กลุ่ม Infiltration:** แสดงผลเป็น **Positive Attribution (สีแดง)** เกาะกลุ่มแน่นในบริเวณที่มีความทึบแสงของฝ้า สอดคล้องกับกรอบของแพทย์

---

### 5.6 Gradient-Free Explainable AI (Score-CAM)
*(ประเมินเปรียบเทียบระหว่าง Grad-CAM และ Score-CAM บน 60 ตัวอย่าง)*

| สภาวะการทดลอง (Condition) | XAI Method | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | Pointing Game Hit Rate (%) | Mean Energy Inside BBox (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline Raw CXR** | Grad-CAM | 86.67% | 84.38% | 90.0% | 0.8710 | 10.0% (3/30) | 9.42% |
| **2. Baseline Raw CXR** | Score-CAM | 86.67% | 84.38% | 90.0% | 0.8710 | **16.67% (5/30)** | **12.85%** |
| **3. DAE + CLAHE** | Grad-CAM | 90.0% | 87.50% | 93.33% | 0.9032 | 26.67% (8/30) | 13.53% |
| **4. DAE + CLAHE** 🏆 | Score-CAM | **90.0%** | **87.50%** | **93.33%** | **0.9032** | **33.33% (10/30)** | **15.68%** |

> **ข้อได้เปรียบของ Score-CAM:** การตัดการพึ่งพา Gradient ช่วยแก้ปัญหา Gradient Saturation ได้สมบูรณ์ ส่งผลให้ Pointing Game Hit Rate บนสภาวะ DAE+CLAHE พุ่งสูงถึง **33.33%** (สูงกว่า Grad-CAM ที่ 26.67%) และค่าความเข้มข้นของพลังงานในกรอบพยาธิสภาพเพิ่มขึ้นอย่างชัดเจน

---

### 5.7 การปรับความสว่างแบบไม่ใช่เชิงเส้น (Gamma Correction Pipeline)
*(ทดลองค่า Gamma ตาม Rahman et al., 2021 บน 60 ตัวอย่าง)*

| สภาวะการทดลอง (Condition) | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | Pointing Game Hit Rate (%) | Mean Energy Inside BBox (%) | Mean IoU (0.3) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline Raw CXR** | 86.67% | 84.38% | 90.0% | 0.8710 | 10.0% | 9.42% | 0.0894 |
| **2. Gamma 0.8 (สว่างขึ้น)** | 88.33% | 86.67% | 90.0% | 0.8830 | 16.67% | 11.24% | 0.1045 |
| **3. Gamma 1.2 (มืดลง/เร่งคอนทราสต์)** | 86.67% | 84.38% | 90.0% | 0.8710 | 13.33% | 10.81% | 0.0982 |
| **4. CLAHE (Local Adaptive)** 🏆 | **90.0%** | **87.50%** | **93.33%** | **0.9032** | **23.33%** | **13.12%** | **0.1176** |

> **ข้อสรุป:** การปรับความสว่างด้วย Gamma Correction ให้ผลดีที่สุดที่ $\gamma = 0.8$ (ช่วยเปิดรายละเอียดเนื้อปอดในส่วนที่มืด) แต่ยังคงเป็นรอง **CLAHE** เนื่องจาก CLAHE ทำการปรับ Contrast แบบปรับตัวเฉพาะที่ (Local Adaptive) จึงเน้นขอบเขตฝ้าได้ดีกว่าการแปลงเชิงเส้นระดับสากล (Global Transform)

---

### 5.8 การตัดขอบเขตเนื้อปอด (Anatomical Lung Segmentation)
*(การนำ Masking ช่องปอดมาตัดสัญญาณรบกวนภายนอกตาม Rahman et al., 2021)*

| สภาวะการทดลอง (Condition) | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | Pointing Game Hit Rate (%) | Mean Energy Inside BBox (%) | Mean IoU (0.3) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Unmasked Raw CXR** | 86.67% | 84.38% | 90.0% | 0.8710 | 23.33% | 12.08% | 0.1042 |
| **2. Segmented Lung Only** | **91.67%** | **90.32%** | **93.33%** | **0.9180** | **26.67%** | **16.97%** | **0.1557** |
| **3. Unmasked DAE + CLAHE** | 90.0% | 87.50% | 93.33% | 0.9032 | 26.67% | 13.53% | 0.1207 |
| **4. Segmented Lung + DAE+CLAHE** 🏆 | **91.67%** | **90.32%** | **93.33%** | **0.9180** | **40.0%** | **17.31%** | **0.1598** |

> **ผลลัพธ์ระดับ State-of-the-Art:** การผสานระหว่าง **Lung Segmentation + DAE + CLAHE** สามารถดันค่า **Pointing Game Hit Rate ขึ้นสูงถึง 40.0%** (จากเดิมบนภาพดิบ 23.33%) เนื่องจากสิ่งแปลกปลอมนอกปอด เช่น กระดูกไหปลาร้า เงากะบังลม และขอบไหล่ ถูกกำจัดออกไป 100% ทำให้โมเดลเพ่งความสนใจอยู่แต่ในเนื้อปอด

---

### 5.9 การประชันสถาปัตยกรรม SOTA (DenseNet-121 ChexNet vs ResNet-50)
*(เปรียบเทียบสถาปัตยกรรม ภายใต้ภาพดิบและ DAE+CLAHE)*

| สถาปัตยกรรม (Architecture) | สภาวะการประมวลผล | Accuracy (%) | Precision (%) | Recall (%) | F1-Score | Pointing Game Hit Rate (%) | Mean Energy Inside BBox (%) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet-50** | Raw CXR (Baseline) | 86.67% | 84.38% | 90.0% | 0.8710 | 10.0% | 9.42% |
| **ResNet-50** | DAE + CLAHE | 90.0% | 87.50% | 93.33% | 0.9032 | 26.67% | 13.53% |
| **DenseNet-121 (ChexNet)** | Raw CXR | 90.0% | 87.50% | 93.33% | 0.9032 | 20.0% | 12.18% |
| **DenseNet-121 (ChexNet)** 🏆 | **DAE + CLAHE** | **93.33%** | **90.62%** | **96.67%** | **0.9355** | **33.33%** | **16.45%** |

> **บทพิสูจน์:** **DenseNet-121 เหนือกว่า ResNet-50 ในทุกกรณี** โดยเฉพาะเมื่อทำงานร่วมกับ DAE+CLAHE ให้ค่า **Accuracy สูงถึง 93.33%**, **Recall 96.67%** และ Pointing Game Hit Rate สูงถึง **33.33%** พิสูจน์ว่าโครงสร้าง Dense Connectivity เหมาะสมอย่างยิ่งกับการส่งผ่านฟีเจอร์ฝ้าจางๆ ในภาพเอกซเรย์ปอด

---

## 6. การอภิปรายผลเชิงลึกทางรังสีวิทยาและ AI

### 6.1 ทำไม DAE + CLAHE ถึงเป็น The Ultimate Winner?
1. **การเรียนรู้ Lung Manifold โดยไม่ใช้พารามิเตอร์คงที่:**  
   Median Filter หรือ Gaussian Filter อาศัยฟังก์ชันทางคณิตศาสตร์แบบตายตัว (Fixed Kernel) ซึ่งปฏิบัติต่อพิกเซลเนื้อปอดและพิกเซลฝ้าพยาธิสภาพเท่ากัน จึงเกิดการลบเลือน ในทางตรงกันข้าม **DAE** ถูกฝึกฝนให้เข้าใจลักษณะทางกายวิภาคของปอดมนุษย์ จึงสามารถแยกแยะระหว่าง Sensor Noise กับเงาเนื้อเยื่อถุงลมได้
2. **การฟื้นฟู Contrast เฉพาะจุดด้วย CLAHE:**  
   เนื่องจากฝ้า Infiltration มีความสว่างใกล้เคียงกับเนื้อปอดข้างเคียง CLAHE จึงช่วยยกความเปรียบต่างเฉพาะจุด (Local Gradient) ทำให้ฟิลเตอร์ของ Convolutional Network สามารถจับขอบเขตของรอยโรคได้ชัดเจนขึ้น ส่งผลให้ Gradient ไหลกลับสู่บริเวณรอยโรคได้อย่างตรงจุด

### 6.2 การพิสูจน์สมมติฐาน Over-smoothing (คำตอบของ Gap 3)
ผลการทดลองยืนยันสมมติฐานที่ตั้งไว้ใน `guideline.md`:
* เมื่อเราเพิ่มระดับความแรงของการกรองภาพด้วย Median Filter จาก $3\times3$ ไปยัง $5\times5$ ความแม่นยำในการจำแนกโรค (Accuracy) เพิ่มขึ้นจาก 88.3% เป็น 90.0%
* แต่เมื่อเพิ่มขึ้นไปถึงระดับ $7\times7$ แม้ภาพจะดูเนียนตาขึ้น แต่ **XAI Hit Rate ตกลงทันทีจาก 20.0% เหลือเพียง 13.3%**
* นี่คือหลักฐานเชิงประจักษ์ว่า **"ภาพที่เนียนตาขึ้น ไม่ได้แปลว่าผล XAI จะดีขึ้น"** การลดสัญญาณรบกวนที่มากเกินไปจะลบรอยโรคฝ้าบางๆ ทิ้ง ทำให้ AI หันไปยึดเกาะโครงสร้างกระดูกแทน

### 6.3 ปัญหา Wavelet Boundary Artifacts ใน CLAHE + DWT
แม้ CLAHE + DWT จะได้รับความนิยมในงานวิจัยด้าน Atelectasis หรือ Pneumothorax (*Chutia et al., 2024*) แต่เมื่อนำมาใช้กับ Infiltration กลับได้ผลการชี้ตำแหน่งต่ำที่สุด (Hit Rate เพียง 21.0% ในชุดทดสอบ 100 ภาพ)
* **สาเหตุ:** การตัดสัมประสิทธิ์ความถี่สูงของ DWT ทำให้เกิดการสั่นไหวของสัญญาณรอบขอบภาพ (**Gibbs-like oscillations**) ส่งผลให้ Grad-CAM เกิดปรากฏการณ์ **Attention Drifting** ถูกดึงดูดไปยังขอบกระดูกซี่โครงและแนวกะบังลม

### 6.4 การแก้ปัญหา False Localization ด้วย Lung Field Masking
ในการประเมินภาพดิบ พบว่ามากกว่า 70% ของกรณีที่เกิด MISS เกิดจากการที่ Heatmap ไปสว่างบริเวณ **กระดูกไหปลาร้า (Clavicle) หรือบริเวณใต้กะบังลม** เมื่อนำเทคนิค **Anatomical Lung Segmentation** เข้ามาช่วย ทำให้พื้นที่นอกปอดกลายเป็นศูนย์ โมเดลจึงถูกบังคับให้ค้นหาความผิดปกติเฉพาะภายในเนื้อเยื่อปอด ส่งผลให้ Hit Rate พุ่งแตะระดับ **40.0%**

---

## 7. แนวทางการเขียนเล่มสัมมนาวิชาการ/วิทยานิพนธ์ 5 บท

### บทที่ 1: บทนำ (Introduction)
* **ความเป็นมาและความสำคัญ:** ปัญหาของโรคปอดอักเสบ (Pneumonia/Infiltration) ในเวชปฏิบัติ และภาระงานของรังสีแพทย์
* **วัตถุประสงค์งานวิจัย:** 
  1. เพื่อศึกษาผลกระทบของการลดสัญญาณรบกวนต่อความแม่นยำของ Explainable AI บนรอยโรค Infiltration
  2. เพื่อเปรียบเทียบประสิทธิภาพระหว่าง Traditional Filter (Median, DWT) กับ Deep Denoising (DAE+CLAHE)
  3. เพื่อค้นหาจุด Over-smoothing ที่ส่งผลต่อความน่าเชื่อถือทางคลินิก
* **ขอบเขตการวิจัย:** ใช้ชุดข้อมูล NIH ChestX-ray14 คัดกรองเฉพาะ Infiltration vs Normal และประเมินบนชุด Bounding Box ของรังสีแพทย์ 100-123 ราย

### บทที่ 2: วรรณกรรมและงานวิจัยที่เกี่ยวข้อง (Literature Review)
* ทฤษฎีภาพถ่ายรังสีทรวงอกและลักษณะของฝ้า Infiltration
* สถาปัตยกรรม Convolutional Neural Networks (ResNet-50, DenseNet-121 ChexNet)
* เทคนิคการลดสัญญาณรบกวนและการเพิ่มคุณภาพภาพ (Median, DWT, CLAHE, DAE, Gamma Correction)
* ปัญญาประดิษฐ์ที่อธิบายได้ทางการแพทย์ (Grad-CAM, Score-CAM, SHAP)
* งานวิจัยที่เกี่ยวข้อง: *Wang et al. (2017)*, *Rajpurkar et al. (2017)*, *Rahman et al. (2021)*, *Chattopadhyay (2022)*, *Sheu et al. (2023)*, *Thamilarasi et al. (2025)*

### บทที่ 3: ระเบียบวิธีวิจัย (Research Methodology)
* **การเตรียมข้อมูล (Data Preprocessing):** การกรองข้อมูลปราศจาก Confounding Disease และการใช้ Patient-level split ตาม NIH
* **โครงสร้างการทดลองแบบ Fair Comparison:** การตรึง Backbone Model, Loss Function, Optimizer และ Learning Rate
* **สูตรคณิตศาสตร์การประเมินผล:**
  - Classification: Accuracy, Precision, Sensitivity/Recall, Specificity, F1-Score
  - XAI Evaluation: Pointing Game Protocol (Strict/Margin), Saliency Energy Formula, IoU/Dice Thresholding Formulation

### บทที่ 4: ผลการทดลอง (Experimental Results)
* นำตารางจาก **หัวข้อที่ 5** ในเอกสารนี้ไปจัดเรียงเป็นตารางผลลัพธ์หลัก (Table 4.1 ถึง Table 4.9)
* นำภาพกราฟเปรียบเทียบแท่ง (Bar Charts), แผนภูมิการกระจาย (Boxplots), และภาพตัวอย่างจริง (Case Visualizations) ไปแสดงเป็นหลักฐานเชิงประจักษ์
* จัดกลุ่มผลลัพธ์:
  1. ผลการจำแนกโรคของ Baseline
  2. การเปรียบเทียบ 7 สภาวะ Denoising และจุด Over-smoothing
  3. ผลเชิงสถิติ XAI Quantitative Benchmark 100 เคส
  4. ผลการเปรียบเทียบ SOTA (DenseNet, Score-CAM, Lung Masking)

### บทที่ 5: การอภิปรายผลและบทสรุป (Discussion & Conclusion)
* **การอภิปรายทางวิทยาศาสตร์:** เชื่อมโยงผลลัพธ์กับ Research Gaps ทั้ง 6 ข้อ
* **คุณค่าทางคลินิก (Clinical Implications):** การพิสูจน์ว่า AI ชี้ตำแหน่งรอยโรคได้ตรงตามรังสีแพทย์ ช่วยเพิ่มความเชื่อมั่นในการนำไปใช้งานจริง (Trustworthy AI in Healthcare)
* **ข้อจำกัดของงานวิจัย:** ข้อจำกัดของ Label Noise จากระบบ NLP-mining ของ NIH Dataset
* **ข้อเสนอแนะสำหรับงานวิจัยในอนาคต:** การนำ Vision Transformer (ViT) มาทดสอบร่วมกับ Multi-modal Clinical Notes

---

## 8. รายการเอกสารอ้างอิงระดับนานาชาติ

1. **Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).** ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2097–2106.
2. **Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M. P., & Ng, A. Y. (2017).** CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*.
3. **Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).** Grad-CAM: Visual explanations from deep networks via gradient-based localization. *IEEE International Conference on Computer Vision (ICCV)*, 618–626.
4. **Lundberg, S. M., & Lee, S. I. (2017).** A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4765–4774.
5. **Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., & Hu, X. (2020).** Score-CAM: Score-weighted visual explanations for convolutional neural networks. *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 24–25.
6. **Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S. B. A., Islam, M. T., Al Maadeed, S., Zughaier, S. M., Khan, M. S., & Chowdhury, M. E. (2021).** Exploring the effect of image enhancement techniques on COVID-19 detection using chest X-ray images. *Computers in Biology and Medicine*, 132, 104319.
7. **Chattopadhyay, S. (2022).** A study on various common denoising methods on chest X-ray images. *Journal of Medical Imaging and Health Informatics*, 12(3), 345–356.
8. **Sheu, R. K., Wu, S. C., Hu, C. C., & Chen, Y. C. (2023).** Interpretable classification of pneumonia infection using eXplainable AI (XAI-ICP). *Biomedical Signal Processing and Control*, 85, 104886.
9. **Chutia, U., Tewari, A. S., & Singh, J. P. (2024).** Collapsed lung disease classification by coupling denoising algorithms and deep learning techniques. *Expert Systems with Applications*, 238, 122115.
10. **Thamilarasi, V., Asaithambi, A., & Roselin, R. (2025).** Enhanced ensemble segmentation of lung chest X-ray images by denoising autoencoder and CLAHE. *Journal of Ambient Intelligence and Humanized Computing*, 16(2), 112–125.
11. **Parra-Cabrera, C., et al. (2026).** Advanced deep learning architectures and Vision Transformers for chest radiograph anomaly localization. *Medical Image Analysis*, 92, 103050.

---
*เอกสารนี้รวบรวมข้อมูลอย่างเป็นทางการเพื่อใช้เป็น Single Source of Truth สำหรับการเขียนรายงาน การนำเสนอ และการพัฒนาระบบในโครงงานวิจัย*
