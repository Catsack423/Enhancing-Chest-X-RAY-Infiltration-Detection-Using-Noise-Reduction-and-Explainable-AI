# แผน Notebook v3 release — สำหรับตรวจสเปก

สถานะ: **ผู้ใช้ให้สร้างแล้ว — implementation อยู่ใน `Notebook_3 release/` บน branch `codex/notebook-v3-release`**
Frozen manifests และ source-only ZIP สร้างแล้ว ผลทดสอบอ้างอิง `Notebook_3 release/VALIDATION.md`; ยังไม่รันการทดลองเต็มเพื่อรายงานคะแนนวิจัย

ฐานโค้ด: `Notebook_3 beta class 200/` ร่วมกับฟังก์ชันจาก branch `codex/notebook-v2-release`, `codex/notebook-v3-beta`, `codex/notebook-v3-beta-e7`

ข้อกำหนดงานวิจัยยังอ้างอิง `guideline.md` เป็นหลัก แผนนี้ยังไม่ได้แก้ข้อกำหนดนั้น
ผลตรวจข้อมูลจริงอยู่ใน [dataset audit](plans/v3_release_dataset_audit.json)

**ปรับตามรูป Google Drive:** ใช้ dataset ที่มีอยู่ใน `MyDrive/nih -xray/data/3` เป็นแหล่งข้อมูลของ Colab โดยไม่สร้าง/อัปโหลด dataset ZIP ใหม่
ชื่อโฟลเดอร์ `nih -xray` มีช่องว่างก่อน `-` ตามรูป; ค่า path เก็บใน config แก้ได้ และต้องตรวจว่าเข้าถึงได้หลัง mount ก่อนเริ่มงาน
รูปยืนยันโครงสร้างโฟลเดอร์และชื่อ metadata แต่ยังไม่ได้ตรวจเนื้อหา/checksum/จำนวนภาพบน Drive จริง จำนวนข้อมูลด้านล่างเป็นผล audit ของสำเนาบนเครื่อง
ค่าตำแหน่งที่เสนออยู่ใน [Drive layout config](plans/v3_release_drive_layout.json)

## 1. ข้อมูลที่เสนอ: ใช้ทั้งหมดที่ตรงโจทย์

นับจาก metadata, official lists และไฟล์ภาพบนเครื่องจริง ตรวจ checksum ของ metadata/splits ตรงกับ class200 เดิม

| ชุด | No Finding | Infiltration-only | รวม | คนไข้ |
|---|---:|---:|---:|---:|
| Train | 40,371 | 5,737 | **46,108** | 19,260 |
| Validation | 10,129 | 1,590 | **11,719** | 4,815 |
| Classification test | 9,861 | 2,220 | **12,081** | 2,589 |

- Positive ฝึก/validation/classification test: `Finding Labels == "Infiltration"` เท่านั้น
- Negative: `Finding Labels == "No Finding"` เท่านั้น ไม่ใช้ภาพที่มีโรคอื่นเป็น negative
- ขยายภายใน patient partition เดิมของ v2/class200: seed 42, reserved validation patients 20% ของ eligible official train_val patients จำนวน 4,815 คน ไม่สุ่ม partition ใหม่
- Official `train_val_list.txt`/`test_list.txt` คงเดิม และคนไข้ train/validation/test ไม่ทับกัน
- Train เก็บ 400 ภาพเดิมครบ Validation เก็บ 80 ภาพเดิมครบ และ full test รวม 100 ภาพ pilot เดิมไว้ด้วย
- พบภาพต้นทางบนเครื่องทั้งหมด 112,120 ภาพ; ภาพของชุดที่เสนอและ XAI มีครบ ฝั่ง Drive ต้องตรวจตาม manifests อีกครั้ง ไม่เลือกข้อมูลโดยอิงความพร้อมของไฟล์หรือคะแนนโมเดล
- Train เพิ่มจาก 400 เป็น 46,108 ภาพ หรือประมาณ **115 เท่า** ใช้ทุกภาพหนึ่งครั้งต่อ epoch ไม่ downsample Normal
- สัดส่วน Train Normal:Infiltration = 7.037:1 ใช้ weighted BCE (`pos_weight = 40371 / 5737`) ที่คำนวณจาก train เท่านั้น ไม่ oversample พร้อมกับถ่วงน้ำหนักซ้ำ
- Validation/test ใช้สัดส่วนจริง ไม่ปรับให้ 50:50 ตัวเลข Accuracy เทียบกับ balanced pilot เดิมโดยตรงไม่ได้
- บน full test การตอบ Normal ทุกภาพจะได้ Accuracy ประมาณ 81.62%; ต้องอ่าน Balanced Accuracy, Recall, F1, ROC-AUC และ PR-AUC ประกอบ
- Test ชุดนี้ขยาย official filtered test เดิม ไม่ใช่ external dataset ใหม่ที่ไม่เคยใช้เลย; pilot 100 ภาพและ XAI เดิมเคยใช้ประเมินมาก่อน

### XAI: คง 123 ภาพเดิม

| กลุ่ม | ภาพ |
|---|---:|
| Overall | **123** / 115 คนไข้ |
| Pure | 19 |
| Mixed high risk: มี Effusion หรือ Atelectasis | 64 |
| Mixed low risk: โรคร่วมอื่น | 40 |

ทั้งหมดอยู่ใน official test; pure 19 ภาพอยู่ใน classification test ด้วย จึงไม่ใช่สองชุดอิสระต่อกัน
Mixed 104 ภาพใช้เฉพาะ XAI ไม่เพิ่มเข้าการฝึก/classification test ที่กำหนด pure-only
เก็บ bbox, patient IDs และ risk flags เดิมทุกค่า ประเมินทุกภาพรวมทั้งภาพที่ทำนายผิดและ heatmap ว่าง

SHAP ใช้ background 32 ภาพเดิม (16 ต่อคลาส เป็น subset ของ train) เพื่อให้ reference distribution เดียวกันทุก method/seed
บันทึกว่า background เป็น balanced reference ไม่ใช่สัดส่วนประชากรจริง

## 2. รายการ method หลัก: 12 conditions

เสนอให้มี 2 config profiles: `release_controlled` สำหรับผลวิจัยหลัก และ `legacy_v2` สำหรับเก็บชุดพารามิเตอร์เดิม
ค่า L1–L3 ของ release ด้านล่างเป็น **ข้อเสนอใหม่** ไม่ใช่ค่า L1–L3 เดิมทุกตัว

| # | Condition ID ที่เสนอ | Method | พารามิเตอร์ release |
|---|---|---|---|
| 1 | `baseline` | Raw baseline | ไม่มี denoise, 1 view |
| 2 | `median_L1` | Median อ่อน | kernel 3×3 |
| 3 | `median_L2` | Median กลาง | kernel 5×5 |
| 4 | `median_L3` | Median แรง | kernel 7×7 |
| 5 | `clahe_dwt_strength_L1` | DWT → CLAHE อ่อน | threshold multiplier 0.5 |
| 6 | `clahe_dwt_strength_L2` | DWT → CLAHE กลาง | threshold multiplier 1.0 |
| 7 | `clahe_dwt_strength_L3` | DWT → CLAHE แรง | threshold multiplier 2.0 |
| 8 | `dae_clahe_strength_L1` | DAE → CLAHE อ่อน | blend α=0.25 |
| 9 | `dae_clahe_strength_L2` | DAE → CLAHE กลาง | blend α=0.50 |
| 10 | `dae_clahe_strength_L3` | DAE → CLAHE เต็ม | blend α=1.00 |
| 11 | `dwt_L1_L4_fusion` | DWT frequency feature fusion | A4 + D1 + D2 + D3 + D4, 5 views |
| 12 | `dft_low_mid_high_fusion` | DFT frequency feature fusion | low/mid/high, 3 views |

### นิยาม strength และสิ่งที่ตรึง

- กลุ่ม DWT → CLAHE: `db1`, depth=3, `periodization`, soft threshold, CLAHE clip=2.0/grid=8×8 คงที่ เปลี่ยนเฉพาะ threshold multiplier
- Threshold ต่อ detail subband = multiplier × σ̂ × sqrt(2 ln N); σ̂ ประเมินจาก finest diagonal details ด้วย MAD/0.6745, N เป็นจำนวน coefficient ใน subband ไม่ threshold approximation coefficients
- กลุ่ม DAE → CLAHE: ใช้ DAE checkpoint เดียวกันภายใน seed; `z=(1−α)x+α DAE(x)` ก่อน CLAHE clip=2.0/grid=8×8 คงที่ การ blend เป็นตัวควบคุมที่เสนอ ไม่รับประกันว่า texture loss จะเพิ่มเป็นเส้นตรง
- ใช้ชื่อและลำดับ `DWT_then_CLAHE` ให้ตรง implementation เดิมของ v2 แม้ชื่อกลุ่มเรียก CLAHE+DWT
- DWT fusion: db1/depth4/periodization, reconstruct A4 และ D1–D4 แยกเป็นภาพ spatial ไม่ใช้ CLAHE และไม่ตัด coefficients
- DFT fusion: inverse FFT ของ radial low <0.125, mid 0.125–<0.25, high ≥0.25 cycles/pixel; masks ครบและไม่ซ้อนกัน
- Frequency fusion สืบทอด signed-band encoding เดิมและเฉลี่ย feature 2,048 ค่าจาก shared backbone ก่อน head ไม่ใช่เฉลี่ย probabilities
- DWT/DFT fusion เป็นการทดลองเพิ่มเติมด้าน frequency representation แยกรายงานจาก denoising strength sweep
- เก็บ intermediate images/bands และตัวชี้วัดความต่างจาก raw เช่น edge/gradient energy สำหรับอ่านคู่กับ XAI ไม่ใช้ตัวชี้วัดนี้เป็นหลักฐานยืนยันสาเหตุ over-smoothing เพียงอย่างเดียว

### Compatibility ของ v2 release

เก็บ `legacy_v2` preset ครบ 10 denoising conditions เดิม:

- Median: kernels 3/5/7
- DWT → CLAHE: (clip, depth, threshold multiplier) = (2,1,0.5), (4,2,1), (8,3,2), wavelet db1
- DAE → CLAHE: clip 2/4/8 ด้วย DAE output เต็ม
- Baseline raw

เหตุผลที่เสนอ release preset ใหม่: v2 DWT เปลี่ยน clip/depth/threshold พร้อมกัน และ v2 DAE เปลี่ยนเฉพาะ CLAHE clip จึงยังตีความเป็นผลของ denoising strength ล้วนไม่ได้
Legacy preset ต้องใช้ output/config ID แยก ไม่ปนคะแนนกับ release controlled และยังไม่รวมการฝึก legacy เพิ่มในงบ 36 runs ด้านล่าง

## 3. โมเดลและ training protocol ที่เสนอ

สืบทอด backbone ResNet50 ImageNet V2 และ feature averaging/head จาก beta แล้วเพิ่มการ fine-tune layer4 เป็น protocol หลักของ release ตาม guideline ที่ freeze early layers และ fine-tune late layers
การ freeze ทั้ง backbone กับการ fine-tune เป็นคนละ protocol; เก็บ `head_only_legacy` เป็นโหมดรอง ไม่ปนผลสองโหมดในตารางหลัก

| รายการ | ค่าเสนอ |
|---|---|
| Working / CNN input | grayscale 256×256 / 224×224, RGB replicate + ImageNet normalization เดิม |
| Epochs | 20 ทุก condition |
| Epoch 1–3 | freeze backbone ทั้งหมด ฝึก head |
| Epoch 4–20 | เปิด layer4 + head, freeze conv1/layer1–3 |
| BatchNorm | freeze running statistics ทุก layer รวม layer4 |
| Head | mean view features → Dropout(0.3) → Linear(2048,1) |
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | head 1e-4; layer4 1e-5 เมื่อเปิด fine-tuning |
| Loss | weighted BCE, train-only pos_weight≈7.037 |
| Effective batch | 16 source images; microbatch 4 × accumulation 4, AMP |
| Seeds | training 42,43,44; dataset partition/manifests ตรึงเดิม seed42 |
| Augmentation | ยังไม่เพิ่มในชุดหลักเพื่อไม่เพิ่มตัวแปร |
| Checkpoint หลัก | validation balanced BCE ต่ำสุดจาก epochs 1–20 |
| Classification threshold | 0.5 คงที่; รายงาน threshold-independent AUC ด้วย |

ฝึกครบ 20 epochs เพื่อมี budget และ trajectory เท่ากันทุก method เลือก `best.pt` ด้วย validation เท่านั้น ไม่ใช้ last epoch อัตโนมัติ ไม่เลือก epoch/method/threshold จาก test หรือ bbox test
เก็บ `best.pt`, `last.pt` และ snapshots epoch 2/7/10 สำหรับรายงาน trajectory; งดเก็บทุก epoch เป็นค่าเริ่มต้นเพื่อลดขนาด output
Snapshot 2 เทียบ 7 มีการเปลี่ยน training stage ด้วย จึงไม่ใช่ผลของจำนวน epochs เพียงอย่างเดียว

`train_eval_loss` และ `validation_loss` ในกราฟ gap ใช้ balanced BCE นิยามเดียวกัน: 0.5×mean negative BCE + 0.5×mean positive BCE ใน eval mode
แยก optimization loss ที่มี dropout/pos_weight ไว้อีกคอลัมน์ เพื่อไม่เปรียบเทียบ loss คนละสเกล
Weighted training ทำให้ sigmoid scores ไม่ควรถูกตีความเป็น calibrated population probabilities โดยอัตโนมัติ

ทุก method ใช้รายชื่อ train/val/test เดียวกัน, order/initialization seed ที่จับคู่กัน, backbone/head/schedule/loss เดียวกัน
จำนวน source images ต่อ effective batch เท่ากัน; DWT มี 5 views และ DFT มี 3 views จึงใช้ compute มากกว่าแม้จำนวนพารามิเตอร์เท่ากัน
พบ RTX4060 8GB บนเครื่อง; ต้อง smoke-test microbatch 4 กับทั้ง 5-view fusion และ XAI ก่อนล็อก config หากลด batch ต้องกำหนดเหมือนกันทุก condition และรักษา effective batch

### DAE

- Train DAE ใหม่จาก Train 46,108 ภาพเท่านั้น; validation reconstruction ใช้ validation partition เดิม
- base_channels=32, residual architecture เดิม; เสนอ 20 epochs, batch16, Adam lr1e-3, reconstruction loss MSE + 0.2 L1
- สืบทอด synthetic corruption Gaussian sigma0.05/Poisson scale30 จาก source; ภาพ NIH เดิมเป็น reconstruction target proxy ไม่ใช่ภาพ clean ground truth จริง
- ไม่ฉีด synthetic noise เพิ่มในการประเมิน classifier/test หลัก
- DAE หนึ่งตัวต่อ training seed แชร์ทั้ง α=0.25/0.5/1 ของ seed นั้น; เลือกจาก validation reconstruction loss ไม่ใช้ test/XAI
- เก็บ reconstruction error แยก class เพื่อดูว่าการฝึกที่มี Normal มากกว่ากระทบภาพ Infiltration หรือไม่

งบหลัก: **12 methods × 3 seeds = 36 CNN runs + 3 DAE runs** ไม่รวม optional legacy/head-only ablations
ทำ pipeline smoke ก่อน แล้ว baseline full-data seed42 เพื่อวัดเวลา/RAM/VRAM จากนั้นรันครบทุก method ของ seed42 ก่อน seeds43/44
Baseline รอบนี้เป็นรอบหนึ่งใน 36 runs ไม่ต้องฝึกซ้ำ รอตรวจแผนก่อนทุกขั้นที่สร้าง release/เทรน
ยังไม่มี benchmark เต็ม จึงไม่กำหนดจำนวนชั่วโมงจากเวลาของ pilot 400 ภาพ

## 4. Evaluation และ reports

### Classification

- full official filtered test 12,081 ภาพ: Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC, PR-AUC
- confusion matrix แบบ counts และ row percentages; ROC/PR curves และ test predictions รายภาพ
- ใช้ mean±SD ของคะแนน 3 seeds แยกจาก patient-bootstrap 95% CI จำนวน 2,000 draws ไม่ตีความ 3 seeds เป็นคนไข้เพิ่ม
- paired bootstrap เทียบ method กับ baseline บน patient IDs เดียวกันและ seed เดียวกัน; รายงานทิศทางและ CI ไม่อ้าง significance จาก delta อย่างเดียว
- ระบุ positive prevalence และ PR-AUC ของการทำนายคงที่เพื่ออ่าน class imbalance

### Grad-CAM และ SHAP

- ทุก method × seed ใช้ classifier checkpoint เดียวกับ classification บน bbox 123 ภาพ
- Grad-CAM target layer4; รวม contribution ของทุก view ก่อน normalization
- SHAP Expected Gradients: backgrounds 32 เดิม, nsamples200, batch≤8 ตาม smoke; เก็บ signed attribution แยกจาก positive localization map
- IoU ที่ normalized heatmap thresholds0.3/0.5, Pointing Game, energy inside, empty-heatmap rate
- แปลง heatmap กลับ native image coordinates ก่อนวัดกับ bbox ไม่วัด bbox ในพิกัดย่อโดยไม่แปลง
- รายงาน overall/pure/mixed_high_risk/mixed_low_risk พร้อม n_images/n_patients และ patient-bootstrap95%CI
- Frequency SHAP อธิบาย processed spatial bands ไม่ใช่ raw-input attribution ผ่าน NumPy FFT/DWT; ใช้ attribution domain เดิมทุก condition ที่เปรียบเทียบและบอกข้อจำกัด
- รวม 12×3×123 = 4,428 source-image explanations ต่อ explainer หรือ 8,856 fused Grad-CAM/SHAP maps โดยไม่เพิ่ม XAI ไปทุก test image

### Analysis ที่รวมมาจากทุกรุ่น

- ตารางเปรียบเทียบ classification + XAI + compute time/peak VRAM ทุก method และ seed
- กราฟ train_eval/validation พร้อม best epoch, overfit summary และเปิดย้อนหลังบน CPU ได้จาก histories
- กราฟ strength เทียบ Balanced Accuracy/F1 และ IoU/Pointing Game; แยก frequency fusion เป็นอีกกลุ่ม
- per-image paired deltas, risk-stratum reports, representative overlays และ signed frequency bands
- epoch snapshots และรายงานเทียบเวอร์ชันเดิมเป็น descriptive comparison; จำนวนข้อมูล/model protocol เปลี่ยน จึงไม่อ้างว่าเป็นผลของเพิ่มข้อมูลอย่างเดียว
- เป้าหมายเพิ่ม Accuracy 10 จุดเปอร์เซ็นต์จาก pilot ไม่เป็นเงื่อนไขเลือกโมเดล/ข้อมูล; baselineใหม่สูงกว่า90% จะเพิ่ม10จุดไม่ได้ ต้องรายงานค่าจริงและ uncertainty
- สิ่งที่ไม่ได้เพิ่มจาก v1: DenseNet, Score-CAM, gamma correction และ lung segmentation เพราะอยู่นอกช่วง v2 release→v3 beta ที่ร้องขอ และเพิ่ม backbone/ตัวแปรใหม่

## 5. โครงสร้าง release ที่เสนอ

สร้าง branch `codex/notebook-v3-release` จาก class200 ใช้ shared modules ชุดเดียว รายละเอียดโครงสร้างที่สร้างและวิธีใช้ดู `Notebook_3 release/README.md`
โครงสร้างด้านล่างเป็นสเปกที่ใช้สร้าง release

```text
Notebook_3 release/
├── README.md / RUN_GUIDE.md / VALIDATION.md
├── prepare_data.py / build_notebooks.py  # เตรียม manifests/ตรวจ Drive; ไม่ pack ภาพ
├── run_all.ipynb
├── configs/
│   ├── release_full.json
│   ├── drive_paths.json
│   ├── methods_release_controlled.json
│   └── methods_legacy_v2.json
├── manifests/
│   ├── train.csv / validation.csv / test.csv
│   ├── xai.csv / background.csv
│   └── source_hashes.json / dataset_audit.json
├── DAE/train_dae.ipynb
├── CNN/train_all_methods.ipynb
├── Frequency/inspect_bands.ipynb
├── Grad-CAM/evaluate_gradcam.ipynb
├── SHAP/evaluate_shap.ipynb
├── Analysis/compare_all_methods.ipynb
├── check_overfit.ipynb / check_overfit.py
├── export_results_colab.ipynb / export_results_colab.py
├── shared/
│   ├── config.py / experiment.py / data.py / drive_data.py / runtime_cache.py
│   ├── denoising.py / dae_model.py / frequency.py
│   ├── models.py / training.py / explainers.py / xai.py
│   └── evaluation.py / reporting.py / epoch_comparison.py
├── tests/
└── runs/release_full/seed_42/      # local หรือ Drive output; อีกชุด seed_43 และ seed_44
    ├── DAE/checkpoints/
    ├── CNN/results/<condition>/
    ├── Grad-CAM/results/<condition>/
    ├── SHAP/results/<condition>/
    ├── comparison/
    └── shared/artifacts/          # run lock, hashes, versions, epoch IDs
```

### 5.1 ตำแหน่งบน Google Drive / Colab

```text
MyDrive/
└── nih -xray/
    ├── data/3/                         # dataset ต้นทางเดิม อ่านอย่างเดียว
    │   ├── images_001/ ... images_012/
    │   ├── Data_Entry_2017.csv
    │   ├── BBox_List_2017.csv
    │   ├── train_val_list.txt
    │   └── test_list.txt
    └── Notebook_3_release/             # source/config/manifests และผลแยกจาก dataset
        └── runs/release_full/
            ├── seed_42/
            ├── seed_43/
            └── seed_44/

/content/cxr_v3_cache/<data_and_preprocessing_hash>/  # cache ชั่วคราวบน Colab VM
```

```python
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
DATA_ROOT = Path("/content/drive/MyDrive/nih -xray/data/3")
RELEASE_ROOT = Path("/content/drive/MyDrive/nih -xray/Notebook_3_release")
OUTPUT_ROOT = RELEASE_ROOT / "runs/release_full"
CACHE_ROOT = Path("/content/cxr_v3_cache")
```

Code ZIP (หากใช้) มีเฉพาะ source/notebooks/config/manifests ไม่ใส่ภาพหรือ weights
ค่าเดิม `/content/drive/MyDrive/nih-chest-xrays/data/versions/3` และการ extract `nih_cxr_subset.zip` ไม่ใช่เส้นทางของ release นี้
Local mode ยังอ่าน `D:/kaggle_cache/datasets/nih-chest-xrays/data/versions/3` ได้ผ่าน config แยก

### 5.2 การอ่านภาพจำนวนมาก

1. Mount Drive แล้วตรวจ `DATA_ROOT` และ metadata/split ทั้งสี่ไฟล์; checksum ต้องตรง audit เดิมก่อนยืนยันจำนวน Train/Val/Test
2. ใช้ relative paths ใน manifests ไม่ glob Drive ทั้ง dataset ทุก epoch รองรับทั้ง `images_001/images/<filename>` และ `images_001/<filename>` เพราะรูปยังไม่แสดงชั้นใน ต้อง resolve/ล็อก layout ก่อนรัน
3. ตรวจรายชื่อที่เลือกทั้งหมดจาก Drive ระหว่างเตรียม cache; ภาพหาย/อ่านไม่ได้ให้หยุดและแจ้งรายชื่อ ไม่ข้ามภาพหรือเลือกภาพอื่นแทน
4. อ่านภาพที่ใช้จาก Drive ครั้งเดียวต่อการเตรียม cache แล้วสร้าง working grayscale256×256 ด้วย resize เดิม เก็บ pixel arrays แบบ uint8 เป็น shards/memmap บน `/content` ไม่สร้างสำเนา PNG หมื่นไฟล์บน Drive
5. Working cache สำหรับ binary69,908ภาพใช้ pixel payload ประมาณ4.27GiB ก่อน metadata/header; XAI123ภาพเก็บ native pixels แยกเพื่อประเมิน bbox ไม่คัดลอก raw dataset112,120ภาพทั้งหมดมา Colab
6. Train/Val/Test และ DAE อ่าน working cache บน VM; Grad-CAM/SHAP ใช้ input pipeline เดียวกันและ native XAI cache DWT/DFT views สร้างจาก working pixels ไม่เก็บทุก band ของทุกภาพถาวร
7. Cache ของ preprocessing แต่ละ method จำกัดพื้นที่และใช้ hash ของ source/resize/method/DAEcheckpoint ตรวจ disk ก่อนสร้าง แต่ไม่ลบภาพต้นทาง Drive
8. Source metadata/manifests, pixel integrity และ shard completion ต้องถูกตรวจ; shard ที่เขียนไม่จบไม่ถูกใช้ รันต่อใน VM เดิมใช้ shard ที่ตรวจผ่านแล้วได้

Google Colab แนะนำลด Drive I/O เพราะการเข้าถึงไฟล์เล็กจำนวนมากมี latency/operation quota; แผนนี้ใช้ไฟล์เดิมตาม path ที่ผู้ใช้ให้แล้ว cache บน VM เพื่อไม่อ่าน Drive ซ้ำทุก epoch
การเตรียม cache ครั้งแรกยังต้องอ่านภาพเดิมทีละไฟล์ และไม่ได้รับประกันว่าจะไม่ติด quota ต้องวัด I/O จริงและรองรับ retry ที่มีขีดจำกัด ไม่วน retry ไม่สิ้นสุด ([Colab FAQ](https://research.google.com/colaboratory/faq.html))

### 5.3 Resume และขนาดผลลัพธ์

Runtime cache บน `/content` อาจหายเมื่อเปลี่ยน/รีเซ็ต VM; cache ไม่ใช่ที่เก็บถาวรของงานวิจัย
เก็บ checkpoints/history/completion records/manifests/provenance บน `OUTPUT_ROOT` ใน Drive เพื่อ resume การฝึกหรือ XAI ได้ แม้ต้องสร้าง input cache ใหม่
บันทึกผลทีละ condition/seed และ stage output เป็นชุดบน VM ก่อน copy ไป Drive เพื่อลดการเขียนไฟล์เล็ก; completion marker เขียนหลังตรวจไฟล์ปลายทางครบเท่านั้น
ไม่แก้ ย้าย หรือลบภาพ/metadata ใน `data/3` และไม่สร้าง dataset ZIP ใหม่
ข้อมูล, model weights, caches, run outputs และ ZIP ไม่เข้า Git; code/config/manifests/audit เข้า branch ได้
Output บันทึกและ resume ราย condition/seed/ภาพ พร้อม config/source/data/version hashes; ปฏิเสธ resume ที่ provenance เปลี่ยน
จัด signed SHAP/band arrays เป็นไฟล์บีบอัดแยกจาก PNG overlays และ export ต่อ seed เพื่อคุมขนาดไฟล์

## 6. จุดที่ให้ตรวจ

- [x] เปลี่ยนแหล่งข้อมูล Colab เป็น `MyDrive/nih -xray/data/3` ตามรูป; ใช้ dataset ที่มีอยู่ ไม่สร้าง/อัปโหลด dataset ZIP ใหม่
- [ ] ตรวจ metadata/checksum และภาพตาม manifests บน Drive จริงหลัง mount; cache ที่ `/content` และผลถาวรแยกอยู่บน Drive
- [ ] ใช้ Train46,108/Val11,719/fullTest12,081 ตาม partition เดิม แทนเพิ่มเป็นจำนวนกลม ๆ ต่อคลาส
- [ ] ใช้ทุก Normal และ weighted BCE แทน downsampling/oversampling
- [ ] วิธีหลัก12conditions ครอบคลุม v2 denoising + v3 frequency fusion
- [ ] ใช้ controlled strength preset ใหม่ และเก็บค่าดั้งเดิมเป็น legacy config แยก
- [ ] fine-tune layer4 หลัง head warm-up3epochs, รวม20epochs, 3seeds เป็น protocol หลัก
- [ ] Grad-CAM/SHAP ครบ123bboxทุกmethod/seed พร้อมCI และเก็บ check_overfit/resume/export/epoch-analysis

## 7. แหล่งอ้างอิงเชิงเทคนิค

- สองแนวทาง transfer learning (fixed extractor กับ fine-tuning) อธิบายใน [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html); warm-up/learning rates/epoch budget ในแผนนี้เป็นค่าที่เสนอสำหรับโปรเจกต์ ไม่ใช่ค่าที่เอกสารรับรองว่าเหมาะที่สุด
- Soft threshold ลดขนาด coefficients เข้าหาศูนย์ตาม [PyWavelets thresholding](https://pywavelets.readthedocs.io/en/stable/ref/thresholding-functions.html); depth/multipliers/CLAHEคงที่เป็นการออกแบบควบคุมตัวแปรในแผนนี้
- Background references สำหรับ Expected Gradients อธิบายใน [SHAP GradientExplainer](https://shap.readthedocs.io/en/stable/generated/shap.GradientExplainer.html); จำนวน32/nsamples200สืบทอดจากโปรเจกต์
- จำนวนภาพและคนไข้ทั้งหมดมาจาก NIH metadata/splits บนเครื่องและ [audit JSON](plans/v3_release_dataset_audit.json) ไม่ได้คาดจากชื่อเวอร์ชัน
- การลด small-file Drive I/O และข้อจำกัดของ mounted Drive อ้างอิง [Google Colab FAQ](https://research.google.com/colaboratory/faq.html); ใช้ VM cache เป็นการปรับให้เข้ากับ dataset ที่มีอยู่ตาม path ที่ผู้ใช้กำหนด
