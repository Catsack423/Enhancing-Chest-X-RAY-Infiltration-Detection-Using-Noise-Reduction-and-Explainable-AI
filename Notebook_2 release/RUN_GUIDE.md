# การรัน Notebook 2 บน Colab GPU

1. อัปโหลด **ไฟล์เดียว** `Notebook_2_colab.zip` ไปที่ **MyDrive/Notebook_2_colab.zip** (หน้าแรกของไดรฟ์ของฉัน) ไม่ต้องแตก ZIP บน Drive และไม่ต้องอัปโหลดภาพ NIH ทั้งชุด หากกำลังอัปโหลดชุดเดิมอยู่ สามารถยกเลิกการอัปโหลดนั้นได้
2. เปิด Colab → File → Upload notebook แล้วเลือก **ไฟล์ใหม่ล่าสุด** `CNN/cnn_comparison.ipynb` จากเครื่องนี้ (หรือจาก ZIP ที่แตกบนเครื่อง) ไม่ใช้ notebook รุ่นเก่าที่ตั้งเส้นทางข้อมูลเอง
3. เลือก **Runtime → Change runtime type → GPU** รันเซลล์แรกและอนุญาตเชื่อม Drive เซลล์นี้แตกโค้ดลง `MyDrive/Notebook_2/` และแตกข้อมูลที่เลือกแล้วลง `/content/nih_cxr_subset/` โดยอัตโนมัติ อ่านภาพจากดิสก์ของ runtime; ครั้งแรกจะดาวน์โหลดน้ำหนัก ImageNet ของ ResNet50
4. เมื่อ CNN ครบทั้งสิบเงื่อนไข เปิด `Grad-CAM/gradcam_comparison.ipynb` และ `SHAP/shap_comparison.ipynb` รุ่นใหม่ตามลำดับ ใช้ `OUTPUT_ROOT`, `SMOKE` ตรงกันทั้งหมด

หาก ZIP อยู่บน MyDrive แล้ว แต่ error ยังอ้าง `nih -xray/notebooks/Notebook_2/shared/experiment.py` หรือเส้นทาง `nih-chest-xrays/data/versions/3` แสดงว่าเซลล์ตั้งค่าหรือโมดูลที่โหลดอยู่ยังเป็นรุ่นเดิม ให้คัดลอกโค้ดใน `colab_recover_setup.py` ลงเซลล์ใหม่และรันแทนเซลล์แรกเก่า โค้ดนี้ล้างเฉพาะโมดูล shared ของโครงการ แล้วใช้เซลล์ตั้งค่าจาก ZIP ที่อัปโหลดแล้ว จากนั้นรันข้อ 2 ของ notebook ต่อ ไม่ต้องอัปโหลด ZIP ใหญ่ซ้ำ

ชุด ZIP ใหม่นี้รวมโค้ด, notebook, คู่มือ และ `nih_cxr_subset.zip` ที่มีภาพใช้จริง **622 ภาพไม่ซ้ำ**: train pool 320, validation 80, balanced official test subset 100 และ XAI 123 (มี XAI 1 ภาพซ้ำกับ test subset) รวมข้อมูลสำหรับ smoke และ SHAP background แล้ว ขนาดประมาณ **67 MB** ใช้ metadata และ official lists เดิมครบ ไม่ต้องอัปโหลดหรือเตรียมภาพ test ทั้ง 12,081 ภาพ

DAE และ CNN ฝึก **epoch ละ 320 ภาพครบทั้งหมด** (คลาสละ 160) เท่าจำนวนฝึกของรุ่นแรก ใช้ IDs เดียวกันทุกเงื่อนไขและสับลำดับด้วย seed 42 และหมายเลข epoch บันทึก IDs ใน `shared/artifacts/epoch_manifests/` และจำนวนภาพ/ID hash ใน history และ checkpoint โหมด smoke ใช้ 4 ภาพต่อ epoch

ผล classification รอบนี้เป็นการประเมินบน **test subset 100 ภาพ** คลาสละ 50 ที่มาจาก official test; ไม่ใช่ official test เต็ม การลดและปรับสมดุล test ทำให้ผลต่างจากการประเมินบน test เต็ม ส่วน XAI ยังคง 123 ภาพเดิมครบทุกเงื่อนไข

ภาพ XAI 123 ภาพเก็บไฟล์ต้นฉบับเต็มขนาดเพื่อรักษา bbox และ native heatmap ส่วนภาพ classification ที่เหลือเก็บเป็น PNG lossless หลัง resize grayscale 256×256 แบบ INTER_AREA ตามขั้นเตรียมข้อมูลเดิม ก่อน denoising ผู้สร้างแพ็กตรวจทุกภาพว่า working pixels หลังโหลดกลับตรงกับการอ่านจากต้นฉบับทุกค่า จึงได้ input เดียวกันและอัปโหลดน้อยลง

บน Colab ภาพหลัง denoising เก็บ cache ใน `/content/nih_cxr_processed/` เพื่ออ่านจากดิสก์ runtime ระหว่างแต่ละ epoch ไม่อ่าน PNG จาก Drive ซ้ำ เมื่อเปิด runtime ใหม่จะเตรียม cache ขนาดเล็กนี้อีกครั้ง ส่วน checkpoint, history และผลประเมินเก็บบน Drive เหมือนเดิม

บนเครื่องที่มี NIH เต็มอยู่แล้ว รัน `python Notebook_2/pack_colab_data.py` เพื่อสร้างแพ็กใหม่หากเปลี่ยน config ไม่รวม checkpoint, ผลทดลอง หรือ dependencies Windows ลงในชุดอัปโหลด

ผลและ checkpoint เขียนลง `OUTPUT_ROOT` บน Drive โดยตรง ค่าเริ่มต้น `OUTPUT_ROOT=None` ใช้ `MyDrive/Notebook_2/runs/pilot320_val80_test100/` ตาม `shared/config.json` เพื่อแยกจากงาน test เต็มเดิม ส่วนข้อมูลที่แตกใน runtime สร้างกลับจาก ZIP ได้เมื่อเปิด runtime ใหม่ ตรวจ hash ข้อมูลทุกไฟล์เมื่อแตกและเมื่อใช้ cache ซ้ำ ตรวจ config, manifest hashes, package versions, condition ID และ DAE/CNN checkpoint ที่เกี่ยวข้องก่อนนำผลเดิมกลับมาใช้

ควรเริ่มด้วย `SMOKE=True` ในทั้งสาม notebook เพื่อทดสอบการเชื่อมต่อและ pipeline ผลจะอยู่ใต้ `smoke_runs/` แยกจากงานเต็ม โหมดนี้ใช้ train/validation คลาสละ 2 ภาพ, test คลาสละ 2 ภาพ, XAI 2 ภาพ, 1 epoch, SHAP background คลาสละ 1 และ nsamples=4 ผล smoke ไม่ใช้สรุปงานวิจัย

เมื่องานหลักพร้อม ให้เปลี่ยน `SMOKE=False` แล้วรันทั้งสาม notebook ใหม่ ค่าเริ่มต้นเป็น train 160/คลาส (รวม 320 ภาพ) ใช้ครบทุก epoch, validation 40/คลาส (รวม 80 ภาพ), test subset 50/คลาส (รวม 100 ภาพ) และ XAI 123 ภาพต่อเงื่อนไข CNN เตรียมเฉพาะ train/validation/test รวม 500 ภาพต่อวิธี; XAI เตรียมภายหลังเมื่อรันเกณฑ์นั้น หาก Colab ตัดการเชื่อมต่อ เปิด notebook เดิมและรันใหม่ จะเริ่มจาก epoch/ภาพที่บันทึกสำเร็จแล้ว

สำหรับผู้ที่มี ZIP รุ่นก่อนหน้าอยู่บน Drive ให้ **แทนที่ ZIP เดิมด้วยรุ่นใหม่ขนาดประมาณ 67 MB** และเปิด notebook รุ่นใหม่ ไม่ต้องลบ checkpoint เดิม ผลรอบใหม่แยกโฟลเดอร์ไว้แล้ว

ถ้าค่าตั้งหรือแพ็กเกจเปลี่ยน ให้คืนเวอร์ชันเดิมจาก `shared/artifacts/experiment.json` หรือเลือก `OUTPUT_ROOT` ใหม่ เพื่อไม่ผสมผลจากต่างการทดลอง

## ผลลัพธ์

เส้นทางด้านล่างอยู่ภายใน `OUTPUT_ROOT` ซึ่งค่าเริ่มต้นคือ `Notebook_2/runs/pilot320_val80_test100/`

- `CNN/results/<condition>/best.pt`: checkpoint ที่ validation loss ต่ำสุด พร้อม config, manifest hashes และ package versions
- `CNN/results/<condition>/last.pt`: น้ำหนักและ optimizer สำหรับรันต่อจาก epoch ล่าสุด
- `CNN/results/<condition>/test_predictions.csv`, `metrics.json`, `history.csv`, `confusion_matrix.png`: ผลจำแนกจริง
- `shared/artifacts/dae_checkpoint/`: DAE ที่ฝึกจาก train subset เท่านั้น; L1–L3 ใช้ checkpoint เดียวกัน
- `shared/artifacts/manifests/`: train, validation, test, XAI และ SHAP background IDs ที่ใช้ร่วมกัน
- `shared/artifacts/epoch_manifests/`: รายชื่อ 320 ภาพที่ฝึกแต่ละ epoch ร่วมกันทั้ง DAE และทุก CNN condition
- `Grad-CAM/results/<condition>/heatmaps/`: heatmap float32 ที่ขนาดต้นฉบับและช่วง [0,1]
- `SHAP/results/<condition>/heatmaps/`: NPZ เก็บ `channel_attributions` แบบ signed ที่ input resolution, `signed_input`, `signed_native` และ `positive_heatmap` ช่วง [0,1]
- `overlays/`: ภาพขนาดต้นฉบับพร้อม bbox; SHAP แสดงบวกแดง/ลบน้ำเงินโดยใช้สเกลเดียวกัน
- `per_image_metrics.csv`, `summary.csv`: ผลรายภาพ และค่าเฉลี่ย/95% CI แยก risk strata
- `comparison/metrics_all_conditions.csv`: ตารางเปรียบเทียบทุกเงื่อนไขในแต่ละเกณฑ์
- `README.md`: สร้างใหม่จากผลจริง พร้อม Confusion Matrix สิบเงื่อนไขและสถานะความครบถ้วน

ภาพต้นฉบับย่อเป็น grayscale 256×256 แบบไม่ crop ก่อน denoising แล้วเปลี่ยนเป็น RGB 224×224 และ normalize ด้วย ImageNet mean/std เหมือนกันทุกเงื่อนไข Heatmap ขยายกลับขนาดต้นฉบับด้วย bilinear ก่อนเทียบ bbox ที่พิกัดต้นฉบับ IoU ใช้ threshold 0.3 เป็นหลักและ 0.5 เป็นเสริม Pointing Game ใช้จุดสูงสุดใน bbox แบบ strict; แผนที่ศูนย์ทั้งหมดเป็น Miss และ IoU=0

ใช้ Infiltration logit เป็นเป้าหมาย XAI เสมอ รวมกรณี CNN ทำนายผิด SHAP ใช้ Expected Gradients ผ่าน GradientExplainer จริง และใช้เฉพาะส่วนบวกของ signed channel sum ที่ขยายแล้วสำหรับ localization

Bootstrap สุ่มคนไข้ 115 คนแบบคืนตัวอย่าง แล้วนำภาพทั้งหมดของคนไข้ที่สุ่มได้มาคำนวณค่าเฉลี่ย ทำ 2,000 รอบ ใช้ seed และลำดับคนไข้เดียวกันทุกเงื่อนไข แยกรายงาน Overall, Pure (19 ภาพ), Mixed-high-risk (64 ภาพ), Mixed-low-risk (40 ภาพ) ตัวเลขกลุ่มย่อยเป็น exploratory โดยเฉพาะ Pure ที่มีจำนวนน้อย

CLAHE+DWT ใช้ db1, DWT→CLAHE และ joint presets: (clip, depth, threshold multiplier) = (2,1,0.5)/(4,2,1)/(8,3,2) จึงไม่สรุปผลจาก threshold เพียงตัวเดียว DAE+CLAHE ใช้ clip 2/4/8 เปลี่ยน contrast หลัง DAE ซึ่งไม่ใช่การเปลี่ยนความแรงของ DAE

## การตรวจบนเครื่องนี้

รัน `python -m unittest discover -s Notebook_2/tests -v` จากรากโครงการเพื่อตรวจ data leakage, bbox alignment, empty maps, signed SHAP และ checkpoint compatibility รัน `python Notebook_2/verify_pipeline.py` สำหรับ smoke pipeline ครบสามเกณฑ์บนข้อมูลจริง ผลแยกไว้ใน `smoke_runs/` และบันทึกข้อจำกัดของการทดสอบไว้ตามจริง

รัน `python Notebook_2/verify_data_bundle.py` เพื่อตรวจ ZIP ที่ส่งจริง จำลองเซลล์แตกโค้ดบน Colab ตรวจ hashes ของข้อมูลที่แตก เปรียบเทียบ manifests ทั้ง full/smoke กับ NIH ต้นทาง และยืนยันว่า XAI 123 ไฟล์ตรงต้นฉบับทุก byte การตรวจนี้ไม่ใช่การรัน GPU บน Colab

แพ็กเกจหลัก: PyTorch + torchvision ที่จับคู่เวอร์ชันกัน, SHAP, NumPy, pandas, OpenCV, PyWavelets, scikit-learn, matplotlib และ nbformat สำหรับสร้าง notebook ใหม่ โค้ดวิจัยใช้ค่าตั้งเดียวกันทุกเงื่อนไข และบันทึกเวอร์ชันที่ใช้จริงในแต่ละการทดลอง
