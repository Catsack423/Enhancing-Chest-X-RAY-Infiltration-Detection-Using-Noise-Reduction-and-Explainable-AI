# การรัน Notebook 2 บน Colab GPU

1. คัดลอกโฟลเดอร์ `Notebook_2` ทั้งโฟลเดอร์ไปที่ Google Drive: `MyDrive/nih-chest-xrays/notebooks/Notebook_2` ไม่อัปโหลดเฉพาะไฟล์ notebook เพราะต้องใช้โมดูล `shared/` ร่วมกัน
2. เตรียม NIH data ที่ `MyDrive/nih-chest-xrays/data/versions/3` โดยมี `Data_Entry_2017.csv`, `BBox_List_2017.csv`, `train_val_list.txt`, `test_list.txt` และภาพใน `images_001/images/` ถึง `images_012/images/` หรือกำหนด `DATA_ROOT` ให้ชี้ชุดข้อมูลเดียวกันใน Colab
3. เปิด `CNN/cnn_comparison.ipynb` ด้วย Colab และเลือก **Runtime → Change runtime type → GPU** รันเซลล์ตามลำดับ ครั้งแรกจะดาวน์โหลดน้ำหนัก ImageNet ของ ResNet50
4. เมื่อ CNN ครบทั้งสิบเงื่อนไข เปิด `Grad-CAM/gradcam_comparison.ipynb` และ `SHAP/shap_comparison.ipynb` ใช้ `NOTEBOOK_2_DIR`, `DATA_ROOT`, `OUTPUT_ROOT`, `SMOKE` ตรงกันทั้งหมด

ใช้ชุดไฟล์ `Notebook_2_colab.zip` ที่จัดไว้ได้ โดยแตกโฟลเดอร์ `Notebook_2` ลงในตำแหน่งข้อ 1 ชุดนี้มี source, notebook และคู่มือครบ แต่ไม่มีข้อมูล NIH หรือ checkpoint การทดลองขนาดเล็กจากเครื่องนี้ หากคัดลอกด้วยตนเอง ไม่ต้องนำ `.runtime/`, `smoke_runs/` หรือ artifacts ของเครื่องนี้ไปด้วย

ผลและ checkpoint เขียนลง `OUTPUT_ROOT` บน Drive โดยตรง ไม่สูญหายเมื่อ runtime ปิด ค่าเริ่มต้น `OUTPUT_ROOT=None` ใช้ `runs/train_100_per_class/` ใต้ `NOTEBOOK_2_DIR` ตาม `shared/config.json` เพื่อแยกจากผลรอบ 2,000 ภาพต่อคลาส ตรวจข้อมูลต้นทางทุกครั้งก่อนโหลด manifest และตรวจ config, manifest hashes, package versions, condition ID และ DAE/CNN checkpoint ที่เกี่ยวข้องก่อนนำผลเดิมกลับมาใช้

ควรเริ่มด้วย `SMOKE=True` ในทั้งสาม notebook เพื่อทดสอบการเชื่อมต่อและ pipeline ผลจะอยู่ใต้ `smoke_runs/` แยกจากงานเต็ม โหมดนี้ใช้ train/validation คลาสละ 2 ภาพ, test คลาสละ 2 ภาพ, XAI 2 ภาพ, 1 epoch, SHAP background คลาสละ 1 และ nsamples=4 ผล smoke ไม่ใช้สรุปงานวิจัย

เมื่องานเต็มพร้อม ให้เปลี่ยน `SMOKE=False` แล้วรันทั้งสาม notebook ใหม่ ค่าเต็มเป็น train 100/คลาส (รวม 200 ภาพ), validation 25/คลาส (รวม 50 ภาพ), test 12,081 ภาพ และ XAI 123 ภาพต่อเงื่อนไข หาก Colab ตัดการเชื่อมต่อ เปิด notebook เดิมและรันใหม่ จะเริ่มจาก epoch/ภาพที่บันทึกสำเร็จแล้ว

ถ้าค่าตั้งหรือแพ็กเกจเปลี่ยน ให้คืนเวอร์ชันเดิมจาก `shared/artifacts/experiment.json` หรือเลือก `OUTPUT_ROOT` ใหม่ เพื่อไม่ผสมผลจากต่างการทดลอง

## ผลลัพธ์

เส้นทางด้านล่างอยู่ภายใน `OUTPUT_ROOT` ซึ่งค่าเริ่มต้นคือ `Notebook_2/runs/train_100_per_class/`

- `CNN/results/<condition>/best.pt`: checkpoint ที่ validation loss ต่ำสุด พร้อม config, manifest hashes และ package versions
- `CNN/results/<condition>/last.pt`: น้ำหนักและ optimizer สำหรับรันต่อจาก epoch ล่าสุด
- `CNN/results/<condition>/test_predictions.csv`, `metrics.json`, `history.csv`, `confusion_matrix.png`: ผลจำแนกจริง
- `shared/artifacts/dae_checkpoint/`: DAE ที่ฝึกจาก train subset เท่านั้น; L1–L3 ใช้ checkpoint เดียวกัน
- `shared/artifacts/manifests/`: train, validation, test, XAI และ SHAP background IDs ที่ใช้ร่วมกัน
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

แพ็กเกจหลัก: PyTorch + torchvision ที่จับคู่เวอร์ชันกัน, SHAP, NumPy, pandas, OpenCV, PyWavelets, scikit-learn, matplotlib และ nbformat สำหรับสร้าง notebook ใหม่ โค้ดวิจัยใช้ค่าตั้งเดียวกันทุกเงื่อนไข และบันทึกเวอร์ชันที่ใช้จริงในแต่ละการทดลอง
