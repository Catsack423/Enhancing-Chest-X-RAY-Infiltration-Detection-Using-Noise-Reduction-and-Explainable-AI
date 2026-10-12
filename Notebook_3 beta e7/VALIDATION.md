# Validation — v3 beta e7

ตรวจเมื่อ 10 ตุลาคม 2026 ผ่านทั้งหมด ยังไม่ได้รันเต็ม train 320/test 100/XAI 123 ภาพเพื่อวัด Accuracy หรือยืนยันว่า e7 ดีกว่า e2

## Focused checks

`python -m unittest discover -s "Notebook_3 beta e7/tests" -v`: **13 tests ผ่าน**, 7.03 วินาที ไม่รวม import/startup

- Frozen head-only model 2,049 trainable parameters, optimizer update เฉพาะ head, backbone weights/buffers และ BatchNorm statistics ไม่เปลี่ยน
- Input gradients ทุก view และ Grad-CAM ของ frozen model ทำงาน
- Model class AST และ training module bytes เหมือน e2; config เหมือนเดิมทั้งหมดนอกจาก epochs/patience/output path
- Unit regression ให้ validation loss แย่ลงทุก epoch: training loop ยังฝึกครบ **7 epochs** และเลือก best epoch 1 แสดงว่าการจบ training ที่ 7 กับการเลือก best checkpoint เป็นคนละกติกา
- ทั้ง 5 manifests/hash เดิมครบ train 320/validation 80/test 100/XAI 123/background 32 และ epoch 1/2/7 ใช้ train IDs เดิมครบ
- โค้ด/ไฟล์ e2 และ v2 ที่ตรวจ SHA-256 ไม่เปลี่ยน
- Comparison ยอมรับเฉพาะ budget/patience ที่เปลี่ยน; ปฏิเสธ dropout หรือ package versions ที่ต่าง และไม่แทนผล e2 ที่หายด้วยคะแนนอื่น
- Notebook schema/syntax/clean outputs และ ZIP CRC/path/data bytes/manifests ผ่าน

## GPU notebook smoke และ resume

RTX 4060, torch 2.13.0+cu130/torchvision 0.28.0+cu130/SHAP 0.51.0:

- 4 train/4 validation/4 test, 2 bbox XAI/2 background; batch 2, **7 epochs**, SHAP nsamples 4/bootstrap 20
- ฝึก baseline/DWT/DFT ครบ 7 ทุกวิธี บันทึก/reload checkpoints และประเมิน Grad-CAM/SHAP ครบทั้ง 3 วิธี
- การรันครั้งแรกผ่าน training/XAI แล้วหยุดที่ตัวทดสอบ Windows แสดงข้อความไทยด้วย cp1252 แก้ stdout/stderr เป็น UTF-8 แล้ว resume ผลทดสอบเดิมสำเร็จ
- การตรวจหลังแก้ encoding execute 10 code cells (ไม่รวม export) และตรวจ resume ใช้ 36.07 วินาที **ไม่ใช่ benchmark fresh training 7 epochs**
- ทดสอบ comparison helper กับผล e2 smoke จริง: classification 6 แถว/ผลต่าง 3 วิธี/XAI ทั้งสอง budget และกราฟสร้างได้ ค่าตั้ง/versions/manifests ผ่านการตรวจ
- เทียบ history สอง epochs แรกของ e7 กับ e2 smoke: train loss, train_eval_loss, validation loss, gap และ train IDs **ตรงทุกค่าแบบ exact ทั้ง 3 วิธี**
- เซลล์ comparison ใน notebook แสดงไฟล์ที่ขาดเมื่อ E2_RESULTS_ROOT ไม่มีผลจริง ไม่ใช้ smoke outputs แทน

ผลอยู่ `.runtime/gpu_smoke/`: `smoke_validation.json`, `executed_smoke.ipynb`, `first_two_epochs_match.json` และ output ชุดเล็ก ไฟล์เหล่านี้ไม่รวมใน ZIP หรือ output หลัก และไม่รายงานคะแนนชุดเล็กเป็นผลวิจัย

## Portable package

เรียก `colab_data_from_package` จาก member `Notebook_3_e7/nih_cxr_subset.zip` จริง แตกไฟล์/ตรวจ hashes แล้วเปิด e7 experiment สำเร็จพร้อม train 320/validation 80/test 100/XAI 123/background 32, epochs 7/patience 7

ทดสอบด้วยตัว loader เดียวกับ Colab บนเครื่อง local ยังไม่ได้ทดสอบ Drive mount หรือ execute notebook บน Colab GPU

## ขอบเขตที่ยังไม่ได้วัด

ยังไม่รันเต็ม, ไม่ benchmark default batch 16/SHAP nsamples 200 ของ e7 และไม่มีผล Accuracy e2/e7 จริงในเครื่องนี้ ข้อมูล comparison ที่ทดสอบเป็น smoke เท่านั้น เวลา training/classification 1–5 นาทีและพร้อม XAI 45–90 นาทีใน README เป็นการประเมินจาก pipeline เดิม ไม่ใช่ benchmark เต็ม e7
