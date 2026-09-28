# ResNet50 classification pilot

โฟลเดอร์นี้เก็บ notebook ฝึกโมเดลและผลจำแนก Normal (No Finding) กับ Infiltration บนชุดนำร่องเดียวกันทุกวิธี. ผลจาก metrics.json ทั้ง 7 โฟลเดอร์อยู่ใน README นี้เพื่ออ่านได้โดยไม่ต้องเปิดไฟล์สรุปแยก

## วิธีทดลอง

- [colab_cnn_pilot.ipynb](colab_cnn_pilot.ipynb): ฝึก ResNet50 ที่เริ่มจาก ImageNet, freeze backbone แล้วฝึก classifier head บน Colab; หนึ่งวิธีเตรียมภาพต่อหนึ่งรอบ
- ภาพ 224×224, seed 42, batch size 16, learning rate 0.0001, 50 epochs ต่อวิธีตาม metrics.json
- ต่อวิธีมี train 320, validation 80 และ test 100 ภาพ (Normal 50, Infiltration 50). split_manifest.csv ในทั้ง 7 วิธีตรงกันครบ 500 รายการ; ใช้ official train_val/test membership ของ NIH และคัดหนึ่งภาพต่อคนไข้
- Median L1/L2/L3 ใช้ kernel 3/5/7. CLAHE+DWT L1/L2/L3 เป็น preset clip 2/4/8, DWT depth 1/2/3, threshold multiplier 0.5/1/2
- ข้อมูลภาพนำเข้าต้นทางอยู่ใน [../../data/teachable_machine_pilot](../../data/teachable_machine_pilot/README.md). Positive class คือ Infiltration

## ผลบน test 100 ภาพต่อวิธี

| วิธี | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.560 | 0.548 | 0.680 | 0.607 | 0.575 |
| Median L1 | 0.560 | 0.548 | 0.680 | 0.607 | 0.572 |
| Median L2 | 0.560 | 0.554 | 0.620 | 0.585 | 0.576 |
| Median L3 | 0.570 | 0.561 | 0.640 | 0.598 | 0.568 |
| CLAHE+DWT L1 | 0.580 | 0.569 | 0.660 | 0.611 | 0.581 |
| CLAHE+DWT L2 | 0.570 | 0.556 | 0.700 | 0.619 | 0.614 |
| CLAHE+DWT L3 | 0.600 | 0.583 | 0.700 | 0.636 | 0.642 |

CLAHE+DWT L3 สูงสุดใน pilot นี้ โดย accuracy 60/100 เทียบกับ baseline 56/100; ความต่าง 4 ภาพยังไม่พอจะสรุปว่าเหนือกว่าแน่นอน. Median ทั้งสามระดับใกล้ baseline. CLAHE+DWT เปลี่ยนสามพารามิเตอร์พร้อมกัน จึงยังแยกผลของ clip limit, depth และ threshold ไม่ได้

## ไฟล์ผลใน outputs/

แต่ละโฟลเดอร์ของ 7 วิธีเก็บ metrics.json (ค่ารวมและ epochs_run), history.csv (การฝึก), split_manifest.csv (ภาพและ split), test_predictions.csv (ผลรายภาพ) และ confusion_matrix.png (แถวจริง/คอลัมน์ทำนาย เรียง Normal, Infiltration):

- [baseline_cnn_results](outputs/baseline_cnn_results/metrics.json)
- [median_L1_cnn_results](outputs/median_L1_cnn_results/metrics.json), [median_L2_cnn_results](outputs/median_L2_cnn_results/metrics.json), [median_L3_cnn_results](outputs/median_L3_cnn_results/metrics.json)
- [clahe_dwt_L1_cnn_results](outputs/clahe_dwt_L1_cnn_results/metrics.json), [clahe_dwt_L2_cnn_results](outputs/clahe_dwt_L2_cnn_results/metrics.json), [clahe_dwt_L3_cnn_results](outputs/clahe_dwt_L3_cnn_results/metrics.json)

## ข้อจำกัด

เป็น pilot ขนาด 500 ภาพต้นทาง ไม่ใช่ผลบน NIH ทั้งชุด. ค่า accuracy 0.56–0.60 และ AUC 0.568–0.642 ยังต่ำสำหรับการวินิจฉัย. ผลนี้ไม่มี DAE+CLAHE หรือ XAI-IoU; ต้องประเมินแยกตาม guideline. การฝึก 50 epochs ไม่ยืนยันความสามารถทั่วไปกับภาพใหม่. สำหรับงาน over-smoothing ต้องตรวจภาพและวัด heatmap เทียบ bbox พร้อมช่วงความเชื่อมั่น
