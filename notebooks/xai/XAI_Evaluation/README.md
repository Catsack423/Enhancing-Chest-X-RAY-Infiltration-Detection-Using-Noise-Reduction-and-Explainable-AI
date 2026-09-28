# การประเมิน XAI เชิงปริมาณ

โมดูลนี้วัด Grad-CAM เทียบ bbox ของภาพ Infiltration 100 ภาพจาก manifest: Pointing Game แบบ strict/มี margin, สัดส่วนพลังงาน heatmap ใน bbox, IoU ที่ threshold 0.3/0.5 และ Dice ที่ 0.3. ค่านี้เป็น pilot; guideline กำหนดให้ประเมิน 123 ภาพพร้อม 95% bootstrap CI และแยก pure/mixed confound ในผลหลัก

## ไฟล์ในโฟลเดอร์

- [xai_quantitative_metrics.ipynb](xai_quantitative_metrics.ipynb): ดูผลรายภาพ
- [xai_eval_utils.py](xai_eval_utils.py): mask, Pointing Game, energy, IoU/Dice และ GradCAMGenerator
- [run_xai_quantitative_benchmark.py](run_xai_quantitative_benchmark.py): สร้าง CSV และภาพ; [build_xai_eval_notebook.py](build_xai_eval_notebook.py): สร้าง notebook
- [reports/xai_quantitative_metrics_detailed.csv](reports/xai_quantitative_metrics_detailed.csv): 100 แถวรายภาพ/วิธี
- [reports/xai_quantitative_summary.csv](reports/xai_quantitative_summary.csv): ค่าเฉลี่ยและ std ของ 4 วิธี
- [output/xai_metrics_comparison_barchart.png](output/xai_metrics_comparison_barchart.png), [output/xai_energy_boxplot.png](output/xai_energy_boxplot.png), [output/xai_visual_pointing_game_examples.png](output/xai_visual_pointing_game_examples.png): แผนภูมิและตัวอย่าง

## ผลจาก summary.csv

| วิธี | Hit strict | Hit margin | Energy ใน bbox | IoU @ 0.3 |
|---|---:|---:|---:|---:|
| Raw | 21% | 25% | 15.40% | 0.1287 |
| Median L2 | 23% | 27% | 17.02% | 0.1429 |
| CLAHE+DWT L2 | 19% | 21% | 15.91% | 0.1239 |
| DAE+CLAHE L2 | 26% | 30% | 17.50% | 0.1528 |

ค่า 30% เป็น hit แบบมี margin; strict คือ 26%. สคริปต์ใช้ ResNet50 pretrained ImageNet และ DAE checkpoint ที่มีอยู่ซึ่งยังยืนยันที่มาของ training split ไม่ได้; ยังไม่ใช่ classifier สองคลาสที่ฝึกคงที่ทุกวิธี, ไม่มี 95% CI และ n=100 แทน n=123 จึงยังสรุปผู้ชนะตาม protocol หลักไม่ได้

## การรัน

ตั้ง working directory เป็นโฟลเดอร์นี้แล้วรัน python run_xai_quantitative_benchmark.py หรือเปิด notebook. สคริปต์อ่าน manifest จาก ../Gradcam, preprocessing จาก ../../denoising และ NIH data/versions/3. การรันใหม่จะเขียน CSV/ภาพทับผลเดิมในโฟลเดอร์นี้.
