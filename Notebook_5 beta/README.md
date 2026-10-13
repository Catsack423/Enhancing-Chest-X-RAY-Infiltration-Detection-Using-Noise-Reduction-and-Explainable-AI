# Notebook v5 beta — ตรวจช่องว่างผลลัพธ์ NIH (v4) กับ notebook อ้างอิง (VinBigData)

notebook อ้างอิง `Seminae_source_code (1).ipynb` (ไม่ได้ commit; อยู่ที่ root ของ repo) รายงาน
ROC-AUC ≈ 0.92 แต่ release v4 ได้ ≈ 0.67 บน NIH ชุดนี้ replay protocol ของ notebook อ้างอิงบน NIH
แล้วเปลี่ยนทีละปัจจัย เพื่อแยกว่าช่องว่างมาจาก bug ใน pipeline ของเรา หรือจาก data/label/การประเมินผล

## ความต่างระหว่างสองงาน

| ปัจจัย | notebook อ้างอิง | release v4 |
|---|---|---|
| Dataset | VinBigData 512px (label จากรังสีแพทย์, มี bbox) | NIH ChestX-ray14 (label สกัดจาก report ด้วย NLP) |
| Positive | แถวใดก็ได้ที่ `class_name == Infiltration` (มีโรคอื่นร่วมได้) | Infiltration อย่างเดียว |
| Negative | `No finding` ของ annotation แถวแรกหลัง `drop_duplicates` | `No Finding` อย่างเดียว |
| Train | 100 + 100 balanced | 46,108 (prevalence 12%) |
| Test | 50 + 50 balanced, สุ่ม (n = 100) | official test เต็ม 12,081 (prevalence 18%) |
| Split | สุ่มระดับภาพ | official patient-level split |
| Model | ResNet50 fine-tune ทุกชั้น, BN train mode, CE 2 class | head 3 epoch → layer4 lr 1e-5, BN frozen, weighted BCE |
| Threshold | argmax (= 0.5) | Youden J บน validation |

## การทดลอง

`gap_experiments.py` (CLI) และ `gap_experiments.ipynb` (Colab/local) ใช้ฟังก์ชันชุดเดียวกัน
ทุก run แยกผู้ป่วยระหว่าง train/validation/test; train และ validation มาจาก `train_val_list.txt`,
test มาจาก `test_list.txt`

| ID | คำถาม | การตั้งค่า |
|---|---|---|
| E0 | test 50/50 ขนาด 100 ภาพทำให้ v4 ดูเหมือน 0.92 ได้ไหม | สุ่ม 1,000 ชุดจาก `comparison/epochs/*/epoch_7/test_predictions.csv` |
| E1 | protocol อ้างอิงบน NIH ได้เท่าไร | Infiltration-only, 100+100, ทุกชั้น, 7 epoch, 5 seed |
| E2 | label แบบอ้างอิง (Infiltration มีโรคร่วม) ง่ายกว่าไหม | เหมือน E1 แต่ positive = any Infiltration |
| E3 | pipeline เรียนจาก label มั่วได้ไหม (ต้องได้ ≈ 0.5) | เหมือน E1 แต่สลับ label ใน train |
| E4 | freeze backbone แบบ v4 แย่กว่า fine-tune ทุกชั้นไหม | trainable = layer4 / head |
| E5 | data มากขึ้น + fine-tune ทุกชั้น บน official test เต็ม | 2,000+2,000, test เต็ม, only/any |

รันในเครื่อง (ต้องมี `data/versions/3`):

```bash
PY=../.gpu/Scripts/python.exe bash run_matrix.sh
```

ผลแต่ละ run ต่อท้ายใน `results/runs.jsonl`; สรุปผลอยู่ใน `results/SUMMARY.md`
