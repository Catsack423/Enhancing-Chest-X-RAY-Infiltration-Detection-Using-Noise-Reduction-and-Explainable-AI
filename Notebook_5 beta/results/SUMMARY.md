# ผลการทดลอง v5 beta (รันในเครื่อง RTX 4060, 2026-10-14)

ข้อมูลดิบ: `runs.jsonl` (1 แถวต่อ run, มี history ราย epoch และ bootstrap CI ของ AUC)

| ID | การตั้งค่า | seeds | test n | ROC-AUC mean ± SD | min–max |
|---|---|---:|---:|---:|---:|
| E1 | protocol อ้างอิงบน NIH, Infiltration-only | 5 | 100 | 0.512 ± 0.053 | 0.449–0.569 |
| E2 | protocol อ้างอิง, any Infiltration | 5 | 100 | 0.581 ± 0.075 | 0.474–0.652 |
| E3 | สลับ label (control) | 3 | 100 | 0.497 ± 0.023 | 0.475–0.521 |
| E4 | train แค่ layer4 + head | 3 | 100 | 0.531 ± 0.060 | 0.462–0.566 |
| E4 | train แค่ head | 3 | 100 | 0.515 ± 0.032 | 0.484–0.547 |
| E5 | fine-tune ทุกชั้น 2,000+2,000, Infiltration-only | 1 | 12,081 | 0.613 [0.600, 0.625] | – |
| E5 | fine-tune ทุกชั้น 2,000+2,000, any Infiltration | 1 | 15,973 | 0.675 [0.667, 0.684] | – |
| v4 | release v4 baseline (46,108 ภาพ) | 1 | 12,081 | 0.668 | – |

E0: สุ่ม test 50+50 จำนวน 1,000 ชุดจาก prediction ของ v4 (epoch 7) ได้ AUC 2.5–97.5 percentile
≈ 0.58–0.78, ค่าสูงสุด 0.81–0.86 ขึ้นกับ condition; ไม่มีชุดใดถึง 0.92

## ข้อสรุป

1. **ไม่พบ bug ที่ทำให้ v4 ต่ำ** — protocol ของ notebook อ้างอิงเมื่อรันบน NIH ได้ AUC เฉลี่ย 0.51
   (ใกล้ control ที่สลับ label 0.50) ไม่ใช่ 0.92; v4 ใช้ data เต็มได้ 0.668 สูงกว่าทุก run ที่ใช้
   protocol อ้างอิง รวมถึง E5 ที่ fine-tune ทุกชั้นด้วย 4,000 ภาพ (0.613)
2. **ช่องว่างส่วนใหญ่มาจาก dataset/label** — VinBigData ใช้ label จากรังสีแพทย์ ส่วน NIH ใช้ label ที่
   สกัดจาก report ด้วย NLP และ Infiltration เป็นหนึ่งใน class ที่ label noise สูงที่สุด
3. **นิยาม positive มีผลจริง** — นับ Infiltration ที่มีโรคร่วมเพิ่ม AUC ≈ +0.06 ทั้งในชุดเล็ก (E2 vs E1)
   และชุดใหญ่ (E5 any vs only) เพราะโรคร่วมทำให้ภาพผิดปกติชัดกว่า งานอ้างอิงใช้นิยามนี้
4. **test 100 ภาพแกว่งมาก** — ช่วง AUC ระหว่าง seed กว้าง ≈ 0.1–0.2 และ notebook อ้างอิงรายงาน
   seed เดียวโดยไม่มี CI; accuracy/F1 ของ test 50/50 เทียบกับ test prevalence 18% ไม่ได้
5. **E4 ชุดเล็ก**: การ freeze backbone ไม่ได้ทำให้ผลแย่ลงอย่างมีนัย (ทุกแบบ ≈ 0.51–0.53 ที่ 200 ภาพ)

## ข้อสังเกตต่อ notebook อ้างอิง (ยังไม่ได้รันซ้ำบน VinBigData)

- `drop_duplicates(subset=['image_id'])` เลือก annotation แถวแรกของรังสีแพทย์หลายคน ภาพที่คนหนึ่ง
  ให้ Infiltration แต่อีกคนให้ No finding อาจได้ label ใดก็ได้ตามลำดับแถว
- positive ไม่ตัดภาพที่มีโรคร่วม (เช่น consolidation, effusion) จึงเป็นโจทย์ "ผิดปกติ vs ปกติ" มากกว่า
- split ระดับภาพแบบสุ่ม; VinBigData ไม่มี patient ID จึงตรวจการรั่วระหว่างผู้ป่วยไม่ได้
- test 100 ภาพ seed เดียว; ไม่ได้ใช้ validation เลือก checkpoint (ใช้ epoch 7 เสมอ) ซึ่งไม่ทำให้รั่ว

## ทดลองต่อได้

- รัน `gap_experiments.py` บน VinBigData ด้วย label แบบ majority vote เพื่อดูว่า 0.92 คงอยู่ไหม
- E5 หลาย seed และขนาด data หลายระดับ (500/2,000/ทั้งหมด) เพื่อดู learning curve บน NIH
