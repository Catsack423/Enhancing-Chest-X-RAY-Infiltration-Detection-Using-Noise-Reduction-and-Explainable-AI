# SHAP

ผลเปรียบเทียบทุกเงื่อนไขอยู่ใน comparison/metrics_all_conditions.csv

IoU ใช้ threshold 0.3 (หลัก) / 0.5 (เสริม); Pointing Game แบบ strict; bootstrap ระดับคนไข้ 2,000 รอบใน full mode

SHAP ประเมินเฉพาะ positive attribution ของ Infiltration logit; signed attribution เก็บใน heatmaps/*.npz และแสดงแดง/น้ำเงินใน overlays/
