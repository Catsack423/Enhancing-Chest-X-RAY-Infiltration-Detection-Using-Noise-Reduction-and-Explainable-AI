# Explainable AI (XAI)

หมวดนี้รวมการอธิบายผลของภาพ Infiltration และการวัด heatmap กับกรอบรอยโรคของ NIH. [guideline](../../guideline.md) กำหนดผลหลักเป็น Grad-CAM และ SHAP บน bbox 123 ภาพ พร้อม 95% bootstrap CI

| โมดูล | สิ่งที่ทำ | ผลที่มี |
|---|---|---|
| [Gradcam](Gradcam/README.md) | raw-image Grad-CAM, manifest 100 Normal + 100 Infiltration | ภาพตัวอย่างและ bbox |
| [SHAP](SHAP/README.md) | positive/negative attribution และผลจาก denoising | ภาพและ CSV ตารางสรุป |
| [XAI_Evaluation](XAI_Evaluation/README.md) | Pointing Game, energy ใน bbox, IoU/Dice | CSV รายภาพ 100 และตารางรวม |
| [Score_CAM](Score_CAM/README.md) | เปรียบเทียบ Score-CAM กับ Grad-CAM | ภาพเปรียบเทียบ |
| [Lung_Segmentation](Lung_Segmentation/README.md) | สร้าง lung mask และเปรียบเทียบ Grad-CAM | ภาพขั้นตอนและภาพก่อน/หลัง |

งานเชื่อมกับ denoising ผ่าน [Median/CLAHE+DWT](../denoising/Median_CLAHE_DWT/README.md) และ [DAE+CLAHE](../denoising/DAE_CLAHE/README.md). ผลเชิงปริมาณที่มีขณะนี้ยังเป็น pilot 100 ภาพ ไม่มี CI และใช้ pretrained model ตามที่ README โมดูลระบุ.
