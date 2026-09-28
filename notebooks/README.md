# คู่มือ notebooks

งานนี้ศึกษาผลของการเตรียมภาพต่อการจำแนก Infiltration และตำแหน่งที่ XAI ชี้บนภาพ NIH ChestX-ray14. ข้อกำหนดการทดลองหลักอยู่ใน [guideline.md](../guideline.md): negative class คือ No Finding เท่านั้น, ใช้ official patient-level split, backbone เดียวกันทุกวิธี และประเมิน XAI บน bbox Infiltration 123 ภาพพร้อม 95% bootstrap CI.

## แผนผัง

| โฟลเดอร์ | เนื้อหา | เริ่มอ่าน |
|---|---|---|
| [denoising/](denoising/README.md) | วิธีเตรียมภาพ: raw, Median, CLAHE+DWT, DAE+CLAHE และ Gamma เสริม | [Median_CLAHE_DWT](denoising/Median_CLAHE_DWT/README.md) |
| [xai/](xai/README.md) | Grad-CAM, SHAP, Score-CAM, lung masking และการวัดตำแหน่งกับ bbox | [Gradcam](xai/Gradcam/README.md) |
| [classification/](classification/README.md) | CNN pilot และการเปรียบเทียบ DenseNet121 กับ ResNet50 | [cnn](classification/cnn/README.md) |
| [data_exploration/](data_exploration/README.md) | สำรวจ bbox และตัวอย่างภาพ | [BBox](data_exploration/BBox/README.md) |
| [data/](data/README.md) | ภาพนำร่องที่เตรียมสำหรับ Teachable Machine/Colab | [teachable_machine_pilot](data/teachable_machine_pilot/README.md) |

แต่ละโมดูลเก็บ notebook, โค้ดสร้าง notebook, utility, output ภาพ และตารางข้อมูลไว้ด้วยกัน README ในโมดูลอธิบายวัตถุประสงค์ วิธีรัน ไฟล์ทั้งหมด ผลที่อ่านได้ และข้อจำกัด; reports/ เก็บเฉพาะ CSV ผลดิบหรือผลสรุปแบบตาราง

## อ่านตามลำดับ

1. [ชุดข้อมูลนำร่อง](data/teachable_machine_pilot/README.md) และ [bbox](data_exploration/BBox/README.md)
2. [วิธี Median/CLAHE+DWT](denoising/Median_CLAHE_DWT/README.md), [DAE+CLAHE](denoising/DAE_CLAHE/README.md)
3. [ผลจำแนก CNN](classification/cnn/README.md)
4. [Grad-CAM](xai/Gradcam/README.md), [SHAP](xai/SHAP/README.md), [XAI เชิงปริมาณ](xai/XAI_Evaluation/README.md)

## สถานะผลและการตีความ

- CNN ใน classification/cnn เป็น pilot 500 ภาพต้นทาง (train 320, validation 80, test 100 ต่อวิธี) จาก official membership ไม่ใช่ผลบนข้อมูลเต็ม; ค่า test accuracy อยู่ที่ 0.56–0.60
- XAI_Evaluation มี CSV 100 ภาพ Infiltration; ยังขาด 23 ภาพจากชุด bbox 123 ภาพ และยังไม่มี 95% bootstrap CI ตาม guideline
- ภาพ XAI หลายโมดูลใช้ ResNet50 น้ำหนัก ImageNet ที่ยังไม่ได้ฝึกสำหรับโจทย์สองคลาส จึงเป็นตัวอย่าง workflow และไม่ควรนำตัวเลขจากคนละโมดูลมาอ้างว่าเป็นการเปรียบเทียบโมเดลเดียวกัน
- DAE report เดิมระบุตัวเลขไม่ตรงกับ CSV ที่เก็บอยู่; README ใหม่ใช้ค่าจาก CSV และแยกข้อสังเกตจากข้อสรุปที่พิสูจน์แล้ว
- Gamma, Score-CAM, lung masking และ DenseNet เป็นงานเสริม ไม่ใช่ 4 กลุ่ม denoising หลัก

## Path และการรัน

ข้อมูลต้นฉบับอยู่ที่ [data/versions/3](../data/versions/3) จากราก repo. Notebook ที่ใช้ relative path ควรรันโดยตั้ง working directory เป็นโฟลเดอร์ของ notebook นั้น; สคริปต์ .py ใช้ตำแหน่งไฟล์ของตัวเอง. ไฟล์ manifest 200 ภาพอยู่ที่ [xai/Gradcam/sample_manifest_200.csv](xai/Gradcam/sample_manifest_200.csv), ส่วนภาพนำร่องสำหรับ Colab อยู่ใน [data/teachable_machine_pilot](data/teachable_machine_pilot/README.md).
