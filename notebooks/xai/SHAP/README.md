# SHAP attribution

โมดูลนี้แสดง positive/negative attribution ของภาพ Normal และ Infiltration และสำรวจการเปลี่ยนแปลงเมื่อใช้ Median หรือ CLAHE+DWT. README นี้เป็นสรุปหลักของโฟลเดอร์; reports/ เก็บ CSV เท่านั้น

## ไฟล์ในโฟลเดอร์

- [shap_normal_vs_infiltration.ipynb](shap_normal_vs_infiltration.ipynb): เปรียบเทียบ attribution, ภาพและ matrix
- [shap_utils.py](shap_utils.py): ฟังก์ชัน attribution; [build_shap_notebook.py](build_shap_notebook.py): สร้าง notebook ใหม่
- [generate_all_shap_artifacts.py](generate_all_shap_artifacts.py): สร้างภาพและ CSV
- [output/shap_comparison_normal_vs_infil.png](output/shap_comparison_normal_vs_infil.png): Normal เทียบ Infiltration
- [output/shap_denoise_effects_comparison.png](output/shap_denoise_effects_comparison.png): attribution หลังปรับภาพ
- [output/confusion_matrix_shap_classification.png](output/confusion_matrix_shap_classification.png), [output/confusion_matrix_shap_localization.png](output/confusion_matrix_shap_localization.png): ภาพ matrix
- [reports/confusion_matrix_summary.csv](reports/confusion_matrix_summary.csv): ตัวเลขสรุปสามวิธี

## ผลที่บันทึกไว้

CSV ระบุ baseline / Median L2 / CLAHE+DWT L3 มี classification accuracy 91.7% / 93.3% / 95.0% และ Pointing Game hit rate 20.0% / 16.7% / 23.3%. เป็นตัวเลขจากสคริปต์สร้าง artifact ที่กำหนดค่าไว้ล่วงหน้า; ยังไม่มีหลักฐานรายภาพหรือ 95% CI ในโฟลเดอร์นี้ จึงไม่ควรตีความว่าเป็นผล SHAP ที่ผ่าน protocol หลัก. ภาพที่เก็บไว้สื่อว่า attribution เปลี่ยนเมื่อ denoise แต่ยังสรุปเรื่อง over-smoothing เชิงสถิติไม่ได้

## การรัน

ตั้ง working directory เป็นโฟลเดอร์นี้แล้วเปิด notebook หรือรัน python generate_all_shap_artifacts.py. สคริปต์อ่าน manifest 200 ภาพจาก ../Gradcam และภาพ NIH จาก data/versions/3. หากต้องการผลหลักต้องใช้ classifier ที่ฝึกสองคลาสบน official split, ประเมิน bbox 123 ภาพ และคำนวณ bootstrap CI.
