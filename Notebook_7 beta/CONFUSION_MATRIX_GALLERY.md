# v7 beta — Confusion Matrix ครบ 12 วิธี × 16 states

แต่ละรูปเป็น test state เดียว เปรียบเทียบครบทั้ง 12 โมเดล; ไม่มีการเลือก Max Accuracy
แกนตั้ง = class จริง, แกนนอน = class ที่ทำนาย; Normal ในรูปหมายถึง No Finding
ทุก matrix มี 100 ภาพ (50 ต่อ class) และใช้สเกลสี 0–50 ภาพร่วมกัน
หัวรูปแสดง Accuracy จาก matrix และ Δ เทียบ Baseline เป็นจุดเปอร์เซ็นต์
ใช้ checkpoint/threshold จาก validation เดิม; ตารางตัวเลขต้นฉบับยังอยู่ใน results_summary/tables/

[กลับ README ข้อ 5.3](README.md)

## clean

![Confusion Matrix ของ clean ครบ 12 วิธี](results_summary/figures/confusion_all_methods/clean.png)

## gaussian_N1

![Confusion Matrix ของ gaussian_N1 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/gaussian_N1.png)

## gaussian_N2

![Confusion Matrix ของ gaussian_N2 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/gaussian_N2.png)

## gaussian_N3

![Confusion Matrix ของ gaussian_N3 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/gaussian_N3.png)

## poisson_N1

![Confusion Matrix ของ poisson_N1 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/poisson_N1.png)

## poisson_N2

![Confusion Matrix ของ poisson_N2 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/poisson_N2.png)

## poisson_N3

![Confusion Matrix ของ poisson_N3 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/poisson_N3.png)

## poisson_gaussian_N1

![Confusion Matrix ของ poisson_gaussian_N1 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/poisson_gaussian_N1.png)

## poisson_gaussian_N2

![Confusion Matrix ของ poisson_gaussian_N2 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/poisson_gaussian_N2.png)

## poisson_gaussian_N3

![Confusion Matrix ของ poisson_gaussian_N3 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/poisson_gaussian_N3.png)

## salt_pepper_N1

![Confusion Matrix ของ salt_pepper_N1 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/salt_pepper_N1.png)

## salt_pepper_N2

![Confusion Matrix ของ salt_pepper_N2 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/salt_pepper_N2.png)

## salt_pepper_N3

![Confusion Matrix ของ salt_pepper_N3 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/salt_pepper_N3.png)

## speckle_N1

![Confusion Matrix ของ speckle_N1 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/speckle_N1.png)

## speckle_N2

![Confusion Matrix ของ speckle_N2 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/speckle_N2.png)

## speckle_N3

![Confusion Matrix ของ speckle_N3 ครบ 12 วิธี](results_summary/figures/confusion_all_methods/speckle_N3.png)
