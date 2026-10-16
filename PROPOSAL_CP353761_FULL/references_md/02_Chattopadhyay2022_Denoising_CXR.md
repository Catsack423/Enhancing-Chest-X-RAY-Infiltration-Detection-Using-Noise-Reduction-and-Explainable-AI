# Chattopadhyay (2022) — A Study on Various Common Denoising Methods on Chest X-ray Images

> ไฟล์ต้นฉบับ: `A_Study_on_Various_Common_Denoising_Methods_on_Che.pdf` (20 หน้า)
> สถานะในเค้าโครง: อ้างอิงข้อ 2 ใน `PROPOSAL_CP353761_FULL.md` (ใช้ใน Sec. 4.3 หัวข้อ Median Filter)

## ข้อมูลบรรณานุกรม (APA 7)

Chattopadhyay, S. (2022). A study on various common denoising methods on chest X-ray images. *Artificial Intelligence Evolution*, 3(2), 87–106. https://doi.org/10.37256/aie.3220221714

- ผู้เขียนคนเดียว นักวิจัยอิสระ (Bangalore, India)
- วารสาร open access (CC BY 4.0) ของสำนักพิมพ์ Universal Wiser Publisher
- Received 18 Jul 2022 / Accepted 16 Oct 2022

## สรุปสั้น (TL;DR)

เทียบตัวกรองเชิงพื้นที่ 4 แบบ (Moving Average, Gaussian = linear; Median, Bilateral = non-linear) ที่ **kernel 5×5 ค่าเดียว** บนภาพ CXR **3 ภาพ** วัดด้วย noise variance, SNR, MSE, PSNR และเวลาประมวลผล สรุปว่า **Median ดีที่สุด** (PSNR เฉลี่ย 34.52 dB, MSE เฉลี่ย 12.23) และให้เหตุผลว่า noise ใน CXR เป็น Poisson (shot noise) ซึ่ง median รับมือได้ดี

## 1. ภูมิหลังที่เปเปอร์ให้ไว้

- ประเภท noise: amplifier (Gaussian, additive), impulsive / salt-and-pepper (พบบ่อยใน radiograph), Poisson, speckle (multiplicative)
- วิธี denoise หลัก 3 ตระกูล: spatial filtering, transform-domain filtering, wavelet thresholding — เปเปอร์นี้ศึกษาเฉพาะ **spatial filtering**
- Linear filter เหมาะกับ additive noise; non-linear เหมาะกับ multiplicative / function-based noise
- Table 1 (literature review): งานก่อนหน้าส่วนใหญ่ (MRI, CT, CXR) พบว่า median และรุ่นปรับปรุงของมันทำงานดี
- หน้า 103: **"CXR image production uses photon-counting statistics that follow the Poisson process. Thus, the noise present in the CXR has the Poisson distribution and the noise is called the Poisson noise or Shot noise"** (อ้าง Thakur et al., 2016)

## 2. วิธีการ

| ขั้นตอน | รายละเอียด |
|---|---|
| ข้อมูล | CXR ผู้ชายวัยผู้ใหญ่ 3 ภาพจากคลินิก: Control (ปกติ, 240×198), Test-1 (135×144), Test-2 (138×130) — ไม่ได้ใช้เพื่อจำแนกโรค |
| เครื่องมือ | Python 3.8.3, OpenCV, skimage, numpy |
| Histogram Equalization | ทดลองก่อน denoise; พบว่า HE **เพิ่ม noise ราว 1%** จึง denoise บนภาพดิบแทน |
| วัด noise | noise variance (สมการ 3–5), SNR = σ_s / σ_n (สมการ 6–9) |
| ตัวกรอง | Moving Average (boxcar), Median, Gaussian, Bilateral — **kernel 5×5 ทั้งหมด** |
| ตัวชี้วัด | MSE, PSNR = 20·log10(MAX/√MSE), "Big(O)" เป็นเวลาวัดจริงหน่วย ms |
| ทดสอบใช้งาน | ภาพ TB 500 ภาพ resize เป็น 224×224, แบ่ง 75/25, ใช้ระยะ Euclidean ของค่าเฉลี่ยพิกเซลเทียบ centroid ("mean-of-means") ตัดที่ 10% |

## 3. ผลลัพธ์หลัก

### MSE (Table 5a) — ต่ำดี

| ภาพ | Moving avg | **Median** | Gaussian | Bilateral |
|---|---|---|---|---|
| Control | 22.0976 | 13.9699 | 12.6138 | 26.8809 |
| Test-1 | 22.625 | 11.7383 | 12.5888 | 26.6119 |
| Test-2 | 21.0355 | 10.9929 | 13.2159 | 27.0693 |
| เฉลี่ย | 21.9194 | **12.2337** | 12.8062 | 26.8540 |

### PSNR dB (Table 5b) — สูงดี

| ภาพ | Moving avg | **Median** | Gaussian | Bilateral |
|---|---|---|---|---|
| Control | 32.2517 | 35.616 | 34.9328 | 31.6817 |
| Test-1 | 31.7725 | 33.8001 | 35.5053 | 31.6851 |
| Test-2 | 27.7098 | 34.1479 | 31.539 | 29.4483 |
| เฉลี่ย | 30.578 | **34.5213** | 33.9924 | 30.9384 |

- เวลาเฉลี่ย (Table 6): Median 0.02873 ms เร็วสุด (เร็วกว่า Moving avg / Gaussian / Bilateral ตามที่ผู้เขียนอ้าง 76% / 67.8% / 57.8%)
- ทดสอบ TB 500 ภาพ: Median 92%, Gaussian 80%, Bilateral 72%, Moving avg 60%
- ข้อจำกัดที่ผู้เขียนยอมรับ: ภาพน้อย และไม่ได้ทดสอบวิธี denoise แบบอื่น

## 4. ข้อสังเกตเชิงวิพากษ์ (สำคัญก่อนอ้าง)

1. **เปเปอร์นี้ไม่ได้ศึกษาขนาด kernel หรือ over-smoothing** — ใช้ kernel 5×5 ค่าเดียวตลอด และไม่มีประโยคใดกล่าวว่า kernel ใหญ่ทำให้รายละเอียดหาย
   - ในเค้าโครง Sec. 4.3 เขียนว่า *"Chattopadhyay (2022) ระบุว่าหากใช้ขนาด Kernel ที่ใหญ่เกินไปจะทำให้เกิด Over-smoothing"* — **ข้ออ้างนี้ไม่มีในเปเปอร์** ควรแก้เป็น "Chattopadhyay (2022) พบว่า Median Filter (5×5) ให้ PSNR/MSE ดีที่สุดในกลุ่มตัวกรองเชิงพื้นที่บน CXR" แล้วหาแหล่งอื่นมาอ้างเรื่อง over-smoothing หรือเขียนเป็นสมมติฐานของงานเราเอง
2. **ขนาดตัวอย่าง 3 ภาพ** ความละเอียดต่ำมาก (~130–240 px) ไม่มีสถิติทดสอบนัยสำคัญ
3. **ไม่มี clean reference** — MSE/PSNR คำนวณระหว่างภาพดิบกับภาพที่กรองแล้ว จึงวัด "ความต่างจากต้นฉบับ" ไม่ใช่ "ความใกล้ภาพไร้ noise" ตัวกรองที่ทำอะไรน้อยที่สุดย่อมได้ PSNR สูง (งานเราแก้ด้วยการจำลอง Mixed Poisson-Gaussian noise บนภาพแล้ววัดเทียบภาพก่อนเติม noise — Sec. 5.4 ของเค้าโครง)
4. **ความไม่สอดคล้องภายใน**:
   - Table 2 คอลัมน์ "Actual" เป็นสถิติของ **ดัชนีพิกเซล** ไม่ใช่ค่าพิกเซล (เช่น count 47,520 = 240×198 และ mean 23,759.5 = (47,520 − 1)/2)
   - เนื้อความอ้างว่า Median มี noise variance เฉลี่ย 0.1073 แล้วสรุปว่าดีที่สุด ทั้งที่นิยามไว้เองว่า variance ต่ำ = noise น้อย และ Table 4b Median มี variance สูงสุด (0.1192)
   - Abstract ระบุ σ ของ PSNR = 0.9618 แต่ Table 5b = 0.9638; เนื้อความระบุ Gaussian MSE เฉลี่ย 12.6062 แต่ตาราง = 12.8062
   - ใน PSNR ภาพ Test-1 Gaussian (35.51) ชนะ Median (33.80) และใน MSE ภาพ Control Gaussian (12.61) ชนะ Median (13.97) — Median ไม่ได้ชนะทุกภาพตามที่สรุป
   - "Big(O)" ถูกใช้เป็นเวลาวัดจริง (ms) ไม่ใช่ความซับซ้อนเชิงอะซิมโทติก
5. การทดสอบ TB 500 ภาพไม่ใช่ classifier จริง (เทียบระยะค่าเฉลี่ยพิกเซลกับ centroid ของคลาสเดียว ไม่มี negative class) จึงอ้างเป็นหลักฐานประสิทธิภาพการวินิจฉัยไม่ได้

## 5. ความเกี่ยวข้องกับโครงงานเรา

| ประเด็นในเค้าโครง | วิธีใช้อย่างเหมาะสม |
|---|---|
| เลือก Median Filter เป็นกลุ่ม Traditional | อ้างได้ว่า Median เป็นตัวกรองเชิงพื้นที่ที่นิยมและได้ PSNR/MSE ดีบน CXR (หลักฐานระดับเบื้องต้น) |
| Noise ใน CXR เป็น Poisson | อ้างประโยคหน้า 103 ได้ (ควรอ้างแหล่งฟิสิกส์ภาพรังสีเพิ่ม) |
| ไม่ควรใช้ HE ก่อน denoise | ผลว่า HE เพิ่ม noise ~1% สนับสนุนว่าการเพิ่ม contrast อาจขยาย noise — สอดคล้องกับเหตุผลที่ใช้ CLAHE (มี clip limit) แทน HE ธรรมดา |
| Kernel 3/5/7 กับ over-smoothing | **อ้างเปเปอร์นี้ไม่ได้** — งานเราเป็นผู้ทดสอบเรื่องนี้เอง (เป็นช่องว่างงานวิจัย ไม่ใช่ข้อค้นพบเดิม) |
| PSNR/MSE เป็นตัวชี้วัด | ใช้เป็นตัวอย่างข้อจำกัดของการวัด PSNR โดยไม่มี clean reference |

## ประโยคที่อ้างตรงได้

- "The study finds that, Median filter is the best of the lot with the PSNR mean = 34.5213 ... MSE mean = 12.2337" (Abstract, p. 87)
- "HE has increased the noise level by 1%, probably by raising the contrast background noise, while decreasing the usable signal." (p. 99)
- "CXR image production uses photon-counting statistics that follow the Poisson process." (p. 103)
- "The limitations of this work are (i) less number of CXR images and (ii) other denoising methods that have been skipped..." (p. 104)
