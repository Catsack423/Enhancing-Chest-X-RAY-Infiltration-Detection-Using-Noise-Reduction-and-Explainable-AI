# ข้อแลกเปลี่ยนจากการปรับความเร็ว training (2026-10-13)

เอกสารนี้บันทึกว่าปรับอะไรเพื่อให้ training เร็วขึ้น ได้อะไร และเสียอะไร ใช้ประกอบการเขียนหัวข้อ Methods/Limitations และการตัดสินใจเรื่องการรันซ้ำ

## สิ่งที่เปลี่ยน

| ส่วน | ก่อน | หลัง | ไฟล์ |
|---|---|---|---|
| DataLoader workers (DAE) | 0 (hardcode, config ไม่มีผล) | `dae.num_workers: 4` | `shared/training.py`, `configs/release_full.json` |
| DataLoader workers (CNN) | `cnn.num_workers: 0` | `4` | `configs/release_full.json` |
| `persistent_workers` | ปิด | เปิดเมื่อ workers > 0 | `shared/training.py` (`loader`) |
| cuDNN ตอน training | `deterministic=True`, `benchmark=False`, `use_deterministic_algorithms(True)` | `deterministic=False`, `benchmark=True` (`dae.deterministic` / `cnn.deterministic: false`) | `shared/experiment.py` (`deterministic_backend`), `shared/training.py` (`_training_backend`) |
| Memory format ตอน training | NCHW | `channels_last` (`dae.channels_last` / `cnn.channels_last: true`) | `shared/training.py` |

ส่วนที่ยัง deterministic เหมือนเดิม:

- การสร้าง cache DAE+CLAHE (DAE inference) เพราะเรียก `deterministic_backend(True)` ก่อน `prep.prepare`
- Test evaluation (`evaluate_classifier`)
- Grad-CAM และ SHAP (`shared/xai.py` เรียก `set_seed` ที่ค่า default เป็น deterministic)
- หลัง DAE training จบหรือ error จะคืนค่าเป็น deterministic เสมอ (`try/finally`)

## ผลที่วัดได้ (NVIDIA L4, DAE batch 64, AMP fp16, ข้อมูลสังเคราะห์บน GPU)

| deterministic | channels_last | images/s |
|---|---|---|
| True | False | 37.5 |
| True | True | 36.3 |
| False | False | 136.3 |
| False | True | 187.9 |

ตอนเทรนจริงก่อนปรับได้ 28.8 images/s. หลังเพิ่ม `num_workers` ได้ 37 images/s ซึ่งชนเพดาน GPU แบบ deterministic แล้ว

ข้อจำกัดของตัวเลขชุดนี้:

- วัดแค่ DAE ด้วยข้อมูลสุ่มบน GPU ไม่ใช่เวลา end-to-end ของ epoch จริง
- ยังไม่ได้วัด CNN (ResNet50) ดังนั้นยังไม่รู้ว่า CNN จะเร็วขึ้นเท่าไร
- ถ้า CPU ฝั่ง DataLoader ส่งข้อมูลไม่ทัน 188 images/s คอขวดจะย้ายกลับไปที่ CPU และความเร็วจริงจะต่ำกว่านี้

## ข้อเสีย

### 1. รันซ้ำแล้วไม่ได้ผลเหมือนเดิมทุกบิต

ข้อนี้เป็นข้อเสียหลัก

- seed ยังคุมลำดับภาพในแต่ละ epoch, synthetic noise ของ DAE, dropout และน้ำหนักเริ่มต้นได้เหมือนเดิม
- แต่ cuDNN แบบ non-deterministic ใช้ atomic add และอาจเลือก algorithm ต่างกันในแต่ละครั้ง ลำดับการบวกเลขทศนิยมจึงต่างกัน ทำให้ gradient ต่างกันในระดับ 1e-7 ถึง 1e-4
- ความต่างเล็ก ๆ นี้สะสมตลอด 20 epoch weight สุดท้ายจึงต่างกัน และ AUC, F1, Grad-CAM IoU และ SHAP IoU ที่ได้ก็อาจต่างกันเล็กน้อยด้วย
- ยังไม่ได้วัดว่าต่างกันเท่าไร โดยทั่วไปจะน้อยกว่าความต่างระหว่าง seed แต่ยืนยันไม่ได้จนกว่าจะทดลอง (ดูหัวข้อ "วิธีวัดผลกระทบ")

ผลที่ตามมา:

- ผู้อื่นที่รันโค้ดเดียวกัน seed เดียวกัน จะได้ตัวเลขใกล้เคียงแต่ไม่ตรงกันเป๊ะ ต้องระบุไว้ใน paper
- เอา checkpoint hash (`best.pt`) มาเทียบเพื่อยืนยันว่ารันซ้ำได้ผลเดิมไม่ได้แล้ว
- เลือก best epoch จาก validation loss ต่ำสุด ถ้า epoch สองตัวมี loss ใกล้กันมาก การรันซ้ำอาจได้ best epoch ต่างกัน

### 2. Resume กลางทางให้ผลต่างจากการรันรวดเดียว

ถ้า Colab หลุดแล้ว resume จาก `last.pt` ผลจะต่างจากการรันรวดเดียวจนจบเล็กน้อย ระบบยังทำงานถูกต้อง (provenance ผ่าน, RNG state ถูก restore) แต่การบอกว่า "resume แล้วได้ผลเหมือนไม่เคยหยุด" ไม่จริงแบบทุกบิตอีกต่อไป

### 3. แยกผลของ method ออกจาก noise ได้ยากขึ้นเล็กน้อย

ความแปรปรวนจาก non-determinism บวกเพิ่มจากความแปรปรวนระหว่าง seed ถ้า 2 method ต่างกันน้อยมาก (เช่น Median 3 กับ Median 5) ความต่างอาจกลืนไปกับ noise ได้ง่ายขึ้น

ส่วนที่ยังดีอยู่: ความแปรปรวนนี้ถูกนับรวมใน mean ± SD ข้าม 3 seeds แล้ว ข้อสรุปจึงไม่ได้ลำเอียง แค่ SD อาจกว้างขึ้นนิดหน่อย

### 4. `cudnn.benchmark` มีผลข้างเคียงของมันเอง

- ช่วงแรกของการรันจะช้ากว่าปกติ เพราะ cuDNN ทดลอง algorithm หลายแบบสำหรับ input shape แต่ละขนาด batch สุดท้ายของ epoch ที่เล็กกว่าปกติก็ทำให้ทดลองใหม่อีกรอบ
- algorithm ที่ถูกเลือกขึ้นกับจังหวะการวัด จึงอาจต่างกันระหว่าง session เป็นอีกแหล่งของ non-determinism
- algorithm ที่เร็วอาจใช้ workspace VRAM มากกว่า batch 64 ใช้ 17.8/22.0 GiB ตอน deterministic ตอนนี้จึงมีความเสี่ยง OOM สูงขึ้น ตอน benchmark ที่ batch 64 ยังไม่ OOM แต่ probe เลือก batch (`tools/dae_gpu_batch.py`) ยังวัดแบบ deterministic ไม่ได้รวม workspace ที่เพิ่มขึ้น

### 5. ผลเก่ากับผลใหม่เอามาปนกันไม่ได้

config และ source code hash เปลี่ยน ทำให้ `config_hash` เปลี่ยน:

- run ใน `runs_old` เอามา resume ด้วยโค้ดใหม่ไม่ได้ ระบบจะ error `Incompatible experiment config_hash`
- ห้ามเอา checkpoint หรือ metrics จาก run เก่า (deterministic) มารวมตารางกับ run ใหม่ ถ้าจะเทียบต้องรันใหม่ทั้งหมดด้วย config เดียวกัน

### 6. ผลต่างตาม GPU ชัดขึ้น

แม้แบบ deterministic ผลบน L4 กับ A100 ก็ต่างกันอยู่แล้ว แต่ `cudnn.benchmark` ทำให้ผลขึ้นกับ GPU และสภาพเครื่องมากขึ้น ควรรันทุก seed และทุก condition บน GPU รุ่นเดียวกัน และบันทึกรุ่น GPU ไว้ (`gpu_batch_probe.json` บันทึกให้แล้ว)

### 7. DataLoader workers ใช้ RAM และ CPU มากขึ้น

- 4 worker = 4 process เพิ่ม แต่ละตัวมี mmap ของ cache shard (สูงสุด `open_shards` = 4 shard ต่อ worker) และ batch ที่ prefetch ไว้ RAM ของระบบจะสูงขึ้น Colab ที่ไม่ได้เปิด High-RAM อาจโดน kill
- ถ้า VM มี vCPU น้อยกว่า 4 worker จะแย่ง CPU กันเอง ไม่เร็วขึ้นแต่ไม่ผิด
- error ใน worker debug ยากกว่า เพราะ traceback มาจาก process ลูก
- ข้อนี้ไม่กระทบผลวิจัย noise ของ DAE seed ตามชื่อภาพ (`SeedSequence([seed, epoch, image_id])`) และลำดับ shuffle มาจาก generator ใน main process จำนวน worker จึงไม่เปลี่ยนข้อมูล

### 8. `channels_last` มีผลข้างเคียงเล็กน้อย

- checkpoint เก็บ tensor แบบ channels_last stride โหลดกลับเข้า model NCHW ได้ปกติ (`load_state_dict` copy ให้) แต่ file hash จะต่างจาก checkpoint ที่บันทึกแบบ NCHW
- ถ้าใช้เดี่ยว ๆ แบบ deterministic จะช้าลงนิดหน่อย (37.5 เป็น 36.3 images/s) ได้ประโยชน์เฉพาะเมื่อใช้คู่กับ non-deterministic

## สิ่งที่ต้องเขียนใน paper

ตัวอย่างข้อความ Methods:

> Training of the DAE and ResNet50 used non-deterministic cuDNN kernels (`cudnn.benchmark=True`) with channels-last memory format for throughput. Random seeds (42, 43, 44) fixed data order, synthetic noise, dropout and initialization, but floating-point accumulation order on the GPU was not fixed, so repeated runs with the same seed may differ slightly. This run-to-run variation is included in the reported mean ± SD across seeds. DAE denoising inference, test-set evaluation, Grad-CAM and SHAP were computed with deterministic algorithms. All runs used a single NVIDIA L4 GPU.

ใน Limitations ระบุว่า reproducibility เป็นระดับสถิติ ไม่ใช่ระดับบิต

## วิธีวัดผลกระทบ (แนะนำถ้ามีเวลา)

1. รัน condition เดียว (เช่น `baseline`) seed 42 สองครั้งใน OUTPUT_ROOT แยกกัน
2. เทียบ test AUC, F1 และ Grad-CAM IoU ระหว่าง 2 run
3. ถ้าความต่างน้อยกว่า SD ระหว่าง seed มาก (เช่น น้อยกว่า 1/5) ก็เขียนใน paper ได้ว่า non-determinism มีผลน้อย

## วิธีย้อนกลับเป็นแบบ deterministic

แก้ `configs/release_full.json`:

```json
"dae": { "deterministic": true, "channels_last": false, ... },
"cnn": { "deterministic": true, "channels_last": false, ... }
```

`num_workers: 4` เก็บไว้ได้ เพราะไม่กระทบ reproducibility config เปลี่ยนแล้วต้องใช้ OUTPUT_ROOT ใหม่ และ DAE จะกลับไปที่ประมาณ 37 images/s (~20 ชม. สำหรับ DAE 3 seeds)
