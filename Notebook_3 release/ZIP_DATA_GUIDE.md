# ZIP-backed Colab preparation

1. หยุด cell เก่าที่อ่านภาพจาก Drive และยกเลิก cells ที่ต่อคิว ไม่ต้องลบ working cache
2. อัปโหลด **13 ZIP + dataset_zip_index.json** จาก `.local-artifacts/v3_dataset_zip` ไป **MyDrive/nih -xray/data/v3_selected_zip/** (ไม่ต้องอัปโหลด `.record.json`)
3. แทน source ZIP ที่ **MyDrive/nih -xray/Notebook_3_release_colab.zip** ด้วยไฟล์ใหม่ แล้วเปิด **run_all.ipynb รุ่นใหม่** บน Colab การแทน ZIP อย่างเดียวไม่แก้ cells ใน notebook ที่เปิดค้างไว้
4. รัน setup: copy ZIP จาก Drive พร้อม % และ MiB/s → ตรวจ SHA-256 → แตกไฟล์บน VM พร้อมจำนวนภาพ → ลบ ZIP สำเนา VM เพื่อคืนพื้นที่ (ZIP บน Drive ยังอยู่)
5. DATA_ROOT จะเป็น `/content/cxr_v3_originals` แล้วรัน preparation / DAE / CNN ตามเดิม cache อยู่ `/content/cxr_v3_cache` และ outputs อยู่ Drive path เดิม

ZIP นี้เก็บ **ภาพ PNG ต้นฉบับ 70,012 ภาพ** เฉพาะ frozen manifests ทั้งห้าชุด ไม่ย่อภาพก่อนแพ็ก ไม่เปลี่ยน labels/splits/XAI/native resolution ใช้ ZIP_STORED เพราะ PNG บีบอัดอยู่แล้ว payload 26.26 GiB แยกตาม 12 image folders และ metadata อีกหนึ่ง ZIP

ไม่ได้รับประกันเวลา transfer: ยังขึ้นกับ upload และ Drive throughput แต่หลัง copy ZIP แล้วไม่มีการเปิดต้นฉบับผ่าน Drive ทีละไฟล์อีก ขั้นแตก ZIP/ย่อภาพเป็น CPU; ใช้ GPU runtime เดิมเพื่อเทรนต่อโดยไม่เปลี่ยน VM

ถ้าหยุดกลางทาง ให้รัน setup ใหม่ ส่วนที่แตกครบตรวจ hashes แล้วใช้ต่อได้ ส่วนที่ยังไม่ครบทำใหม่ ภาพใน VM และ cache อาจหายเมื่อ runtime เปลี่ยน; ให้อ่าน ZIP จาก Drive ใหม่ ส่วน outputs/checkpoints ไม่หาย

แพ็ก dataset ซ้ำจาก Windows:

```powershell
.\.gpu\Scripts\python.exe 'Notebook_3 release/tools/pack_dataset.py' --data-root 'D:/kaggle_cache/datasets/nih-chest-xrays/data/versions/3' --output 'D:/kaggle_cache/datasets/nih-chest-xrays/.local-artifacts/v3_dataset_zip'
```

สร้าง notebooks/source ZIP รุ่นนี้ซ้ำด้วย `python tools/build_zip_release.py` จากโฟลเดอร์ release ไม่ใช้ builder เดิมโดยตรง Production Python ที่คำนวณ scientific fingerprint ไม่เปลี่ยน; tools อ่าน ZIP และคืน original bytes เท่านั้น dataset ZIP และ generated source ZIP ไม่เข้า Git
