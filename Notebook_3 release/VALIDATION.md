# Validation — v3 release

ตรวจบนเครื่อง local วันที่12ตุลาคม2026; Google Driveจริงยังไม่ได้mountในsessionนี้

- Unit/integration tests: **34 passed** (6.36วินาทีใน `.gpu` Python) ครอบคลุมpatientpartition,strictlabels,ครบทุกภาพ,partialbatch16+12,AMPskip,BNfreeze,warmup/resumeRNG,pixelprovenance,corruptcache/heatmaps,SHAPจริงและpairedpatientbootstrap
- Frozen full manifestsตรวจNIHmetadata/splits/bboxบนเครื่อง: Train46,108/Validation11,719/Test12,081/XAI123/background32; ไม่มีpatientleakageและเก็บpilotเดิมครบ
- GPU: **NVIDIA GeForce RTX 4060 8GB**, PyTorch2.13.0+cu130/torchvision0.28.0+cu130, SHAP0.51.0
- GPUpipelineทั้ง12conditionsผ่าน DAE/CNN/classification/Grad-CAM/SHAP/reports และ independent reopen/resume โดยSHAของbest/last/history/predictions/markersคงเดิม
- GPUtinyrunใช้Train4/Val4/Test4/XAI2/background2; CNN4epochsทดสอบhead-only1–3→layer4epoch4 และมีoptimizerupdateจริงทุกepoch; DAE1epoch
- Smokeใช้randominitializationและFP32เพื่อทดสอบgraph/optimizer ไม่ใช่ImageNetfulltraining ไม่ใช่ผลAccuracyของงานวิจัย
- เวลาของGPUtinyintegrationรวมresumeตรวจhashและวาดกราฟ: **315.96วินาที** ห้ามใช้เวลาtinyrunคูณตรงเพื่อประมาณเวลาfullstudy
- CapacityprobeDWT5views,32backgrounds(duplicateภาพsmokeเพื่อวัดshape/memory), SHAP200samples/batch8 ผ่าน: peakallocated **4157.7MiB** บนRTX4060 ไม่ใช่การประเมินXAI123ภาพจริง
- Nativeheatmaps1024², signedview/RGBarrays,empty-maps retained,overall/pure/mixedrisksและpatientCIมีschema/arrayvalidationก่อนถือว่าcomplete
- 9notebooksผ่านnbformatและASTcode-cellvalidation; source-onlyZIPไม่รวมdataset/weights/runoutputs

## ข้อจำกัดที่ยังไม่ได้วัด

ยังไม่รัน36CNN+3DAEบนfulltraining และยังไม่มีscores/benchmarkของv3releaseเต็ม
ยังไม่ได้อ่าน/checksumภาพบนDriveของผู้ใช้; notebookจะตรวจmetadata/pathlayout/ภาพทั้งหมดเมื่อเตรียมcacheครั้งแรก
Driveoperationquota/latencyและColabruntimehardwareต้องตรวจบนruntimeจริง
ทดสอบWindowslocalCUDAและdirect/nestedfolderfixturesแล้ว; ความเข้ากันได้ของTorch/torchvisionpairในColabถูกบันทึกตอนrunและปฏิเสธresumeเมื่อversionsเปลี่ยน

## ทำซ้ำ

```powershell
python -m unittest discover -s tests -v
python validate_notebook_gpu.py --data-root D:/kaggle_cache/datasets/nih-chest-xrays/data/versions/3
```

ใช้environmentที่มีdependenciesในrequirements.txtและPyTorchCUDAที่ตรงกับtorchvision
หลักฐานGPUอยู่`.runtime/gpu_validation/validation.json`และ`shap_capacity.json` (ignored; ไม่อยู่sourceZIP)
