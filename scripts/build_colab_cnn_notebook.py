"""Generate the self-contained Colab notebook for the Teachable Machine pilot."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "colab_cnn_pilot.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "colab": {"name": "colab_cnn_pilot.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

def md(source):
    nb.cells.append(nbf.v4.new_markdown_cell(source.strip()))

def code(source):
    nb.cells.append(nbf.v4.new_code_cell(source.strip()))

md("""
# ทดลองจำแนกภาพเอกซเรย์ด้วย CNN บน Google Colab

โน้ตบุ๊กนี้ใช้ภาพนำร่องที่เตรียมไว้ใน `outputs/teachable_machine_pilot` เพื่อแทนการฝึกบน Teachable Machine
โดยใช้ ResNet50 (CNN ที่มีน้ำหนักเริ่มต้นจาก ImageNet) และคลาส **Normal** กับ **Infiltration**
ฝึก **หนึ่งวิธี denoising ต่อหนึ่งรอบ**: baseline, median_L1–L3 หรือ clahe_dwt_L1–L3

ก่อนรัน: เปิดใน Google Colab แล้วเลือก **Runtime → Change runtime type → GPU** ถ้ามี GPU
จากนั้นรันเซลล์ตามลำดับและอัปโหลด ZIP ของวิธีที่ต้องการเพียงไฟล์เดียว
หรือกำหนด `ZIP_PATH` เป็นไฟล์ ZIP ใน Google Drive ที่ mount ไว้แล้ว
การรัน 7 วิธีให้เริ่ม runtime/รันเซลล์ใหม่ทีละ ZIP และใช้ค่าตั้งเดียวกันทั้งหมด

ชุดนำร่องนี้มี 200 ภาพต่อคลาสใน official train_val และ 50 ภาพต่อคลาสใน official test
โค้ดแบ่งภาพ train_val เป็นฝึก 160 และ validation 40 ภาพต่อคลาส (seed 42)
ภาพต้นทางเลือกไว้หนึ่งภาพต่อคนไข้ทั้งชุด จึงไม่ซ้ำคนไข้ระหว่าง train/validation/test
ผลนี้เป็นการทดลองขนาดเล็ก ไม่ใช่ผลวิจัยบนข้อมูลเต็ม และ label ไม่ระบุสาเหตุ เช่น COVID-19
DAE ยังไม่ได้ฝึก จึงยังไม่มี DAE+CLAHE
""")

code("""
import os, random, json, csv, zipfile
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 1e-4
CLASSES = ("Normal", "Infiltration")  # labels: 0, 1
ALLOWED_METHODS = {"baseline", "median_L1", "median_L2", "median_L3",
                   "clahe_dwt_L1", "clahe_dwt_L2", "clahe_dwt_L3"}
tf.keras.utils.set_random_seed(SEED)
print("TensorFlow:", tf.__version__, "GPUs:", tf.config.list_physical_devices("GPU"))
""")

md("""
## 1. เลือกภาพหนึ่งวิธี

ค่าเริ่มต้น `ZIP_PATH = ""` จะเปิดหน้าต่างให้เลือกไฟล์ เช่น `baseline.zip`
ถ้าเก็บไฟล์บน Google Drive ให้ mount Drive ใน Colab ก่อน แล้วตั้ง `ZIP_PATH` เป็น path ของ ZIP
ไฟล์ ZIP เดิมมี `train/` และ `test/` อยู่ข้างใน ไม่ต้องแตกไฟล์ด้วยตนเอง
""")

code("""
ZIP_PATH = ""  # ตัวอย่าง: "/content/drive/MyDrive/teachable_machine_pilot/baseline.zip"
if not ZIP_PATH:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("กรุณาอัปโหลด ZIP เพียงหนึ่งวิธีต่อหนึ่งรอบ")
    ZIP_PATH = next(iter(uploaded))

zip_path = Path(ZIP_PATH)
METHOD = zip_path.stem
if METHOD not in ALLOWED_METHODS:
    raise ValueError(f"ชื่อ ZIP ไม่ถูกต้อง: {METHOD}; ต้องเป็นหนึ่งใน {sorted(ALLOWED_METHODS)}")
if not zip_path.is_file():
    raise FileNotFoundError(zip_path)

DATA_ROOT = Path("/content/cxr_pilot") / METHOD
DATA_ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as archive:
    members = [m for m in archive.infolist() if not m.is_dir()]
    paths = [Path(m.filename) for m in members]
    for p in paths:
        if (len(p.parts) != 3 or p.parts[0] not in {"train", "test"}
                or p.parts[1] not in CLASSES or p.suffix.lower() != ".png"
                or any(part in {"..", "."} for part in p.parts)):
            raise ValueError(f"รูปแบบ ZIP ไม่ถูกต้อง: {p}")
    if len(paths) != 500 or len(set(paths)) != 500:
        raise ValueError("ZIP ต้องมี PNG ไม่ซ้ำ 500 ภาพ")
    if sum(m.file_size for m in members) > 300_000_000:
        raise ValueError("ZIP มีขนาดภาพหลังแตกเกินขีดจำกัดที่คาดไว้")
    for split, expected in (("train", 200), ("test", 50)):
        for cls in CLASSES:
            actual = sum(p.parts[:2] == (split, cls) for p in paths)
            if actual != expected:
                raise ValueError(f"{split}/{cls}: พบ {actual} ภาพ คาด {expected}")
    archive.extractall(DATA_ROOT)
print("วิธี:", METHOD, "| ภาพทั้งหมด:", len(paths), "| โฟลเดอร์:", DATA_ROOT)
""")

md("""
## 2. แบ่ง train/validation และสร้างตัวอ่านภาพ

ใช้เฉพาะ `train/` สำหรับฝึกและเลือก epoch ส่วน `test/` เก็บไว้ประเมินครั้งเดียวหลังฝึก
ภาพถูกเลือกไว้หนึ่งภาพต่อคนไข้ในขั้นเตรียมข้อมูล การแบ่งภายใน `train/` จึงไม่ทำให้คนไข้ซ้ำ
""")

code("""
rng = np.random.default_rng(SEED)
train_records, val_records, test_records = [], [], []
for label, cls in enumerate(CLASSES):
    candidates = sorted((DATA_ROOT / "train" / cls).glob("*.png"))
    if len(candidates) != 200:
        raise ValueError(f"train/{cls}: expected 200, got {len(candidates)}")
    perm = rng.permutation(len(candidates))
    val_records += [(str(candidates[i]), label) for i in perm[:40]]
    train_records += [(str(candidates[i]), label) for i in perm[40:]]
    tests = sorted((DATA_ROOT / "test" / cls).glob("*.png"))
    if len(tests) != 50:
        raise ValueError(f"test/{cls}: expected 50, got {len(tests)}")
    test_records += [(str(p), label) for p in tests]
random.Random(SEED).shuffle(train_records)
assert len({Path(p).name for p, _ in train_records + val_records + test_records}) == 500

def decode_image(path, label):
    image = tf.image.decode_png(tf.io.read_file(path), channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    image = tf.keras.applications.resnet50.preprocess_input(tf.cast(image, tf.float32))
    return image, tf.cast(label, tf.float32)

def make_dataset(records, training=False):
    paths, labels = zip(*records)
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(len(records), seed=SEED, reshuffle_each_iteration=True)
    return ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_records, training=True)
val_ds = make_dataset(val_records)
test_ds = make_dataset(test_records)
print("train:", len(train_records), "validation:", len(val_records), "test:", len(test_records))
""")

md("""
## 3. ฝึก ResNet50

ชั้น ResNet50 เริ่มจาก ImageNet และถูก freeze; ฝึกชั้นจำแนกด้านบนเท่านั้น
ทุกวิธีใช้สถาปัตยกรรม, seed, batch size, learning rate และเกณฑ์หยุดเหมือนกัน
ดาวน์โหลดน้ำหนัก ImageNet ครั้งแรกต้องมีอินเทอร์เน็ต
""")

code("""
tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(SEED)
backbone = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3), pooling="avg"
)
backbone.trainable = False
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = backbone(inputs, training=False)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="infiltration_probability")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
)
callbacks = [tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)]
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
""")

md("""
## 4. วัดผลบน official test และบันทึกผล

แสดง Accuracy, Precision, Recall, F1, AUC และ confusion matrix
ผลคะแนนดิบของทุกภาพอยู่ใน `test_predictions.csv` เพื่อกลับมาตรวจได้
""")

code("""
y_true = np.array([label for _, label in test_records], dtype=int)
y_prob = model.predict(test_ds, verbose=1).reshape(-1)
y_pred = (y_prob >= 0.5).astype(int)
matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
metrics = {
    "method": METHOD, "seed": SEED, "image_size": IMAGE_SIZE,
    "model": "ResNet50_ImageNet_frozen_backbone", "train_n": len(train_records),
    "val_n": len(val_records), "test_n": len(test_records),
    "epochs_run": len(history.history["loss"]), "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    "auc": float(roc_auc_score(y_true, y_prob)),
    "confusion_matrix_normal_infiltration": matrix.tolist(),
}
print(json.dumps(metrics, ensure_ascii=False, indent=2))
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(matrix, cmap="Blues")
ax.set_xticks([0, 1], CLASSES)
ax.set_yticks([0, 1], CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for (i, j), value in np.ndenumerate(matrix):
    ax.text(j, i, str(value), ha="center", va="center")
plt.tight_layout()
plt.show()
""")

code("""
RESULT_DIR = Path("/content/cxr_results") / METHOD
RESULT_DIR.mkdir(parents=True, exist_ok=True)
(RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
with (RESULT_DIR / "test_predictions.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "true_label", "prob_infiltration", "pred_label"])
    for (path, truth), probability, pred in zip(test_records, y_prob, y_pred):
        writer.writerow([Path(path).name, CLASSES[truth], float(probability), CLASSES[pred]])
with (RESULT_DIR / "split_manifest.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "image", "label"])
    for split, records in (("train", train_records), ("validation", val_records), ("test", test_records)):
        writer.writerows((split, Path(path).name, CLASSES[label]) for path, label in records)
with (RESULT_DIR / "history.csv").open("w", newline="") as f:
    writer = csv.writer(f)
    columns = list(history.history)
    writer.writerow(["epoch", *columns])
    for epoch in range(len(history.history["loss"])):
        writer.writerow([epoch + 1, *(history.history[key][epoch] for key in columns)])
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(matrix, cmap="Blues")
ax.set_xticks([0, 1], CLASSES)
ax.set_yticks([0, 1], CLASSES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for (i, j), value in np.ndenumerate(matrix):
    ax.text(j, i, str(value), ha="center", va="center")
fig.tight_layout()
fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)
model.save(RESULT_DIR / "model.keras")
print("บันทึกผลและโมเดลที่:", RESULT_DIR)
""")

md("""
## 5. ดาวน์โหลดผลก่อนปิด Colab

ZIP ผลลัพธ์ขนาดเล็กประกอบด้วยคะแนนและรายชื่อภาพที่ใช้
โมเดล `.keras` มีขนาดใหญ่ ให้ดาวน์โหลดแยกหากต้องการเก็บหรือใช้ทำนายต่อ
runtime ของ Colab เป็นชั่วคราว; ควรดาวน์โหลดผลก่อนปิด
""")

code("""
from google.colab import files
RESULT_ZIP = Path('/content') / f'{METHOD}_cnn_results.zip'
with zipfile.ZipFile(RESULT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in RESULT_DIR.iterdir():
        if path.name != 'model.keras':
            archive.write(path, path.name)
files.download(str(RESULT_ZIP))
print('หากต้องการเก็บโมเดลด้วย ให้รัน: files.download(str(RESULT_DIR / "model.keras"))')
""")

OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.validate(nb)
nbf.write(nb, OUT)
print(OUT)
