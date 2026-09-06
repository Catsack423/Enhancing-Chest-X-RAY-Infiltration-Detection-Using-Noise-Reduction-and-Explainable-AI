"""
Phase 1: Data Preparation & Verification
NIH ChestX-ray14 — Binary Classification: Infiltration vs Normal (No Finding only)

Steps:
  1. Parse Finding Labels (multi-label, '|' separated)
  2. Categorize: Infiltration-only / No-Finding-only / Excluded
  3. Join with official train_val_list.txt / test_list.txt (patient-level split)
  4. Join with BBox_List_2017.csv for XAI evaluation subset
  5. Build full image path mapping
  6. Verify constraints (patient leakage, bbox in test, class overlap)
  7. Output manifest CSVs + summary report
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = r"data/versions/3"
OUT_DIR   = r"data/manifests"

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load raw data ───────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: DATA PREPARATION & VERIFICATION")
print("=" * 60)

# Data_Entry: skip bad column parsing — read raw then rename
raw_cols = [
    "Image Index", "Finding Labels", "Follow-up #", "Patient ID",
    "Patient Age", "Patient Gender", "View Position",
    "OriginalImageWidth", "OriginalImageHeight",
    "OriginalImagePixelSpacingX", "OriginalImagePixelSpacingY",
    "Unnamed"
]
de = pd.read_csv(
    os.path.join(DATA_ROOT, "Data_Entry_2017.csv"),
    header=0, names=raw_cols, usecols=range(12)
)
# Convert Patient ID to int
de["Patient ID"] = de["Patient ID"].astype(int)

# BBox
bb = pd.read_csv(os.path.join(DATA_ROOT, "BBox_List_2017.csv"))
# Rename messy bbox columns
bb.rename(columns={
    "Bbox [x": "Bbox_x",
    "y": "Bbox_y",
    "w": "Bbox_w",
    "h]": "Bbox_h"
}, inplace=True)

# Splits
tv = pd.read_csv(os.path.join(DATA_ROOT, "train_val_list.txt"), header=None, names=["Image Index"])
te = pd.read_csv(os.path.join(DATA_ROOT, "test_list.txt"), header=None, names=["Image Index"])

# ── 2. Build image path map ────────────────────────────────────────────
# Structure: data/versions/3/images_NNN/images/<filename>.png
img_path_map = {}
for d in sorted(os.listdir(DATA_ROOT)):
    if not d.startswith("images_"):
        continue
    img_dir = os.path.join(DATA_ROOT, d, "images")
    if not os.path.isdir(img_dir):
        continue
    for fname in os.listdir(img_dir):
        if fname.endswith(".png"):
            img_path_map[fname] = os.path.abspath(os.path.join(img_dir, fname))

print(f"\n[1] Image path map built: {len(img_path_map)} files")
print(f"    (Total on disk matches Data_Entry: {len(img_path_map) == len(de)})")

# ── 3. Categorize images by Finding Labels ─────────────────────────────
def categorize(label_str):
    """Categorize a Finding Labels string into one of three groups."""
    labels = [lbl.strip() for lbl in str(label_str).split("|")]
    if len(labels) == 1 and labels[0] == "No Finding":
        return "No-Finding-only"
    if "Infiltration" in labels and len(labels) == 1:
        return "Infiltration-only"
    if "Infiltration" in labels and len(labels) > 1:
        return "Excluded (Infiltration + others)"
    # Has other diseases but NOT Infiltration
    return "Excluded (other diseases)"


def risk_flag(label_str):
    """Assign XAI evaluation risk flag from Finding Labels."""
    labels = {lbl.strip() for lbl in str(label_str).split("|")}
    if labels == {"Infiltration"}:
        return "pure"
    if labels & {"Effusion", "Atelectasis"}:
        return "mixed_high_risk"
    return "mixed_low_risk"

de["Category"] = de["Finding Labels"].apply(categorize)

cat_counts = de["Category"].value_counts()
print(f"\n[2] Category counts (raw, unfiltered by split):")
for cat, cnt in cat_counts.items():
    print(f"    {cat}: {cnt}")

# ── 4. Tag split membership & build filtered dataset ───────────────────
de["InSplit"] = "none"
de.loc[de["Image Index"].isin(tv["Image Index"]), "InSplit"] = "train_val"
de.loc[de["Image Index"].isin(te["Image Index"]), "InSplit"] = "test"

# Filter to only usable categories for training
usable = de[de["Category"].isin(["Infiltration-only", "No-Finding-only"])].copy()

print(f"\n[3] Usable images after category filter: {len(usable)}")
print(f"    (Excluded: {len(de) - len(usable)})")

# ── 5. Split into train/val vs test ────────────────────────────────────
train_usable = usable[usable["InSplit"] == "train_val"]
test_usable  = usable[usable["InSplit"] == "test"]

print(f"\n[4] By split (usable only):")
print(f"    train_val: {len(train_usable)}")
print(f"    test:      {len(test_usable)}")

train_inf = train_usable[train_usable["Category"] == "Infiltration-only"].copy()
train_nrm = train_usable[train_usable["Category"] == "No-Finding-only"].copy()
test_inf  = test_usable[test_usable["Category"] == "Infiltration-only"].copy()
test_nrm  = test_usable[test_usable["Category"] == "No-Finding-only"].copy()

# ── 6. XAI Evaluation Set (BBox Infiltrate) ────────────────────────────
# Note: BBox uses "Infiltrate" label while Data_Entry uses "Infiltration"
bbox_infiltrate = bb[bb["Finding Label"] == "Infiltrate"].copy()
# Join with Data_Entry to get Patient ID and metadata
xai_meta = bbox_infiltrate.merge(de[["Image Index", "Patient ID", "Finding Labels"]], on="Image Index", how="left")
# Verify all found
xai_meta["path"] = xai_meta["Image Index"].map(img_path_map)
xai_meta["in_test"] = xai_meta["Image Index"].isin(te["Image Index"])
xai_meta["in_train"] = xai_meta["Image Index"].isin(tv["Image Index"])

print(f"\n[5] XAI Evaluation Set (BBox Infiltrate):")
print(f"    Total BBox Infiltrate entries: {len(bbox_infiltrate)}")
print(f"    Joining with Data_Entry (unique Image Index): {len(xai_meta)}")
print(f"    In test_list.txt: {xai_meta['in_test'].sum()}")
print(f"    In train_val_list.txt: {xai_meta['in_train'].sum()}")
print(f"    File found on disk: {xai_meta['path'].notna().sum()}")

# ── 7. VERIFICATIONS ───────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("VERIFICATION RESULTS")
print("=" * 60)

errors = []

# 7a. All 123 bbox images in test_list.txt
all_bbox_in_test = xai_meta["in_test"].all()
print(f"\n[V1] All 123 BBox Infiltrate in test_list.txt: "
      f"{'PASS' if all_bbox_in_test else 'FAIL'}")
if not all_bbox_in_test:
    missing_from_test = xai_meta[~xai_meta["in_test"]]
    print(f"      Images NOT in test: {missing_from_test['Image Index'].tolist()}")
    errors.append("Some BBox Infiltrate images not in test_list.txt")

# 7b. No patient overlap between train/val and test
tv_patients = set(de[de["Image Index"].isin(tv["Image Index"])]["Patient ID"])
te_patients = set(de[de["Image Index"].isin(te["Image Index"])]["Patient ID"])
patient_leak = len(tv_patients & te_patients)
print(f"\n[V2] Patient overlap train_val vs test: "
      f"{'PASS (0)' if patient_leak == 0 else f'FAIL ({patient_leak})'}")
if patient_leak > 0:
    errors.append(f"Patient leakage: {patient_leak} patients in both splits")

# 7c. No image overlap between Infiltration-only and No-Finding-only
inf_images = set(de[de["Category"] == "Infiltration-only"]["Image Index"])
nrm_images = set(de[de["Category"] == "No-Finding-only"]["Image Index"])
overlap_cat = len(inf_images & nrm_images)
print(f"\n[V3] Image overlap Infiltration-only vs No-Finding-only: "
      f"{'PASS (0)' if overlap_cat == 0 else f'FAIL ({overlap_cat})'}")
if overlap_cat > 0:
    errors.append(f"Category overlap: {overlap_cat} images in both Infiltration-only and No-Finding-only")

# 7d. All image paths exist
missing_paths = []
for df, name in [(train_inf, "train_inf"), (train_nrm, "train_nrm"),
                 (test_inf, "test_inf"), (test_nrm, "test_nrm")]:
    for _, row in df.iterrows():
        p = img_path_map.get(row["Image Index"])
        if p is None or not os.path.exists(p):
            missing_paths.append((name, row["Image Index"]))
print(f"\n[V4] All image files exist: "
      f"{'PASS' if len(missing_paths) == 0 else f'FAIL ({len(missing_paths)} missing)'}")
if missing_paths:
    errors.append(f"{len(missing_paths)} image files missing from disk")
    for name, img in missing_paths[:10]:
        print(f"      {name}: {img}")

# 7e. BBox label mapping correct
# "Infiltrate" from BBox should correspond to "Infiltration" in Data_Entry
wrong_labels = xai_meta[~xai_meta["Finding Labels"].str.contains("Infiltration", na=False)]
print(f"\n[V5] BBox 'Infiltrate' matches Data_Entry 'Infiltration': "
      f"{'PASS' if len(wrong_labels) == 0 else f'WARN ({len(wrong_labels)} mismatches)'}")
if len(wrong_labels) > 0:
    print(f"      Mismatches: {wrong_labels['Image Index'].tolist()}")

# ── 8. OUTPUT manifest CSVs ────────────────────────────────────────────
def write_manifest(df, name, label_val):
    """Write manifest CSV with Image Index, full path, label, split info."""
    out = df.copy()
    out["path"] = out["Image Index"].map(img_path_map)
    out["label"] = label_val
    out["split"] = name.split("_")[0]  # train or test
    out = out[["Image Index", "path", "label", "Patient ID", "split"]]
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    out.to_csv(out_path, index=False)
    print(f"    Wrote {out_path} ({len(out)} rows)")
    return out

print(f"\n{'=' * 60}")
print("MANIFEST OUTPUT")
print("=" * 60)

train_inf_out = write_manifest(train_inf, "train_infiltration", 1)
train_nrm_out = write_manifest(train_nrm, "train_normal", 0)
test_inf_out  = write_manifest(test_inf, "test_infiltration", 1)
test_nrm_out  = write_manifest(test_nrm, "test_normal", 0)

# XAI eval set
xai_out = xai_meta.copy()
xai_out["label"] = 1
xai_out = xai_out.rename(columns={"Bbox_x": "bbox_x", "Bbox_y": "bbox_y",
                                   "Bbox_w": "bbox_w", "Bbox_h": "bbox_h"})
xai_out["risk_flag"] = xai_out["Finding Labels"].apply(risk_flag)
xai_out = xai_out[["Image Index", "path", "label", "Patient ID",
                    "Finding Labels", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                    "risk_flag"]]
xai_out_path = os.path.join(OUT_DIR, "xai_eval_set.csv")
xai_out.to_csv(xai_out_path, index=False)
print(f"    Wrote {xai_out_path} ({len(xai_out)} rows)")

# Risk flag distribution
rf_counts = xai_out["risk_flag"].value_counts()
print(f"\n[6] XAI eval risk_flag distribution:")
for flag in ["pure", "mixed_high_risk", "mixed_low_risk"]:
    print(f"    {flag}: {rf_counts.get(flag, 0)}")

# ── 9. SUMMARY ─────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("PHASE 1 SUMMARY")
print("=" * 60)

total_usable = len(train_inf) + len(train_nrm) + len(test_inf) + len(test_nrm)
print(f"Total usable images: {total_usable}")
print(f"  Train Infiltration:    {len(train_inf)}")
print(f"  Train Normal:          {len(train_nrm)}")
print(f"  Test Infiltration:     {len(test_inf)}")
print(f"  Test Normal:           {len(test_nrm)}")
print(f"  XAI Eval Set (bbox):   {len(xai_out)}")

# Class imbalance ratio (train only for training; full dataset also)
if len(train_nrm) > 0:
    train_imbalance = len(train_inf) / len(train_nrm)
else:
    train_imbalance = float('inf')
if len(test_nrm) > 0:
    test_imbalance = len(test_inf) / len(test_nrm)
else:
    test_imbalance = float('inf')

print(f"\nClass imbalance ratio (Infiltration : Normal):")
print(f"  Train: {len(train_inf)}:{len(train_nrm)} = {train_imbalance:.4f}")
print(f"  Test:  {len(test_inf)}:{len(test_nrm)} = {test_imbalance:.4f}")

# Excluded breakdown
excluded = de[~de["Category"].isin(["Infiltration-only", "No-Finding-only"])]
print(f"\nExcluded images: {len(excluded)}")
excl_cats = excluded["Category"].value_counts()
for cat, cnt in excl_cats.items():
    print(f"  {cat}: {cnt}")

if errors:
    print(f"\n{'!' * 60}")
    print(f"FLAGS / ISSUES FOUND ({len(errors)}):")
    for i, err in enumerate(errors, 1):
        print(f"  [{i}] {err}")
    print(f"{'!' * 60}")
else:
    print(f"\n{'=' * 60}")
    print("ALL VERIFICATIONS PASSED - Data ready for Phase 2")
    print(f"{'=' * 60}")

print(f"\nAll manifests saved to: {OUT_DIR}")
print("Done.")