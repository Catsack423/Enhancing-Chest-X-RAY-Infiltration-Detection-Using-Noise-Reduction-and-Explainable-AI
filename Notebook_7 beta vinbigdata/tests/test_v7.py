from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys
import unittest

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v7.audit import estimate, audit, extract_binary_cohort
from v7.noise import STATES, corrupt
from v7.prepare import prepare
from v7.dataset import InputData
from v7.colab import stage_selected


class VinBigDataV7Tests(unittest.TestCase):
    def test_binary_cohort_normal_and_infiltration_with_comorbidities(self):
        # Verify:
        # 1) Normal ('No finding' only) -> label 0 ('normal')
        # 2) Infiltration alone OR Infiltration + other diseases -> label 1 ('infiltration')
        # 3) Other diseases without Infiltration -> excluded
        rows = [
            {"image_id": "img_normal_1", "class_name": "No finding", "class_id": 14},
            {"image_id": "img_normal_2", "class_name": "No finding", "class_id": 14},
            # Image with both Infiltration and Pleural effusion -> must be included as label 1 ('infiltration')
            {"image_id": "img_inf_mixed", "class_name": "Pleural effusion", "class_id": 10, "x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
            {"image_id": "img_inf_mixed", "class_name": "Infiltration", "class_id": 6, "x_min": 40, "y_min": 50, "x_max": 200, "y_max": 220},
            # Pure Infiltration
            {"image_id": "img_inf_pure", "class_name": "Infiltration", "class_id": 6, "x_min": 30, "y_min": 30, "x_max": 180, "y_max": 180},
            # Other abnormality without Infiltration -> must be excluded
            {"image_id": "img_other_only", "class_name": "Cardiomegaly", "class_id": 3, "x_min": 20, "y_min": 20, "x_max": 100, "y_max": 100},
        ]
        df = pd.DataFrame(rows)
        cohort = extract_binary_cohort(df, seed=42)
        by_id = {r["image_id"]: r for r in cohort}
        self.assertEqual(set(by_id.keys()), {"img_normal_1", "img_normal_2", "img_inf_mixed", "img_inf_pure"})
        self.assertEqual(by_id["img_normal_1"]["label"], 0)
        self.assertEqual(by_id["img_normal_1"]["finding_labels"], "normal")
        self.assertEqual(by_id["img_inf_mixed"]["label"], 1)
        self.assertEqual(by_id["img_inf_mixed"]["finding_labels"], "infiltration")
        self.assertEqual(by_id["img_inf_mixed"]["risk_flag"], "mixed_high_risk")
        self.assertEqual(by_id["img_inf_pure"]["label"], 1)
        self.assertEqual(by_id["img_inf_pure"]["risk_flag"], "pure")

    def test_end_to_end_vinbigdata_audit_prepare_and_stage(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            vin_root = base / "vinbigdata_root"
            train_dir = vin_root / "vinbigdata/train"
            train_dir.mkdir(parents=True)
            code_root = base / "code"
            (code_root / "configs").mkdir(parents=True)
            config = json.loads((ROOT / "configs/protocol.json").read_text())
            config["counts_per_class"] = {"train": 2, "validation": 1, "test": 1}
            (code_root / "configs/protocol.json").write_text(json.dumps(config))

            rng = np.random.default_rng(99)
            csv_rows = []
            # Create 10 normal and 10 infiltration images (some with co-occurring findings)
            for label in (0, 1):
                for idx in range(10):
                    img_id = f"vin_{label}_{idx:03d}"
                    img = np.clip(
                        120 + 25 * np.sin(np.linspace(0, 8, 512))[None, :]
                        + rng.normal(0, 4 + idx, (512, 512)),
                        0, 255
                    ).astype(np.uint8)
                    cv2.imwrite(str(train_dir / f"{img_id}.png"), img)
                    if label == 0:
                        csv_rows.append({"image_id": img_id, "class_name": "No finding", "class_id": 14})
                    else:
                        csv_rows.append({
                            "image_id": img_id, "class_name": "Infiltration", "class_id": 6,
                            "x_min": 50, "y_min": 60, "x_max": 300, "y_max": 320,
                        })
                        if idx % 2 == 0:
                            csv_rows.append({
                                "image_id": img_id, "class_name": "Pulmonary fibrosis", "class_id": 13,
                                "x_min": 80, "y_min": 80, "x_max": 200, "y_max": 200,
                            })
            pd.DataFrame(csv_rows).to_csv(vin_root / "vinbigdata/train.csv", index=False)

            audit_dir = base / "audit"
            work_dir = base / "work"
            frame = audit(vin_root, audit_dir, workers=2)
            self.assertEqual(len(frame), 20)
            self.assertTrue((frame.status == "ok").all())

            selected = prepare(code_root, vin_root, audit_dir, work_dir)
            self.assertTrue((work_dir / "v7_selected_data.zip").is_file())
            staged = stage_selected(work_dir, base / "cache")
            data = InputData(staged, config)
            self.assertEqual(len(data.frames["train"]), 4)
            self.assertEqual(len(data.frames["validation"]), 2)
            self.assertEqual(len(data.frames["test"]), 2)
            self.assertEqual(set(data.frames["train"].finding_labels), {"normal", "infiltration"})


if __name__ == "__main__":
    unittest.main()
