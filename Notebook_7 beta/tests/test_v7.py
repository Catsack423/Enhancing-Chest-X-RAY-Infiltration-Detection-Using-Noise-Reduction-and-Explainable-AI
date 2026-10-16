from pathlib import Path
from tempfile import TemporaryDirectory
import json
import zipfile
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v7.audit import estimate, audit
from v7.noise import STATES, corrupt, training_state
from v7.prepare import select_frames, validate_frames, prepare
from v7.report import image_quality
from v7.dataset import InputData, Preprocessor
from v7.runner import Run
from v7.smoke import fixture
from v7.colab import selected_bundle, stage_selected, safe_extract, ensure_source, mount_on_fresh_path
from v7.release.experiment import file_hash
from xai_interface import open_experiment


class NoiseTests(unittest.TestCase):
    def test_known_gaussian_and_invalid_patches(self):
        for sigma in (.01, .03, .05):
            image = np.rint(np.clip(.5 + np.random.default_rng(3).normal(0, sigma, (1024, 1024)), 0, 1)*255).astype(np.uint8)
            result = estimate(image)
            self.assertEqual(result["status"], "ok")
            self.assertAlmostEqual(result["noise_sigma"], sigma, delta=.12*sigma)
        for image in (np.zeros((256, 256), np.uint8), np.full((256, 256), 128, np.uint8), np.ones((64, 64), np.uint8)):
            self.assertEqual(estimate(image)["status"], "insufficient_patches")
        with self.assertRaises(ValueError):
            estimate(np.zeros((16, 16), np.uint8))

    def test_black_borders_do_not_define_noise(self):
        image = np.clip(np.random.default_rng(7).normal(128, 8, (1024, 1024)), 0, 255).astype(np.uint8)
        image[:100] = image[-100:] = 0
        image[:, :100] = image[:, -100:] = 0
        self.assertAlmostEqual(estimate(image)["noise_sigma"], 8/255, delta=.004)

    def test_corruption_seed_and_formulas(self):
        image = np.full((256, 256), 128, np.uint8)
        self.assertEqual(len(STATES), 16)
        np.testing.assert_array_equal(corrupt(image)[0], image)
        for state in STATES:
            a, stats = corrupt(image, state["id"], image_id="test")
            b, _ = corrupt(image, state["id"], image_id="test")
            np.testing.assert_array_equal(a, b)
            self.assertEqual(a.dtype, np.uint8)
            self.assertGreaterEqual(stats["clipping_fraction"], 0)
            if state["id"] != "clean":
                self.assertGreater(stats["noise_rmse"], 0)
                self.assertFalse(np.array_equal(a, corrupt(image, state["id"], image_id="other")[0]))
        expected_variance = {"gaussian_N1": .02**2, "poisson_N1": (128/255)/500,
            "poisson_gaussian_N1": (128/255)/500+.01**2, "speckle_N1": (128/255*.05)**2}
        for state, expected in expected_variance.items():
            actual = corrupt(image, state, image_id="formula")[1]["noise_std"]**2
            self.assertAlmostEqual(actual, expected, delta=expected*.06)
        noisy, _ = corrupt(image, "salt_pepper_N3", image_id="formula")
        self.assertAlmostEqual(np.mean(noisy != image), .05, delta=.003)
        self.assertEqual(training_state(42, "a", 2), training_state(42, "a", 2))

    def test_identity_psnr_and_ssim(self):
        image = np.arange(64*64, dtype=np.uint8).reshape(64, 64)
        self.assertTrue(np.isinf(image_quality(image, image)["psnr"]))
        self.assertEqual(image_quality(image, image)["ssim"], 1.)
        other = corrupt(image, "gaussian_N3", image_id="ssim")[0]
        self.assertLess(image_quality(image, other)["ssim"], 1.)


class ProtocolTests(unittest.TestCase):
    def test_mount_avoids_populated_old_mountpoint(self):
        with TemporaryDirectory() as temporary:
            content = Path(temporary)
            stale = content / "drive/MyDrive/nih -xray/Notebook_7_beta"
            stale.mkdir(parents=True)
            (stale / "keep.txt").write_text("preserve")
            def mount(destination):
                path = Path(destination)
                if list(path.iterdir()):
                    raise ValueError("Mountpoint must not already contain files")
                (path / "MyDrive/nih -xray/data/3").mkdir(parents=True)
            mounted = mount_on_fresh_path(mount, content)
            self.assertTrue((mounted / "MyDrive/nih -xray/data/3").is_dir())
            self.assertEqual((stale / "keep.txt").read_text(), "preserve")
            self.assertNotEqual(mounted, content / "drive")

    def test_mounted_drive_without_source_and_remount_recovery(self):
        with TemporaryDirectory() as temporary:
            drive = Path(temporary) / "MyDrive/nih -xray"
            (drive / "Notebook_7_beta").mkdir(parents=True)
            source = drive / "data/3"
            with self.assertRaisesRegex(FileNotFoundError, "Notebook_7_beta"):
                ensure_source(source)
            def remount():
                source.mkdir(parents=True)
                for name in ("Data_Entry_2017.csv", "train_val_list.txt", "test_list.txt"):
                    (source / name).write_text("fixture")
            self.assertEqual(ensure_source(source, remount=remount), source)

    def test_preparation_full_xai_package_and_frozen_selection(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = fixture(base / "fixture")
            root, nih = base / "code", base / "nih"
            (root / "configs").mkdir(parents=True)
            (root / "partition_reference").mkdir()
            images = nih / "images_001/images"
            images.mkdir(parents=True)
            config = json.loads((ROOT / "configs/protocol.json").read_text())
            config.update(expected_audit_images=135, counts_per_class={s: 2 for s in ("train", "validation", "test")})
            (root / "configs/protocol.json").write_text(json.dumps(config))
            frames = {s: pd.read_csv(source / f"{s}.csv") for s in ("train", "validation", "test")}
            template = frames["test"].query("label == 1").iloc[0]
            frames["xai"] = pd.DataFrame([{**template.to_dict(), "image": f"{100000+i:08d}_000.png",
                "patient_id": 100000+i, "finding_labels": "Infiltration|Effusion", "bbox_x": 40., "bbox_y": 40.,
                "bbox_w": 80., "bbox_h": 80., "risk_flag": "mixed_high_risk"} for i in range(123)])
            metadata = []
            for split, frame in frames.items():
                frame = frame.assign(relative_path="images_001/images/" + frame.image)
                # Match real frozen references: native dimensions/hashes are added by the audit.
                frame.drop(columns=["sha256", "width", "height"], errors="ignore").to_csv(root / f"partition_reference/{split}.csv", index=False)
                for row in frame.itertuples(index=False):
                    origin = source / "images" / (template.image if split == "xai" else row.image)
                    (images / row.image).write_bytes(origin.read_bytes())
                    metadata.append({"Image Index": row.image, "Patient ID": row.patient_id,
                        "Finding Labels": row.finding_labels, "View Position": "PA"})
            pd.DataFrame(metadata).to_csv(nih / "Data_Entry_2017.csv", index=False)
            (nih / "train_val_list.txt").write_text("\n".join(pd.concat([frames["train"], frames["validation"]]).image))
            (nih / "test_list.txt").write_text("\n".join(pd.concat([frames["test"], frames["xai"]]).image))
            audit(nih, base / "audit", root / "partition_reference", workers=2)
            work = base / "Drive/Notebook_7_beta"
            data = prepare(root, nih, base / "audit", work)
            self.assertEqual(len(pd.read_csv(data / "xai.csv")), 123)
            before = {s: file_hash(data / f"{s}.csv") for s in ("train", "validation", "test", "xai", "background")}
            prepare(root, nih, base / "audit", work)
            self.assertEqual(before, {s: file_hash(data / f"{s}.csv") for s in before})
            local = stage_selected(work, base / "VM")
            full_config = json.loads((ROOT / "configs/protocol.json").read_text())
            full_config["counts_per_class"] = config["counts_per_class"]
            verified = InputData(local, full_config)
            self.assertEqual(len(verified.frames["train"]), 4)
            self.assertEqual(len(verified.frames["xai"]), 123)

    def test_drive_package_roundtrip_and_integrity(self):
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            work = base / "Drive/Notebook_7_beta"
            work.mkdir(parents=True)
            source = fixture(base / "source")
            package = work / "v7_selected_data.zip"
            with zipfile.ZipFile(package, "w") as archive:
                for path in source.rglob("*"):
                    if path.is_file():
                        archive.write(path, "v7_data/" + path.relative_to(source).as_posix())
            receipt = {"sha256": file_hash(package), "dataset_receipt_sha256": file_hash(source / "dataset.json")}
            (work / "selected_package.json").write_text(json.dumps(receipt))
            local = stage_selected(work, base / "VM")
            self.assertEqual(file_hash(local / "dataset.json"), receipt["dataset_receipt_sha256"])
            self.assertEqual(stage_selected(work, base / "VM"), local)
            for path in source.rglob("*"):
                if path.is_file():
                    self.assertEqual(file_hash(path), file_hash(local / path.relative_to(source)))
            package.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "changed"):
                selected_bundle(work)
            with zipfile.ZipFile(base / "escape.zip", "w") as archive:
                archive.writestr("../escape", "bad")
            with self.assertRaises(ValueError):
                safe_extract(base / "escape.zip", base / "safe")

    def test_twelve_frozen_formulas_and_vendor_bytes(self):
        config = json.loads((ROOT / "configs/protocol.json").read_text())
        self.assertEqual(len(config["conditions"]), 12)
        self.assertFalse(any(c["id"].endswith("strength_L4") or c["id"] == "median_L4" for c in config["conditions"]))
        for name, record in json.loads((ROOT / "VENDOR_MANIFEST.json").read_text())["files"].items():
            self.assertEqual(file_hash(ROOT / "v7/release" / name), record["vendored_sha256"])
        for group, key, values in (("median", "kernel", [3, 5, 7]), ("clahe_dwt", "threshold_multiplier", [.5, 1, 2]),
                                   ("dae_clahe", "alpha", [.25, .5, 1])):
            self.assertEqual([c[key] for c in config["conditions"] if c["group"] == group], values)

    def test_checkpoint_provenance_and_multiview_shape(self):
        with TemporaryDirectory() as temporary:
            data = fixture(Path(temporary) / "data")
            run = Run(ROOT, data, Path(temporary) / "run", smoke=True)
            recorded = file_hash(run.output / "experiment.json")
            restored = open_experiment(ROOT, data, run.output)
            self.assertEqual(restored.signature, run.signature)
            self.assertEqual(file_hash(run.output / "experiment.json"), recorded)
            row = run.data.frames["train"].iloc[0]
            for condition in run.config["conditions"]:
                if condition["group"] == "dae_clahe":
                    continue
                prep = Preprocessor(run.data, condition, run.config)
                views = prep.tensor(row.image, "poisson_N1")
                expected = {"dwt_features": 5, "dft_features": 3}.get(condition["group"], 1)
                self.assertEqual(tuple(views.shape), (expected, 3, 64, 64))
            changed = json.loads((data / "dataset.json").read_text())
            changed["selection"] = "changed"
            (data / "dataset.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "changed"):
                Run(ROOT, data, Path(temporary) / "run", smoke=True)
            frames = run.data.frames.copy()
            frames["validation"] = frames["train"].copy()
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_frames(frames)

    def test_audit_resume_corruption_and_selection(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root / "source")
            dataset = root / "nih"
            folder = dataset / "images_001/images"
            folder.mkdir(parents=True)
            meta = []
            reference = root / "partition"
            reference.mkdir()
            frames = {s: pd.read_csv(source / f"{s}.csv") for s in ("train", "validation", "test", "xai")}
            for s, frame in frames.items():
                frame = frame.assign(relative_path="images_001/images/" + frame.image)
                frame.to_csv(reference / f"{s}.csv", index=False)
            for row in pd.concat([frames[s] for s in ("train", "validation", "test")]).itertuples(index=False):
                (folder / row.image).write_bytes((source / "images" / row.image).read_bytes())
                meta.append({"Image Index": row.image, "Patient ID": row.patient_id, "Finding Labels": row.finding_labels, "View Position": "PA"})
            pd.DataFrame(meta).to_csv(dataset / "Data_Entry_2017.csv", index=False)
            (dataset / "train_val_list.txt").write_text("\n".join(pd.concat([frames["train"], frames["validation"]]).image))
            (dataset / "test_list.txt").write_text("\n".join(frames["test"].image))
            with patch("v7.audit.image_index", return_value={}):
                with self.assertRaisesRegex(FileNotFoundError, "Image index found 0"):
                    audit(dataset, root / "empty_index", reference, workers=2)
            self.assertFalse((root / "empty_index/summary.json").exists())
            self.assertEqual(list((root / "empty_index/chunks").glob("*.json")), [])
            with patch("v7.audit.read_image", side_effect=OSError("Drive inaccessible")):
                with self.assertRaisesRegex(OSError, "All image reads failed"):
                    audit(dataset, root / "unreadable", reference, workers=2)
            broken = json.loads((root / "unreadable/summary.json").read_text())
            self.assertFalse(broken["complete"])
            self.assertEqual(broken["readable_images"], 0)
            broken["complete"] = True
            (root / "unreadable/summary.json").write_text(json.dumps(broken))
            recovered = audit(dataset, root / "unreadable", reference, workers=2, reuse_complete=True)
            self.assertTrue((recovered.status == "ok").all())
            result = audit(dataset, root / "audit", reference, workers=2)
            again = audit(dataset, root / "audit", reference, workers=2)
            pd.testing.assert_frame_equal(result, again)
            selected = select_frames(result, reference, {s: 1 for s in ("train", "validation", "test")})
            validate_frames(selected, {s: 1 for s in ("train", "validation", "test")})
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                select_frames(result, reference, {s: 3 for s in ("train", "validation", "test")})
            # A decoder failure is a recorded row, never a black replacement image.
            bad = folder / frames["train"].image.iloc[0]
            bad.write_bytes(b"not png")
            damaged = audit(dataset, root / "audit", reference, workers=2)
            self.assertEqual(damaged.set_index("image").loc[bad.name, "status"], "error")


if __name__ == "__main__":
    unittest.main()
