"""Small synthetic data: no real training, Drive access or downloaded weights."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import json
import shutil
import sys
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.data import SOURCE_NAMES, prepare_manifests, verify_manifests, resolve_image_path
from shared.denoising import dae_blend, working_image, dwt_denoise, apply_condition, load_dae
from shared.experiment import file_hash
from shared.frequency import dwt_components, dft_components
from shared.preprocessing import ReleasePreprocessor
from shared.runtime_cache import RuntimeCache


class DataCacheTests(TestCase):
    def setUp(self):
        sandbox_temp = Path(__file__).resolve().parents[1]/".runtime/test_tmp"
        sandbox_temp.mkdir(parents=True, exist_ok=True)
        self.temp = SimpleNamespace(name=str(sandbox_temp/("data_cache_"+uuid4().hex)))
        Path(self.temp.name).mkdir()
        if not Path(self.temp.name).resolve().is_relative_to(sandbox_temp):
            raise ValueError("Test temporary directory escaped runtime workspace")
        base = Path(self.temp.name)
        root, data = base/"release", base/"dataset"
        (root/"manifests").mkdir(parents=True)
        (data/"images_001"/"images").mkdir(parents=True)
        records, self.frames = [], {}
        for split, first in (("train", 0), ("validation", 4), ("test", 8)):
            rows = []
            for i in range(first, first+4):
                image = f"{i:08}_000.png"
                label = i % 2
                finding = "Infiltration" if label else "No Finding"
                row = {"image": image, "patient_id": i, "relative_path": f"images_001/images/{image}",
                       "label": label, "finding_labels": finding}
                rows.append(row)
                records.append({"Image Index": image, "Patient ID": i, "Finding Labels": finding})
                raw = (np.arange(90*80).reshape(90, 80)+i*7).astype(np.uint8)
                cv2.imwrite(str(data/row["relative_path"]), raw)
            self.frames[split] = pd.DataFrame(rows)
        self.frames["background"] = self.frames["train"].iloc[:2].copy()
        xai = self.frames["test"].loc[self.frames["test"].label == 1].copy()
        for key, value in {"bbox_x": 3., "bbox_y": 4., "bbox_w": 30., "bbox_h": 20., "risk_flag": "pure"}.items():
            xai[key] = value
        self.frames["xai"] = xai
        pd.DataFrame(records).to_csv(data/"Data_Entry_2017.csv", index=False)
        pd.DataFrame({"image": xai.image, "finding": "Infiltrate", "x": 3., "y": 4., "w": 30., "h": 20.}).to_csv(data/"BBox_List_2017.csv", index=False)
        (data/"train_val_list.txt").write_text("\n".join(pd.concat([self.frames["train"], self.frames["validation"]]).image))
        (data/"test_list.txt").write_text("\n".join(self.frames["test"].image))
        for name, frame in self.frames.items():
            frame.to_csv(root/"manifests"/(name+".csv"), index=False)
        config = {"run_mode": "smoke", "profile": "release_controlled", "data": {
                    "source_hashes": {name: file_hash(data/name) for name in SOURCE_NAMES},
                    "expected_counts": {"train": 4, "validation": 4, "test": 4, "xai": 2, "background": 2}},
                  "preprocessing": {"working_size": 64, "cnn_size": 32, "mean": [.485, .456, .406],
                                    "std": [.229, .224, .225], "clahe_grid": [8, 8]},
                  "runtime_cache": {"shard_size": 4, "derived_budget_bytes": 1024**2, "read_retries": 1}}
        self.conditions = {"baseline": {"id": "baseline", "group": "baseline"},
                           "median": {"id": "median", "group": "median", "kernel": 3},
                           "dwt": {"id": "dwt", "group": "dwt_features"},
                           "dft": {"id": "dft", "group": "dft_features"}}
        self.exp = SimpleNamespace(root=root, data_root=data, artifacts=base/"outputs/artifacts",
                   config=config, cache_root=base/"runtime", cache=None,
                   condition=lambda name: self.conditions[name], lock=lambda: None)

    def tearDown(self):
        self.exp.cache = None
        path = Path(self.temp.name).resolve()
        expected = Path(__file__).resolve().parents[1]/".runtime/test_tmp"
        if not path.is_relative_to(expected):
            raise ValueError("Unsafe test cleanup")
        shutil.rmtree(path)

    def test_smoke_uses_frozen_partitions_and_background(self):
        frames = prepare_manifests(self.exp)
        self.assertEqual({k: len(v) for k, v in frames.items()}, self.exp.config["data"]["expected_counts"])
        self.assertLessEqual(set(frames["background"].image), set(frames["train"].image))
        for name, frame in frames.items():
            self.assertLessEqual(set(frame.image), set(self.frames[name].image))
        prepare_manifests(self.exp)  # compatible resume
        frames["train"].loc[frames["train"].index[0], "relative_path"] = "../leak.png"
        with self.assertRaises(ValueError):
            verify_manifests(self.exp, frames)

    def test_rejects_source_label_and_partition_changes(self):
        frames = {name: f.copy() for name, f in self.frames.items()}
        frames["train"].iloc[0] = frames["validation"].iloc[0]
        with self.assertRaisesRegex(ValueError, "leakage|partition"):
            verify_manifests(self.exp, frames)
        with (self.exp.data_root/"test_list.txt").open("a") as stream:
            stream.write("\nchanged.png")
        with self.assertRaisesRegex(ValueError, "checksums"):
            verify_manifests(self.exp, self.frames)

    def test_cache_missing_image_never_substitutes(self):
        row = self.frames["train"].iloc[0]
        (self.exp.data_root/row.relative_path).unlink()
        with self.assertRaisesRegex(FileNotFoundError, row.image):
            RuntimeCache(self.exp).prepare(self.frames)
        self.assertFalse(list(self.exp.cache_root.glob("source/*/complete.json")))

    def test_cache_resumes_partial_and_corrupt_shards(self):
        cache = RuntimeCache(self.exp).prepare(self.frames)
        row = self.frames["train"].iloc[0]
        expected = working_image(cv2.imread(str(self.exp.data_root/row.relative_path), 0), self.exp.config)
        np.testing.assert_array_equal(cache.working(row), expected)
        cache._maps.clear()
        first = cache.shards[0]
        first.write_bytes(b"corrupted")
        cache.shards[1].with_suffix(".json").unlink()
        with patch.object(RuntimeCache, "_read", wraps=cache._read) as reads:
            resumed = RuntimeCache(self.exp).prepare(self.frames)
            self.assertEqual(reads.call_count, 8)
        np.testing.assert_array_equal(resumed.working(row), expected)
        xrow = self.frames["xai"].iloc[0]
        self.assertEqual(resumed.native(xrow).shape, (90, 80))
        resumed._maps.clear()
        native = resumed._native_path(xrow.image)
        native.write_bytes(b"broken native")
        resumed = RuntimeCache(self.exp).prepare(self.frames)
        self.assertEqual(resumed.native(xrow).shape, (90, 80))
        with self.assertRaises(ValueError):
            resumed.native(row)

    def test_direct_folder_layout_and_resize_key(self):
        image_dir = self.exp.data_root/"images_001/images"
        for path in image_dir.iterdir():
            path.replace(image_dir.parent/path.name)
        image_dir.rmdir()
        self.assertTrue(resolve_image_path(self.exp, self.frames["train"].iloc[0]).exists())
        cache = RuntimeCache(self.exp).prepare(self.frames)
        key = cache.key
        self.exp.config["preprocessing"]["working_size"] = 80
        self.exp.artifacts = self.exp.artifacts.parent/"resized_output/artifacts"
        resized = RuntimeCache(self.exp).prepare(self.frames)
        self.assertNotEqual(key, resized.key)
        self.assertEqual(resized.working(self.frames["train"].iloc[0]).shape, (80, 80))

    def test_signed_views_reconstruct_and_tensor_shapes(self):
        rng = np.random.default_rng(5)
        image = rng.random((64, 64), dtype=np.float32)
        for components, count in ((dwt_components, 5), (dft_components, 3)):
            bands = components(image)
            self.assertEqual(bands.shape, (count, 64, 64))
            self.assertLess(bands[1:].min(), 0)
            np.testing.assert_allclose(bands.sum(0), image, atol=5e-7)
        RuntimeCache(self.exp).prepare(self.frames)
        row = self.frames["train"].iloc[0]
        for condition, views in (("baseline", 1), ("median", 1), ("dwt", 5), ("dft", 3)):
            prep = ReleasePreprocessor(self.exp, condition).prepare(self.frames)
            tensor = prep.tensor(row)
            self.assertEqual(tuple(tensor.shape), (views, 3, 32, 32))
            self.assertTrue(torch.isfinite(tensor).all())
        self.exp.cache._maps.clear()

    def test_dae_alpha_and_threshold_zero_preserve_identity(self):
        image = np.arange(64*64).reshape(64, 64).astype(np.uint8)
        class ConstantDAE(torch.nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)
        np.testing.assert_array_equal(dae_blend(image, ConstantDAE(), 0), image)
        np.testing.assert_array_equal(dae_blend(image, ConstantDAE(), 1), np.zeros_like(image))
        np.testing.assert_array_equal(dae_blend(image, ConstantDAE(), .5), np.round(image.astype(float)*.5).astype(np.uint8))
        np.testing.assert_array_equal(dwt_denoise(image, {"threshold_multiplier": 0}), image)
        with self.assertRaises(ValueError):
            dae_blend(image, ConstantDAE(), 2)

    def test_l4_dae_passes_and_seminae_dwt_ratio(self):
        image = np.arange(64*64).reshape(64, 64).astype(np.uint8)
        class HalfDAE(torch.nn.Module):
            def forward(self, x):
                return x*.5
        twice = np.round(image.astype(np.float32)/255*.25*255).astype(np.uint8)
        np.testing.assert_array_equal(dae_blend(image, HalfDAE(), 1, passes=2), twice)
        np.testing.assert_array_equal(dae_blend(image, HalfDAE(), 1), dae_blend(image, HalfDAE(), 1, passes=1))
        with self.assertRaises(ValueError):
            dae_blend(image, HalfDAE(), 1, passes=0)
        import pywt
        condition = {"threshold_policy": "max_detail_ratio", "threshold_ratio": .35, "dwt_depth": 1, "wavelet": "db1"}
        rng = np.random.default_rng(3)
        noisy = rng.integers(0, 256, (64, 64)).astype(np.uint8)
        # Seminae used float32; float64 here differs by at most one grey level after rounding.
        LL, (LH, HL, HH) = pywt.dwt2(noisy.astype(np.float64), "db1")
        cutoff = .35*np.max([np.abs(LH), np.abs(HL), np.abs(HH)])
        bands = tuple(pywt.threshold(c, cutoff, mode="soft") for c in (LH, HL, HH))
        seminae = np.clip(np.round(pywt.idwt2((LL, bands), "db1")), 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(dwt_denoise(noisy, condition), seminae)
        np.testing.assert_array_equal(dwt_denoise(noisy, {**condition, "threshold_ratio": 0}), noisy)

    def test_derived_cache_checks_integrity_and_budget(self):
        RuntimeCache(self.exp).prepare(self.frames)
        prep = ReleasePreprocessor(self.exp, "median").prepare(self.frames)
        row = self.frames["train"].iloc[0]
        expected = cv2.medianBlur(self.exp.cache.working(row), 3)
        np.testing.assert_array_equal(prep.image(row), expected)
        prep._shards.clear()
        path = prep.directory/"shard_00000.npy"
        path.write_bytes(b"corrupt derived payload")
        resumed = ReleasePreprocessor(self.exp, "median").prepare(self.frames)
        np.testing.assert_array_equal(resumed.image(row), expected)
        resumed._shards.clear()
        self.exp.config["runtime_cache"]["derived_budget_bytes"] = 50
        with self.assertRaisesRegex(OSError, "exceeding"):
            ReleasePreprocessor(self.exp, "median").prepare(self.frames)
        # Dataset originals are intact; only runtime intermediates were touched.
        self.assertTrue((self.exp.data_root/row.relative_path).is_file())
        self.exp.cache._maps.clear()

    def test_new_vm_changed_pixels_cannot_resume_same_output(self):
        cache = RuntimeCache(self.exp).prepare(self.frames)
        fingerprint = json.loads((self.exp.artifacts/"input_source.json").read_text())
        self.assertEqual(fingerprint["source_images"], 12)
        row = self.frames["train"].iloc[0]
        cv2.imwrite(str(self.exp.data_root/row.relative_path), np.full((90, 80), 17, dtype=np.uint8))
        self.exp.cache_root = self.exp.cache_root.parent/"new_vm_runtime"
        fresh = RuntimeCache(self.exp)
        with self.assertRaisesRegex(ValueError, "pixel provenance changed"):
            fresh.prepare(self.frames)
        self.assertFalse(fresh.ready)
        self.assertEqual(json.loads((self.exp.artifacts/"input_source.json").read_text()), fingerprint)
        cache._maps.clear()

    def test_native_repair_rejects_changed_source_pixels(self):
        cache = RuntimeCache(self.exp).prepare(self.frames)
        row = self.frames["xai"].iloc[0]
        cache._native_path(row.image).write_bytes(b"corrupt native cache")
        cv2.imwrite(str(self.exp.data_root/row.relative_path), np.full((90, 80), 29, dtype=np.uint8))
        repaired = RuntimeCache(self.exp)
        with self.assertRaisesRegex(ValueError, "pixels changed while repairing"):
            repaired.prepare(self.frames)
        self.assertFalse(repaired.ready)
        cache._maps.clear()

    def test_partial_dae_best_without_completion_is_rejected(self):
        folder = self.exp.artifacts/"dae_checkpoint"
        folder.mkdir(parents=True)
        torch.save({"epoch": 1}, folder/"best.pt")
        with self.assertRaisesRegex(RuntimeError, "training is incomplete"):
            load_dae(self.exp, "cpu")

    def test_standalone_legacy_helper_uses_original_threshold_policy(self):
        image = np.random.default_rng(7).integers(0, 256, (64, 64), dtype=np.uint8)
        condition = {"group": "clahe_dwt", "wavelet": "db1", "dwt_depth": 2,
                     "threshold_multiplier": .5, "clip_limit": 4, "order": "DWT_then_CLAHE"}
        self.exp.config["profile"] = "legacy_v2"
        expected = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8)).apply(
                   dwt_denoise(image, {**condition, "profile": "legacy_v2"}))
        np.testing.assert_array_equal(apply_condition(image, condition, self.exp.config), expected)
