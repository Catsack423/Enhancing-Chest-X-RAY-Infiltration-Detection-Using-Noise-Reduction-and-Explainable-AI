"""Speed-only patches in tools/parallel_session.py must not change batches and must restore."""
from pathlib import Path
import runpy
import sys
import unittest

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import shared.training as training
from shared.preprocessing import ReleasePreprocessor

parallel = runpy.run_path(str(ROOT / "tools/parallel_session.py"))


class Prep:
    def tensor(self, row):
        return torch.full((1, 3, 8, 8), float(row["label"]) + int(row["image"][1:]) / 10)


class SpeedupTests(unittest.TestCase):
    def test_cnn_loader_workers_raised_and_batches_unchanged(self):
        frame = pd.DataFrame({"image": [f"i{i}" for i in range(10)], "label": [i % 2 for i in range(10)]})
        dataset = training.ImageDataset(frame, Prep())
        plain = list(training.loader(dataset, 3, 7))
        prepare = ReleasePreprocessor.prepare
        with parallel["runtime_speedups"]("cnn", workers=2) as info:
            patched = training.loader(dataset, 3, 7)
            self.assertEqual(patched.num_workers, 2)
            fast = list(patched)
            # Non-CNN datasets keep their configured workers.
            other = training.loader(torch.utils.data.TensorDataset(torch.zeros(4, 1)), 2, 7)
            self.assertEqual(other.num_workers, 0)
            self.assertIs(ReleasePreprocessor.prepare, prepare)  # CNN training keeps the derived cache
        self.assertEqual(info, {"cnn_loader_workers": 2, "xai_without_derived_cache": False})
        for (x0, y0), (x1, y1) in zip(plain, fast):
            self.assertTrue(torch.equal(x0, x1) and torch.equal(y0, y1))
        self.assertEqual(training.loader.__module__, "shared.training")

    def test_xai_skips_derived_cache_and_restores_prepare(self):
        original = ReleasePreprocessor.prepare
        with parallel["runtime_speedups"]("shap", workers=2) as info:
            self.assertIsNot(ReleasePreprocessor.prepare, original)
            prep = ReleasePreprocessor.__new__(ReleasePreprocessor)
            prep._shards, prep.directory = {"x": 1}, "old"
            prep.exp = type("E", (), {"cache": type("C", (), {"ready": True})()})()
            self.assertIs(ReleasePreprocessor.prepare(prep, {}), prep)
            self.assertIsNone(prep.directory)
            self.assertEqual(prep._shards, {})
        self.assertTrue(info["xai_without_derived_cache"])
        self.assertIs(ReleasePreprocessor.prepare, original)


if __name__ == "__main__":
    unittest.main()
