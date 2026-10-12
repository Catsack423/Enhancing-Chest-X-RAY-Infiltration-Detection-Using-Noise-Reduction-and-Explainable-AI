"""Notebook observation preserves results/hooks and reports blocked reads."""
import contextlib
import io
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from shared import data
from shared.runtime_cache import RuntimeCache


class PrepareProgressTests(unittest.TestCase):
    def context(self):
        namespace = {}
        source = (ROOT / "prepare_progress_cell.txt").read_text(encoding="utf-8")
        exec(source.split("\nruns = {}\n")[0], namespace)
        return namespace["preparation_progress"]

    def test_heartbeat_reports_current_image_and_restores_hooks(self):
        frames = {"train": pd.DataFrame({"image": ["a.png", "b.png"]})}
        cache = RuntimeCache.__new__(RuntimeCache)
        cache.root = Path("runtime-cache")
        cache.indices = {"a.png": 0, "b.png": 1}
        result = object()
        def read(obj, row):
            time.sleep(.06)
            return result
        def prepare(obj, current):
            self.assertIs(obj._read({"image": "a.png"}), result)
            return obj
        output = io.StringIO()
        with patch.object(RuntimeCache, "prepare", prepare), patch.object(RuntimeCache, "_read", read):
            original_manifest, original_verify = data.prepare_manifests, data.verify_manifests
            with contextlib.redirect_stdout(output):
                with self.context()(42, interval=.01):
                    self.assertIs(cache.prepare(frames), cache)
            self.assertIs(RuntimeCache.prepare, prepare)
            self.assertIs(RuntimeCache._read, read)
            self.assertIs(data.prepare_manifests, original_manifest)
            self.assertIs(data.verify_manifests, original_verify)
        logs = output.getvalue()
        self.assertIn("model training has not started", logs)
        self.assertIn("source cache: 2 unique images", logs)
        self.assertIn("current read", logs)
        self.assertIn("a.png", logs)
        self.assertIn("cache ready: 2 images", logs)

    def test_exception_preserved_and_hooks_restored(self):
        original = (data.prepare_manifests, data.verify_manifests, RuntimeCache.prepare, RuntimeCache._read)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "original failure"):
                with self.context()(42):
                    raise ValueError("original failure")
        self.assertEqual(original, (data.prepare_manifests, data.verify_manifests, RuntimeCache.prepare, RuntimeCache._read))


if __name__ == "__main__":
    unittest.main()
