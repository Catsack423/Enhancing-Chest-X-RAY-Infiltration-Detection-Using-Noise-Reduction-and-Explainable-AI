"""DAE import gates: only CNN-side differences are allowed."""
from pathlib import Path
import runpy
import unittest

ROOT = Path(__file__).resolve().parents[1]
importer = runpy.run_path(str(ROOT / "tools/import_dae.py"))


class ImportDaeTests(unittest.TestCase):
    def test_reference_is_old_training_and_dae_functions_unchanged(self):
        before = importer["_definitions"](importer["REFERENCE"].read_text(encoding="utf-8"))
        after = importer["_definitions"]((ROOT / "shared/training.py").read_text(encoding="utf-8"))
        for name in importer["DAE_FUNCTIONS"]:
            self.assertIn(name, before)
            self.assertEqual(before[name], after[name], name)
        self.assertNotEqual(before["train_condition"], after["train_condition"])

    def test_dae_scope_ignores_cnn_and_output_paths_only(self):
        old = {"seed": 42, "cnn": {"epochs": 20}, "output_subdirectory": "a", "source_code_hashes": {"x": 1},
               "data": {"colab_output_root": "a", "test_scope": "full"}, "dae": {"batch_size": 64}}
        new = {"seed": 42, "cnn": {"epochs": 20, "early_stopping_patience": 3}, "output_subdirectory": "b",
               "source_code_hashes": {"x": 2}, "data": {"colab_output_root": "b", "test_scope": "full"}, "dae": {"batch_size": 64}}
        scope, differences = importer["_dae_scope"], importer["_differences"]
        self.assertEqual(differences(scope(old), scope(new)), [])
        new["dae"]["batch_size"] = 16
        new["seed"] = 43
        self.assertEqual(differences(scope(old), scope(new)), ["dae.batch_size", "seed"])


if __name__ == "__main__":
    unittest.main()
