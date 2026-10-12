"""Regression checks for frozen weights/statistics and differentiable XAI."""
import ast
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import torch
from torch import nn

from shared.beta import HeadOnlyFusionResNet50, beta_model_factory
from shared.experiment import Experiment, file_hash, set_seed
from shared.frequency import open_frequency_experiment, frequency_gradcam, FrequencyPreprocessor
from shared.training import cnn_optimizer, loss_epoch, epoch_train_frame
from shared.epoch_comparison import _check_controls, compare_with_e2

torch.set_num_threads(2)


class HeadOnlyTests(unittest.TestCase):
    def test_optimizer_updates_only_head_and_no_backbone_state_changes(self):
        set_seed(42)
        model = HeadOnlyFusionResNet50(pretrained=False)
        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(sum(p.numel() for p in trainable.values()), 2049)
        self.assertTrue(all(n.startswith("head.") for n in trainable))
        frozen = {name: value.detach().clone() for name, value in model.backbone.state_dict().items()}
        before = model.head[1].weight.detach().clone()
        settings = {"optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4}
        optimizer = cnn_optimizer(model, settings)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]}, {id(p) for p in trainable.values()})
        x = torch.rand(2, 5, 3, 32, 32)
        y = torch.tensor([[0.], [1.]])
        self.assertTrue(np.isfinite(loss_epoch(model, [(x, y)], nn.BCEWithLogitsLoss(), "cpu", optimizer)))
        self.assertTrue(model.head[0].training)
        self.assertTrue(all(not m.training for m in model.backbone.modules()))
        self.assertFalse(torch.equal(before, model.head[1].weight))
        for name, value in model.backbone.state_dict().items():
            torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertTrue(all(not m.training for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)))

    def test_eval_feature_mean_and_all_view_input_gradients(self):
        set_seed(42)
        model = HeadOnlyFusionResNet50(pretrained=False).eval()
        x = torch.rand(2, 5, 3, 32, 32, requires_grad=True)
        features = model.backbone(x.flatten(0, 1)).reshape(2, 5, 2048)
        expected = model.head(features.mean(1))
        torch.testing.assert_close(model(x), expected)
        torch.testing.assert_close(model(x), model(x.flip(1)), atol=1e-6, rtol=1e-5)
        gradient, = torch.autograd.grad(model(x).sum(), x)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient.abs().sum((0, 2, 3, 4)) > 0).all())
        self.assertFalse(model.head[0].training)

    def test_gradcam_with_fully_frozen_parameters(self):
        set_seed(42)
        model = HeadOnlyFusionResNet50(pretrained=False)
        for views in (1, 3, 5):
            x = torch.rand(1, views, 3, 32, 32)
            heatmap, probability = frequency_gradcam(model, x, (64, 64))
            self.assertEqual(heatmap.shape, (64, 64))
            self.assertTrue(np.isfinite(heatmap).all())
            self.assertTrue(0 <= probability <= 1)
            self.assertEqual(len(model.target_layer._forward_hooks), 0)


class ExistingSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / ".runtime/tests" / uuid4().hex
        cls.exp, cls.frames = open_frequency_experiment(
            ROOT, Experiment.open(ROOT).data_root, ROOT / "source_run", cls.directory)

    @classmethod
    def tearDownClass(cls):
        if not cls.directory.resolve().is_relative_to((ROOT / ".runtime/tests").resolve()):
            raise ValueError("Invalid test cleanup path")
        shutil.rmtree(cls.directory)

    def test_manifests_exact_and_beta_settings_shared(self):
        original = ROOT.parent / "Notebook_2 release/runs/pilot320_val80_test100/shared/artifacts"
        for name in self.frames:
            self.assertEqual(file_hash(self.exp.artifacts / f"manifests/{name}.csv"),
                             file_hash(original / f"manifests/{name}.csv"))
        self.assertEqual(self.exp.config["cnn"]["epochs"], 7)
        self.assertEqual(self.exp.config["cnn"]["patience"], 7)
        self.assertEqual(self.exp.config["cnn"]["trainable_layers"], "head_only")
        self.assertEqual(self.exp.config["cnn"]["optimizer"], "AdamW")
        self.assertEqual(self.exp.config["cnn"]["head_dropout"], 0.3)
        self.assertEqual(self.frames["train"].label.value_counts().to_dict(), {0: 160, 1: 160})
        for epoch in (1, 2, 7):
            selected = epoch_train_frame(self.exp, self.frames["train"], epoch)
            self.assertEqual(set(selected.image), set(self.frames["train"].image))

    def test_baseline_and_frequency_inputs_preserved(self):
        row = self.frames["train"].iloc[0].to_dict()
        from shared.denoising import Preprocessor
        old = Preprocessor(self.exp, "baseline").tensor(row)
        new = FrequencyPreprocessor(self.exp, "baseline").tensor(row)
        torch.testing.assert_close(old, new[0], rtol=0, atol=0)
        for condition, count in [("dwt_L1_L4_fusion", 5), ("dft_low_mid_high_fusion", 3)]:
            tensor = FrequencyPreprocessor(self.exp, condition).tensor(row)
            self.assertEqual(tuple(tensor.shape), (count, 3, 224, 224))
            self.assertTrue(torch.isfinite(tensor).all())

    def test_config_change_cannot_resume_old_lock(self):
        self.exp.config["cnn"]["head_dropout"] = 0.4
        try:
            with self.assertRaises(ValueError):
                self.exp.lock()
        finally:
            self.exp.config["cnn"]["head_dropout"] = 0.3

    def test_seven_epochs_finish_even_with_worsening_validation(self):
        # Exercise the public training loop with a tiny injected model and a
        # deliberately worsening validation sequence. No research run is made.
        from shared.training import train_condition
        from shared.experiment import write_json
        class TinyModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.head = nn.Linear(1, 1)
            def forward(self, x):
                return self.head(x)
        class TinyPreprocessor:
            def __init__(self, *args):
                pass
            def prepare(self, frames):
                pass
            def tensor(self, row):
                return torch.tensor([float(row["label"])])
        config = deepcopy(self.exp.config)
        config["training"]["images_per_epoch"] = 4
        trial = Experiment(ROOT, self.exp.data_root, self.directory / "budget_regression", config)
        target = trial.artifacts / "manifests"
        target.mkdir(parents=True)
        for path in (self.exp.artifacts / "manifests").glob("*.csv"):
            shutil.copyfile(path, target / path.name)
        validation_calls = []
        def scripted_loss(model, batches, criterion, device, optimizer=None):
            if optimizer is not None:
                return 0.4
            validation_calls.append(1)
            return 0.4 if len(validation_calls) % 2 else 0.5 + len(validation_calls) / 20
        with patch("shared.training.loss_epoch", side_effect=scripted_loss):
            result = train_condition(trial, self.frames, "baseline", "cpu",
                                     model_factory=TinyModel, preprocessor_factory=TinyPreprocessor)
        self.assertEqual(result["epochs_run"], 7)
        self.assertEqual(result["best_epoch"], 1)

    def test_comparison_accepts_only_budget_changes(self):
        b = self.exp.provenance("baseline")
        a = deepcopy(b)
        a["config"]["cnn"].update(epochs=2, patience=2)
        a["config"]["output_subdirectory"] = "runs/head_only_e2"
        _check_controls(a, b)
        a["config"]["cnn"]["head_dropout"] = 0.4
        with self.assertRaises(ValueError):
            _check_controls(a, b)
        a["config"]["cnn"]["head_dropout"] = 0.3
        a["versions"]["torch"] = "different"
        with self.assertRaises(ValueError):
            _check_controls(a, b)

    def test_missing_e2_results_are_not_substituted(self):
        missing_root = self.directory / "missing_e2"
        report = compare_with_e2(self.exp, missing_root)
        self.assertFalse(report["ready"])
        self.assertTrue(report["missing"])
        self.assertFalse(missing_root.exists())


class NotebookPackageTests(unittest.TestCase):
    def test_notebook_clean_schema_and_code(self):
        notebook = nbformat.read(ROOT / "frequency_head_only_e7.ipynb", as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
        code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        self.assertIn("train_beta(exp", code)
        self.assertIn("evaluate_beta(exp", code)
        self.assertNotIn("train_frequency(exp", code)

    def test_package_contains_identical_data_and_manifests(self):
        from hashlib import sha256
        with zipfile.ZipFile(ROOT / "Notebook_3_beta_e7_colab.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
            self.assertTrue(all(n.startswith("Notebook_3_e7/") for n in archive.namelist()))
            self.assertFalse(any(".runtime/" in n or "runs/" in n for n in archive.namelist()))
            self.assertEqual(sha256(archive.read("Notebook_3_e7/nih_cxr_subset.zip")).hexdigest(),
                             file_hash(ROOT.parent / "Notebook_2 release/nih_cxr_subset.zip"))
            for name in ("train", "validation", "test", "xai", "background"):
                member = f"Notebook_3_e7/source_run/shared/artifacts/manifests/{name}.csv"
                self.assertEqual(sha256(archive.read(member)).hexdigest(),
                                 file_hash(ROOT / f"source_run/shared/artifacts/manifests/{name}.csv"))
            bundled = json.loads(archive.read("Notebook_3_e7/shared/config.json"))
            self.assertEqual(bundled["cnn"]["epochs"], 7)

    def test_v2_sources_unchanged(self):
        lineage = json.loads((ROOT / "V2_SOURCE_HASHES.json").read_text())
        for relative, expected in lineage.items():
            if relative.startswith("source_run/"):
                v2 = ROOT.parent / "Notebook_2 release/runs/pilot320_val80_test100" / relative.removeprefix("source_run/")
            else:
                v2 = ROOT.parent / "Notebook_2 release" / relative
            self.assertEqual(file_hash(v2), expected)

    def test_e2_model_loop_data_and_original_files_preserved(self):
        original = ROOT.parent / "Notebook_3 beta"
        a = json.loads((original / "shared/config.json").read_text())
        b = json.loads((ROOT / "shared/config.json").read_text())
        for config in (a, b):
            config.pop("output_subdirectory")
            config["cnn"].pop("epochs")
            config["cnn"].pop("patience")
        self.assertEqual(a, b)
        self.assertEqual(file_hash(ROOT / "shared/training.py"), file_hash(original / "shared/training.py"))
        def model_ast(path):
            module = ast.parse(path.read_text(encoding="utf-8"))
            return ast.dump(next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "HeadOnlyFusionResNet50"))
        self.assertEqual(model_ast(ROOT / "shared/beta.py"), model_ast(original / "shared/beta.py"))
        for relative, expected in json.loads((ROOT / "E2_SOURCE_HASHES.json").read_text()).items():
            self.assertEqual(file_hash(original / relative), expected)


if __name__ == "__main__":
    unittest.main()
