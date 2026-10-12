"""Regression checks for frozen weights/statistics and differentiable XAI."""
import ast
from copy import deepcopy
import io
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
import pandas as pd
import torch
from torch import nn

from shared.beta import HeadOnlyFusionResNet50, beta_model_factory
from shared.experiment import Experiment, file_hash, set_seed
from shared.frequency import open_frequency_experiment, frequency_gradcam, FrequencyPreprocessor
from shared.training import cnn_optimizer, loss_epoch, epoch_train_frame

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


class ExpandedTrainingTests(unittest.TestCase):
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

    def test_manifests_extended_and_beta_settings_shared(self):
        original = ROOT / "reference_run/shared/artifacts"
        for name in self.frames:
            self.assertEqual(file_hash(self.exp.artifacts / f"manifests/{name}.csv"),
                             file_hash(ROOT / f"source_run/shared/artifacts/manifests/{name}.csv"))
            if name != "train":
                self.assertEqual(file_hash(self.exp.artifacts / f"manifests/{name}.csv"),
                                 file_hash(original / f"manifests/{name}.csv"))
        old_train = pd.read_csv(original / "manifests/train.csv")
        train = self.frames["train"]
        self.assertEqual(len(train), 400)
        self.assertEqual(train.image.nunique(), 400)
        retained = train[train.image.isin(old_train.image)].reset_index(drop=True)
        pd.testing.assert_frame_equal(retained, old_train.sort_values("image").reset_index(drop=True))
        self.assertEqual(train[~train.image.isin(old_train.image)].label.value_counts().to_dict(), {0: 40, 1: 40})
        self.assertEqual(self.exp.config["cnn"]["epochs"], 10)
        self.assertEqual(self.exp.config["cnn"]["patience"], 10)
        self.assertEqual(self.exp.config["cnn"]["trainable_layers"], "head_only")
        self.assertEqual(self.exp.config["cnn"]["optimizer"], "AdamW")
        self.assertEqual(self.exp.config["cnn"]["head_dropout"], 0.3)
        self.assertEqual(self.frames["train"].label.value_counts().to_dict(), {0: 200, 1: 200})
        for epoch in (1, 2, 10):
            selected = epoch_train_frame(self.exp, self.frames["train"], epoch)
            self.assertEqual(set(selected.image), set(self.frames["train"].image))

    def test_added_images_avoid_entire_original_validation_partition(self):
        from shared.data import load_metadata
        meta, official = load_metadata(self.exp)
        pool = meta[meta["Image Index"].isin(official["train_val"]) &
                    meta["Finding Labels"].isin(["No Finding", "Infiltration"])]
        patients = np.random.default_rng(42).permutation(sorted(pool["Patient ID"].unique()))
        reserved = set(patients[:int(np.ceil(len(patients) * 0.2))])
        self.assertEqual(len(reserved), 4815)
        self.assertFalse(set(self.frames["train"].patient_id) & reserved)
        self.assertTrue(set(self.frames["validation"].patient_id) <= reserved)
        for name in ("validation", "test", "xai"):
            self.assertFalse(set(self.frames["train"].patient_id) & set(self.frames[name].patient_id))
        self.assertEqual(set(self.frames["train"].finding_labels), {"No Finding", "Infiltration"})

    def test_expansion_is_deterministic_on_rerun(self):
        from shared.class_dataset import prepare_class_dataset
        before = file_hash(ROOT / "source_run/shared/artifacts/manifests/train.csv")
        _, repeated, report = prepare_class_dataset(ROOT, self.exp.data_root)
        pd.testing.assert_frame_equal(repeated["train"], self.frames["train"])
        self.assertEqual(before, file_hash(ROOT / "source_run/shared/artifacts/manifests/train.csv"))
        self.assertEqual(report["retained_train_images"], 320)
        self.assertEqual(report["added_train_images"], 80)
        self.assertFalse(report["image_availability_used_for_selection"])

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

    def test_ten_epochs_finish_even_with_worsening_validation(self):
        from shared.training import train_condition

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
        self.assertEqual(result["epochs_run"], 10)
        self.assertEqual(result["best_epoch"], 1)


class NotebookPackageTests(unittest.TestCase):
    def test_notebook_clean_schema_and_code(self):
        notebook = nbformat.read(ROOT / "frequency_class200_e10.ipynb", as_version=4)
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

    def test_package_contains_complete_expanded_data_and_manifests(self):
        from hashlib import sha256
        with zipfile.ZipFile(ROOT / "Notebook_3_beta_class_colab.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
            self.assertTrue(all(n.startswith("Notebook_3_class/") for n in archive.namelist()))
            self.assertFalse(any(".runtime/" in n or "runs/" in n for n in archive.namelist()))
            self.assertEqual(sha256(archive.read("Notebook_3_class/nih_cxr_subset.zip")).hexdigest(),
                             file_hash(ROOT / "nih_cxr_subset.zip"))
            with zipfile.ZipFile(io.BytesIO(archive.read("Notebook_3_class/nih_cxr_subset.zip"))) as data_zip:
                self.assertIsNone(data_zip.testzip())
                report = json.loads(data_zip.read("bundle.json"))
                self.assertEqual(report["image_count"], 702)
                self.assertEqual(report["counts"], {"train": 400, "validation": 80, "test": 100,
                                                  "xai": 123, "background": 32})
                added = pd.read_csv(ROOT / "source_run/shared/artifacts/added_train_images.csv")
                self.assertTrue(set(added.relative_path) <= set(data_zip.namelist()))
                old = json.loads((ROOT.parent / "Notebook_2 release/nih_cxr_subset.json").read_text())
                # Old selected pixels and metadata remain identical in the new bundle.
                self.assertTrue(set(old["files"]) <= set(report["files"]))
                for name, info in old["files"].items():
                    self.assertEqual(info["sha256"], report["files"][name]["sha256"])
            for name in ("train", "validation", "test", "xai", "background"):
                member = f"Notebook_3_class/source_run/shared/artifacts/manifests/{name}.csv"
                self.assertEqual(sha256(archive.read(member)).hexdigest(),
                                 file_hash(ROOT / f"source_run/shared/artifacts/manifests/{name}.csv"))
            bundled = json.loads(archive.read("Notebook_3_class/shared/config.json"))
            self.assertEqual(bundled["cnn"]["epochs"], 10)
            self.assertEqual(bundled["data"]["train_per_class"], 200)

    def test_portable_colab_loader_opens_all_400_training_images(self):
        from shared.data_bundle import colab_data_from_package
        directory = ROOT / ".runtime/tests" / uuid4().hex
        try:
            data = colab_data_from_package(ROOT / "Notebook_3_beta_class_colab.zip", ROOT,
                                           cache_root=directory / "cache")
            exp, frames = open_frequency_experiment(ROOT, data, ROOT / "source_run", directory / "portable_run")
            self.assertEqual(len(frames["train"]), 400)
            for frame in frames.values():
                self.assertTrue(all((data / path).is_file() for path in frame.relative_path))
            self.assertEqual(exp.config["cnn"]["epochs"], 10)
        finally:
            if not directory.resolve().is_relative_to((ROOT / ".runtime/tests").resolve()):
                raise ValueError("Invalid portable test cleanup path")
            if directory.is_dir():
                shutil.rmtree(directory)

    def test_v2_sources_unchanged(self):
        lineage = json.loads((ROOT / "V2_SOURCE_HASHES.json").read_text())
        for relative, expected in lineage.items():
            if relative.startswith("source_run/"):
                v2 = ROOT.parent / "Notebook_2 release/runs/pilot320_val80_test100" / relative.removeprefix("source_run/")
            else:
                v2 = ROOT.parent / "Notebook_2 release" / relative
            self.assertEqual(file_hash(v2), expected)

    def test_e2_model_loop_and_regularization_preserved(self):
        original = ROOT.parent / "Notebook_3 beta"
        a = json.loads((original / "shared/config.json").read_text())
        b = json.loads((ROOT / "shared/config.json").read_text())
        b.pop("training_data_change")
        for config in (a, b):
            config.pop("output_subdirectory")
            config["cnn"].pop("epochs")
            config["cnn"].pop("patience")
            config["training"].pop("images_per_epoch")
            config["data"].pop("train_per_class")
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
