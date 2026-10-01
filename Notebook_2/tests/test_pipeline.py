"""Focused scientific and persistence invariants; no full research training here."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import sys
import shutil
from uuid import uuid4
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if (ROOT / ".runtime").is_dir():
    sys.path.insert(0, str(ROOT / ".runtime"))

import nbformat
import numpy as np
import pandas as pd
import torch
from torch import nn

from shared.data import prepare_manifests, verify_manifests
from shared.denoising import apply_condition
from shared.evaluation import bbox_mask, positive_normalize, resize_map, localization_metrics, patient_bootstrap, classification_metrics
from shared.experiment import Experiment, save_torch, write_json
from shared.explainers import gradcam, extract_shap_channels, shap_maps

torch.set_num_threads(2)


class TestDirectory:
    """Inherited Windows ACLs avoid Python 3.14 mode-700 temp-directory issues."""
    def __init__(self):
        self.path = ROOT / ".runtime" / "test_runs" / uuid4().hex
        self.path.mkdir(parents=True)
        self.name = str(self.path)

    def __enter__(self):
        return self.name

    def __exit__(self, *args):
        self.cleanup()

    def cleanup(self):
        parent = (ROOT / ".runtime/test_runs").resolve()
        if not self.path.resolve().is_relative_to(parent):
            raise ValueError("Test cleanup must stay inside the test workspace")
        shutil.rmtree(self.path)


class MetricTests(unittest.TestCase):
    def test_native_bbox_alignment_and_boundaries(self):
        mask = bbox_mask((12, 20), [(4.2, 2.5, 5.1, 3.1)])
        self.assertEqual(int(mask.sum()), 6 * 4)
        self.assertTrue(mask[2, 4])
        self.assertTrue(mask[5, 9])
        self.assertFalse(mask[6, 9])
        heat = np.zeros(mask.shape, dtype=np.float32)
        heat[3, 5] = 1
        result = localization_metrics(heat, mask)
        self.assertEqual(result["pointing_hit"], 1)
        self.assertAlmostEqual(result["iou_0.3"], 1 / mask.sum())

    def test_zero_heatmap_is_always_a_miss(self):
        heat = positive_normalize(np.zeros((10, 10)))
        mask = bbox_mask(heat.shape, [(0, 0, 10, 10)])
        result = localization_metrics(heat, mask)
        self.assertEqual(result["pointing_hit"], 0)
        self.assertEqual(result["iou_0.3"], 0)
        self.assertIsNone(result["peak_x"])

    def test_negative_shap_does_not_localize(self):
        signed = -np.ones((3, 4))
        positive = positive_normalize(resize_map(signed, (15, 20)))
        self.assertTrue(np.array_equal(positive, np.zeros((15, 20))))
        self.assertTrue((signed < 0).all())

    def test_invalid_maps_are_rejected(self):
        with self.assertRaises(ValueError):
            positive_normalize(np.array([[np.nan]]))
        with self.assertRaises(ValueError):
            localization_metrics(np.ones((2, 2)) * 2, np.ones((2, 2), dtype=bool))

    def test_bootstrap_resamples_patients_and_is_paired(self):
        frame = pd.DataFrame({"patient_id": [1, 1, 2], "pointing_hit": [1, 1, 0]})
        a = patient_bootstrap(frame, "pointing_hit", 42, 100)
        b = patient_bootstrap(frame.sample(frac=1, random_state=9), "pointing_hit", 42, 100)
        self.assertEqual(a, b)
        # With two patient clusters, all-zero and all-one replicates are possible.
        self.assertEqual(a, (0.0, 1.0))

    def test_confusion_comes_from_probabilities(self):
        result = classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)


class ModelTests(unittest.TestCase):
    def test_frozen_batchnorm_and_parameter_scope(self):
        from shared.models import BinaryResNet50
        model = BinaryResNet50(pretrained=False).train()
        self.assertFalse(model.backbone.bn1.training)
        self.assertFalse(model.backbone.layer3[0].bn1.training)
        self.assertTrue(model.backbone.layer4[0].bn1.training)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(all(name.startswith(("backbone.layer4.", "backbone.fc.")) for name in trainable))
        before = model.backbone.bn1.running_mean.clone()
        with torch.no_grad():
            model(torch.randn(2, 3, 32, 32))
        torch.testing.assert_close(before, model.backbone.bn1.running_mean, rtol=0, atol=0)

    def test_gradcam_explains_negative_infiltration_logit(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.target_layer = nn.Conv2d(3, 2, 1)
            def forward(self, x):
                return self.target_layer(x).mean((1, 2, 3)).unsqueeze(1) - 100
        model = Toy()
        heatmap, probability = gradcam(model, torch.ones(1, 3, 8, 8), (24, 32))
        self.assertLess(probability, 0.5)
        self.assertEqual(heatmap.shape, (24, 32))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertEqual(len(model.target_layer._forward_hooks), 0)

    def test_real_gradient_explainer_preserves_signed_values(self):
        import shap
        class SignedLogit(nn.Module):
            def forward(self, x):
                return (x[:, 0] - 2 * x[:, 1] + 0.2 * x[:, 2]).sum((1, 2)).unsqueeze(1)
        model = SignedLogit()
        background = torch.zeros(2, 3, 8, 8)
        x = torch.ones(1, 3, 8, 8)
        x[:, 1, :, :4] = 0
        explainer = shap.GradientExplainer(model, background, batch_size=4)
        config = {"seed": 42, "xai": {"shap_nsamples": 16}}
        maps = shap_maps(explainer, x, (24, 32), config)
        self.assertTrue((maps["channel_attributions"] < 0).any())
        self.assertTrue((maps["channel_attributions"] > 0).any())
        self.assertTrue((maps["signed_native"] < 0).any())
        self.assertTrue((maps["signed_native"] > 0).any())
        self.assertEqual(maps["positive_heatmap"].shape, (24, 32))
        self.assertTrue((maps["positive_heatmap"][maps["signed_native"] <= 0] == 0).all())
        self.assertAlmostEqual(float(maps["channel_attributions"].sum()), float(model(x)[0, 0]), places=4)

    def test_shap_output_shape_variants(self):
        values = np.ones((1, 3, 4, 5))
        np.testing.assert_array_equal(extract_shap_channels([values], 4, 5), values[0])
        np.testing.assert_array_equal(extract_shap_channels(values[..., None], 4, 5), values[0])
        with self.assertRaises(ValueError):
            extract_shap_channels(np.ones((1, 1000)), 4, 5)


class PersistenceTests(unittest.TestCase):
    def test_wrong_checkpoint_condition_config_and_manifest_fail(self):
        config = json.loads((ROOT / "shared/config.json").read_text())
        with TestDirectory() as tmp:
            exp = Experiment(ROOT, Path(tmp), Path(tmp), config)
            folder = exp.artifacts / "manifests"
            folder.mkdir(parents=True)
            for name in ("train", "validation", "test", "xai", "background"):
                (folder / f"{name}.csv").write_text("image,label\nx.png,1\n")
            saved = exp.provenance("baseline")
            exp.check_provenance(saved, "baseline")
            with self.assertRaises(ValueError):
                exp.check_provenance(saved, "median_L1")
            changed = deepcopy(config)
            changed["seed"] = 43
            with self.assertRaises(ValueError):
                Experiment(ROOT, Path(tmp), Path(tmp), changed).check_provenance(saved, "baseline")
            checkpoint_path = Path(tmp) / "test.pt"
            save_torch(checkpoint_path, {"provenance": saved, "weights": torch.tensor([1.0])})
            loaded = exp.load_checkpoint(checkpoint_path, "baseline")
            torch.testing.assert_close(loaded["weights"], torch.tensor([1.0]))
            (folder / "train.csv").write_text("image,label\ny.png,1\n")
            with self.assertRaises(ValueError):
                exp.load_checkpoint(checkpoint_path, "baseline")

    def test_notebooks_validate_and_code_cells_compile(self):
        notebooks = sorted(ROOT.glob("*/*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for path in notebooks:
            nb = nbformat.read(path, as_version=4)
            nbformat.validate(nb)
            for cell in nb.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)


class DenoisingTests(unittest.TestCase):
    def test_all_traditional_presets_and_baseline_identity(self):
        config = json.loads((ROOT / "shared/config.json").read_text())
        image = np.random.default_rng(42).integers(0, 256, (32, 32), dtype=np.uint8)
        for condition in config["conditions"][:7]:
            result = apply_condition(image, condition, config)
            self.assertEqual(result.shape, image.shape)
            self.assertEqual(result.dtype, np.uint8)
            if condition["group"] == "baseline":
                np.testing.assert_array_equal(result, image)
        for condition in config["conditions"][7:]:
            with self.assertRaises(RuntimeError):
                apply_condition(image, condition, config)

    def test_constant_wavelet_image_is_finite(self):
        config = json.loads((ROOT / "shared/config.json").read_text())
        for condition in config["conditions"][4:7]:
            output = apply_condition(np.zeros((32, 32), dtype=np.uint8), condition, config)
            self.assertTrue(np.isfinite(output).all())


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / ".runtime").mkdir(exist_ok=True)
        cls.temp = TestDirectory()
        cls.exp = Experiment.open(ROOT, output_root=cls.temp.name)
        cls.manifests = prepare_manifests(cls.exp)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_counts_and_official_xai_exclusion(self):
        counts = {name: len(frame) for name, frame in self.manifests.items()}
        self.assertEqual(counts, {"train": 200, "validation": 50, "test": 12081, "xai": 123, "background": 32})
        verify_manifests(self.exp, self.manifests)
        self.assertEqual(self.manifests["xai"].patient_id.nunique(), 115)
        for name in ("train", "validation"):
            self.assertFalse(set(self.manifests[name].patient_id) & set(self.manifests["xai"].patient_id))

    def test_same_seed_reuses_identical_manifests(self):
        repeated = prepare_manifests(self.exp)
        for name in repeated:
            pd.testing.assert_frame_equal(self.manifests[name], repeated[name])

    def test_patient_leakage_is_rejected(self):
        changed = {k: v.copy() for k, v in self.manifests.items()}
        changed["train"].iloc[0] = changed["validation"].iloc[0]
        with self.assertRaises(ValueError):
            verify_manifests(self.exp, changed)


if __name__ == "__main__":
    unittest.main()
