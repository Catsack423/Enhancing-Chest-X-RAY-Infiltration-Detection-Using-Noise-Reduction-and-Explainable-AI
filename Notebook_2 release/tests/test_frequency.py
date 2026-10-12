"""Frequency reconstruction, spatial XAI fusion, exact split and notebook checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
import ast
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat
import numpy as np
import torch
from torch import nn

from shared.frequency import (dwt_components, dft_components, FrequencyPreprocessor,
                              FeatureFusionResNet50, frequency_gradcam, frequency_shap_maps,
                              open_frequency_experiment)
from shared.denoising import Preprocessor
from shared.experiment import Experiment, file_hash
from shared.explainers import gradcam
from shared.training import epoch_train_frame

torch.set_num_threads(2)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Conv2d(3, 4, 1, bias=False)
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(self.layer4(x).mean(dim=(2, 3)))


def tiny_fusion():
    model = FeatureFusionResNet50.__new__(FeatureFusionResNet50)
    nn.Module.__init__(model)
    model.backbone = TinyBackbone()
    # BinaryResNet50.train expects these frozen stem/stages to exist.
    for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"):
        setattr(model.backbone, name, nn.Identity())
    return model


class FrequencyMathTests(unittest.TestCase):
    def test_dwt_reconstructs_input_and_level_order(self):
        rng = np.random.default_rng(12)
        x = rng.random((64, 64)).astype(np.float32)
        bands = dwt_components(x)
        self.assertEqual(bands.shape, (5, 64, 64))
        np.testing.assert_allclose(bands.sum(0), x, atol=1e-6)
        checker = ((np.indices((64, 64)).sum(0) % 2) * 2 - 1).astype(np.float32)
        detail = dwt_components(checker)
        np.testing.assert_allclose(detail[1], checker, atol=1e-6)  # D1, not D4
        np.testing.assert_allclose(detail[[0, 2, 3, 4]], 0, atol=1e-6)

    def test_constant_image_has_only_low_frequency(self):
        x = np.full((64, 64), .4, dtype=np.float32)
        for bands in (dwt_components(x), dft_components(x)):
            np.testing.assert_allclose(bands[0], x, atol=1e-6)
            np.testing.assert_allclose(bands[1:], 0, atol=1e-6)

    def test_fft_routes_sinusoids_and_exact_cutoffs(self):
        for frequency, expected_band in ((4/64, 0), (8/64, 1), (12/64, 1), (16/64, 2), (24/64, 2)):
            wave = np.cos(2*np.pi*frequency*np.arange(64))
            x = np.repeat(wave[None], 64, axis=0).astype(np.float32)
            bands = dft_components(x)
            np.testing.assert_allclose(bands.sum(0), x, atol=1e-6)
            np.testing.assert_allclose(bands[expected_band], x, atol=1e-6)
            np.testing.assert_allclose(np.delete(bands, expected_band, axis=0), 0, atol=1e-6)

    def test_invalid_cutoffs_and_depth_fail(self):
        x = np.zeros((64, 64))
        for low, high in ((.25, .125), (0, .25), (.125, .5)):
            with self.assertRaises(ValueError):
                dft_components(x, low, high)
        with self.assertRaises(ValueError):
            dwt_components(x, depth=3)

    def test_logits_equal_mean_feature_head_and_all_views_get_gradients(self):
        model = tiny_fusion()
        x = torch.rand(2, 5, 3, 16, 16, requires_grad=True)
        f = model.backbone.layer4(x.flatten(0, 1)).mean((2, 3)).reshape(2, 5, 4)
        torch.testing.assert_close(model(x), model.backbone.fc(f.mean(1)))
        gradient, = torch.autograd.grad(model(x).sum(), x)
        self.assertTrue((gradient.abs().sum((0, 2, 3, 4)) > 0).all())

    def test_gradcam_matches_baseline_and_view_permutation(self):
        model = tiny_fusion()
        model.backbone.layer4.weight.data.fill_(.1)
        model.backbone.fc.weight.data.fill_(1.)
        x = torch.rand(1, 3, 3, 16, 16)
        fused, prob = frequency_gradcam(model, x[:, :1], (32, 32))
        class Single(nn.Module):
            @property
            def target_layer(self):
                return model.target_layer
            def forward(self, x):
                return model.backbone(x)
        raw, raw_prob = gradcam(Single(), x[:, 0], (32, 32))
        np.testing.assert_allclose(fused, raw, atol=1e-6)
        self.assertAlmostEqual(prob, raw_prob)
        a, pa = frequency_gradcam(model, x, (32, 32))
        b, pb = frequency_gradcam(model, x.flip(1), (32, 32))
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertAlmostEqual(pa, pb)
        self.assertEqual(len(model.target_layer._forward_hooks), 0)

    def test_shap_signed_views_cancel_before_relu_and_shapes(self):
        x = torch.zeros(1, 5, 3, 4, 4)
        values = np.ones(tuple(x.shape), dtype=np.float32)
        values[:, 1:] = -1
        settings = {"seed": 42, "xai": {"shap_nsamples": 4}}
        for output in (values, values[..., None], [values]):
            class Explainer:
                def shap_values(self, *args, **kwargs):
                    return output
            result = frequency_shap_maps(Explainer(), x, (8, 8), settings)
            self.assertEqual(result["view_channel_attributions"].shape, (5, 3, 4, 4))
            self.assertTrue((result["signed_input"] == -9).all())
            self.assertEqual(result["positive_heatmap"].max(), 0)


class ExistingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # mkdtemp uses permissions incompatible with this host's inherited ACL;
        # use the workspace's existing temporary-test parent instead.
        from uuid import uuid4
        cls.directory = ROOT / ".runtime/frequency_tests" / uuid4().hex
        cls.directory.mkdir(parents=True)
        cls.source = ROOT / "runs/pilot320_val80_test100"
        cls.exp, cls.frames = open_frequency_experiment(ROOT, Experiment.open(ROOT).data_root,
                                                       cls.source, cls.directory)

    @classmethod
    def tearDownClass(cls):
        import shutil
        if not cls.directory.resolve().is_relative_to((ROOT / ".runtime/frequency_tests").resolve()):
            raise ValueError("Invalid test cleanup target")
        shutil.rmtree(cls.directory)

    def test_all_manifest_bytes_and_settings_preserved(self):
        for name in self.frames:
            self.assertEqual(file_hash(self.exp.artifacts / f"manifests/{name}.csv"),
                             file_hash(self.source / f"shared/artifacts/manifests/{name}.csv"))
        old = json.loads((self.source / "shared/artifacts/experiment.json").read_text())["config"]
        for key in ("seed", "cnn", "training", "xai", "data", "preprocessing"):
            self.assertEqual(self.exp.config[key], old[key])
        frame = epoch_train_frame(self.exp, self.frames["train"], 1)
        self.assertEqual(set(frame.image), set(self.frames["train"].image))
        self.assertEqual(len(frame), 320)

    def test_baseline_pixels_unchanged_and_frequency_views_finite(self):
        row = self.frames["train"].iloc[0].to_dict()
        old = Preprocessor(self.exp, "baseline").tensor(row)
        new = FrequencyPreprocessor(self.exp, "baseline").tensor(row)
        torch.testing.assert_close(new[0], old, rtol=0, atol=0)
        for condition, n in (("dwt_L1_L4_fusion", 5), ("dft_low_mid_high_fusion", 3)):
            x = FrequencyPreprocessor(self.exp, condition).tensor(row)
            self.assertEqual(tuple(x.shape), (n, 3, 224, 224))
            self.assertTrue(torch.isfinite(x).all())

    def test_resume_keeps_manifests_and_rejects_changed_csv(self):
        open_frequency_experiment(ROOT, self.exp.data_root, self.source, self.directory)
        path = self.exp.artifacts / "manifests/train.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaises(ValueError):
                open_frequency_experiment(ROOT, self.exp.data_root, self.source, self.directory)
        finally:
            path.write_bytes(original)


class NotebookTests(unittest.TestCase):
    def test_notebook_schema_and_syntax(self):
        nb = nbformat.read(ROOT / "Frequency/frequency_comparison.ipynb", as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                self.assertEqual(cell.outputs, [])


if __name__ == "__main__":
    unittest.main()
