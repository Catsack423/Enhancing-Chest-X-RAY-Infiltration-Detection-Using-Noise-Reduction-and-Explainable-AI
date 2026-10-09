"""Focused scientific and persistence invariants; no full research training here."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import sys
import shutil
import zipfile
import types
from unittest.mock import patch
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
from shared.data_bundle import build_data_bundle, extract_data_bundle
from shared.denoising import read_gray, working_image, cnn_tensor, Preprocessor
from shared.training import epoch_train_frame, loader, ImageDataset
from build_notebooks import SETUP

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
        self.assertEqual({p.name for p in notebooks}, {
            "cnn_comparison.ipynb", "gradcam_comparison.ipynb",
            "shap_comparison.ipynb", "frequency_comparison.ipynb",
        })
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


class DataBundleTests(unittest.TestCase):
    def test_runtime_preprocessing_cache_preserves_pixels_and_reuses_file(self):
        with TestDirectory() as folder:
            exp, _, _ = self.make_bundle(folder)
            cache_root = Path(folder) / "runtime_cache"
            row = {"image": "training.png", "relative_path": "images_001/images/training.png"}
            with patch.dict("os.environ", {"CXR_PREPROCESS_CACHE_ROOT": str(cache_root)}):
                prep = Preprocessor(exp, "median_L1")
                first = prep.image(row)
                cached = prep.cache / row["image"]
                timestamp = cached.stat().st_mtime_ns
                second = prep.image(row)
                self.assertTrue(cached.is_relative_to(cache_root))
                self.assertEqual(cached.stat().st_mtime_ns, timestamp)
                np.testing.assert_array_equal(first, second)
                expected = apply_condition(working_image(read_gray(exp.data_root / row["relative_path"]), exp.config), exp.condition("median_L1"), exp.config)
                np.testing.assert_array_equal(second, expected)

    def test_colab_bootstrap_discards_modules_from_old_drive_folder(self):
        with TestDirectory() as folder:
            root = Path(folder)
            drive_root = root / "drive"
            upload = drive_root / "MyDrive/Notebook_2_colab.zip"
            upload.parent.mkdir(parents=True)
            with zipfile.ZipFile(upload, "w") as package:
                package.writestr("Notebook_2/shared/config.json", "{}")
                package.writestr("Notebook_2/shared/__init__.py", "")
                package.writestr("Notebook_2/nih_cxr_subset.zip", "skip nested data at source bootstrap")
            google = types.ModuleType("google")
            colab = types.ModuleType("google.colab")
            colab.drive = types.SimpleNamespace(mount=lambda *args: None)
            google.colab = colab
            old_shared = types.ModuleType("shared")
            old_shared.__file__ = "/content/drive/MyDrive/nih -xray/notebooks/Notebook_2/shared/__init__.py"
            old_experiment = types.ModuleType("shared.experiment")
            prefix = SETUP.split("\nif IN_COLAB:\n    import subprocess", 1)[0]
            prefix = prefix.replace("/content/drive", drive_root.as_posix())
            namespace = {}
            with patch.dict(sys.modules, {"google": google, "google.colab": colab,
                                          "shared": old_shared, "shared.experiment": old_experiment}):
                exec(compile(prefix, "colab_recovery", "exec"), namespace)
                self.assertNotIn("shared", sys.modules)
                self.assertNotIn("shared.experiment", sys.modules)
                self.assertIs(sys.modules["google.colab"], colab)
            self.assertEqual(namespace["NOTEBOOK_2_DIR"], drive_root / "MyDrive/Notebook_2")
            self.assertTrue((namespace["NOTEBOOK_2_DIR"] / "shared/config.json").is_file())
            self.assertFalse((namespace["NOTEBOOK_2_DIR"] / "nih_cxr_subset.zip").exists())

    def make_bundle(self, folder):
        import cv2
        root = Path(folder)
        data = root / "source"
        data.mkdir()
        for name in ("Data_Entry_2017.csv", "BBox_List_2017.csv", "train_val_list.txt", "test_list.txt"):
            (data / name).write_text("original metadata\n")
        images = data / "images_001/images"
        images.mkdir(parents=True)
        raw = np.random.default_rng(42).integers(0, 256, (512, 512), dtype=np.uint8)
        for name in ("training.png", "native.png"):
            cv2.imencode(".png", raw)[1].tofile(images / name)
        config = json.loads((ROOT / "shared/config.json").read_text())
        exp = Experiment(ROOT, data, root / "results", config)
        manifests = {"train": pd.DataFrame({"relative_path": ["images_001/images/training.png"]}),
                     "xai": pd.DataFrame({"relative_path": ["images_001/images/native.png"]})}
        archive = root / "data.zip"
        build_data_bundle(exp, manifests, archive)
        return exp, config, archive

    def test_compact_data_preserves_cnn_pixels_and_native_xai(self):
        with TestDirectory() as folder:
            exp, config, archive = self.make_bundle(folder)
            target = extract_data_bundle(archive, config, Path(folder) / "cache")
            name = "images_001/images/training.png"
            original = working_image(read_gray(exp.data_root / name), config)
            compact = working_image(read_gray(target / name), config)
            np.testing.assert_array_equal(cnn_tensor(compact, config), cnn_tensor(original, config))
            self.assertEqual(read_gray(target / name).shape, (256, 256))
            native = "images_001/images/native.png"
            self.assertEqual((target / native).read_bytes(), (exp.data_root / native).read_bytes())
            self.assertEqual(extract_data_bundle(archive, config, Path(folder) / "cache"), target)
            original_bytes = (target / name).read_bytes()
            (target / name).write_bytes(b"x" * len(original_bytes))
            self.assertEqual(extract_data_bundle(archive, config, Path(folder) / "cache"), target)
            self.assertEqual((target / name).read_bytes(), original_bytes)

    def test_bundle_rejects_changed_experiment(self):
        with TestDirectory() as folder:
            _, config, archive = self.make_bundle(folder)
            config["seed"] = 43
            with self.assertRaisesRegex(ValueError, "config changed"):
                extract_data_bundle(archive, config, Path(folder) / "cache")

    def test_bundle_rejects_unsafe_paths_and_corrupt_content(self):
        with TestDirectory() as folder:
            _, config, original = self.make_bundle(folder)
            with zipfile.ZipFile(original) as source:
                report = json.loads(source.read("bundle.json"))
                payload = {name: source.read(name) for name in report["files"]}
            for unsafe in (False, True):
                changed = deepcopy(report)
                content = dict(payload)
                name = "images_001/images/training.png"
                if unsafe:
                    content["../escape.png"] = content.pop(name)
                    changed["files"]["../escape.png"] = changed["files"].pop(name)
                else:
                    content[name] = b"x" * len(content[name])
                archive = Path(folder) / f"bad_{unsafe}.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    for key, value in content.items():
                        output.writestr(key, value)
                    output.writestr("bundle.json", json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "Unsafe|Corrupt"):
                    extract_data_bundle(archive, config, Path(folder) / f"cache_{unsafe}")


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
        self.assertEqual(counts, {"train": 320, "validation": 80, "test": 100, "xai": 123, "background": 32})
        self.assertEqual(self.manifests["test"].label.value_counts().to_dict(), {0: 50, 1: 50})
        verify_manifests(self.exp, self.manifests)
        self.assertEqual(self.manifests["xai"].patient_id.nunique(), 115)
        for name in ("train", "validation"):
            self.assertFalse(set(self.manifests[name].patient_id) & set(self.manifests["xai"].patient_id))

    def test_same_seed_reuses_identical_manifests(self):
        repeated = prepare_manifests(self.exp)
        for name in repeated:
            pd.testing.assert_frame_equal(self.manifests[name], repeated[name])

    def test_epoch_uses_all_320_balanced_images_with_shared_ids(self):
        first = epoch_train_frame(self.exp, self.manifests["train"], 1)
        repeated = epoch_train_frame(self.exp, self.manifests["train"], 1)
        second = epoch_train_frame(self.exp, self.manifests["train"], 2)
        self.assertEqual(len(first), 320)
        self.assertEqual(first.label.value_counts().to_dict(), {0: 160, 1: 160})
        self.assertEqual(first.image.nunique(), 320)
        self.assertEqual(set(first.image), set(self.manifests["train"].image))
        pd.testing.assert_frame_equal(first, repeated)
        pd.testing.assert_frame_equal(first, second)
        class FakePreprocessor:
            def tensor(self, row):
                return torch.zeros(3, 4, 4)
        batches = loader(ImageDataset(first, FakePreprocessor()), 16, 42, shuffle=True)
        self.assertEqual(sum(len(images) for images, _ in batches), 320)

    def test_different_official_test_subset_is_rejected(self):
        changed = {k: v.copy() for k, v in self.manifests.items()}
        changed["test"] = changed["test"].iloc[:-1]
        with self.assertRaises(ValueError):
            verify_manifests(self.exp, changed)

    def test_patient_leakage_is_rejected(self):
        changed = {k: v.copy() for k, v in self.manifests.items()}
        changed["train"].iloc[0] = changed["validation"].iloc[0]
        with self.assertRaises(ValueError):
            verify_manifests(self.exp, changed)


if __name__ == "__main__":
    unittest.main()
