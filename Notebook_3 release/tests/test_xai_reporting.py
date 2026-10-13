"""Regression proof for coordinate frames, signs, clustering and durable resume."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.evaluation import (bbox_mask, resize_map, positive_normalize, localization_metrics,
                               classification_metrics, classification_bootstrap, _weighted_scores, pair_records,
                               youden_threshold)
from shared.explainers import gradcam_maps, extract_shap_channels, shap_maps
from shared.experiment import digest, file_hash, write_json
from shared.xai import evaluate_condition
from shared.reporting import report_all, summarize_seeds


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.target_layer = nn.Identity()
    def forward(self, x):
        n, v, c, h, w = x.shape
        features = self.target_layer(x.reshape(n*v, c, h, w))
        return features.reshape(n, v, -1).mean((1,2)).unsqueeze(1)


class FakeExperiment:
    def __init__(self, directory):
        self.root = Path(directory)
        self.output_root = self.root/"output"
        self.staging_root = self.root/"stage"
        self.artifacts = self.output_root/"shared/artifacts"
        self.config = {"seed":42, "run_mode":"smoke", "xai":{"iou_thresholds":[.3,.5], "bootstrap_replicates":20,
                        "shap_nsamples":4,"shap_batch_size":2}, "cnn":{"classification_threshold":.5},
                       "conditions":[{"id":"baseline", "group":"baseline", "level":0}]}
        self.condition_ids = ["baseline"]
        self.cache = SimpleNamespace(native=lambda row:np.zeros((16,32), dtype=np.uint8))
    def condition(self, condition):
        return self.config["conditions"][0]
    def lock(self):
        pass
    def results(self, criterion, condition):
        return self.output_root/criterion/"results"/condition
    def provenance(self, condition=None):
        return {"condition_id":condition, "fixed":"source-manifest-model-config"}
    def check_provenance(self, value, condition=None):
        if value != self.provenance(condition):
            raise ValueError("Provenance changed")


class TinyPreprocessor:
    def __init__(self, *args):
        pass
    def tensor(self, row):
        return torch.ones(1,3,8,8)
    def prepare(self, frames):
        return self


class ExplanationTests(unittest.TestCase):
    def test_bbox_and_resized_map_use_native_nonsquare_coordinates(self):
        small = np.zeros((4,8), np.float32)
        small[1,5] = 1
        native = positive_normalize(resize_map(small, (40,160)))
        mask = bbox_mask(native.shape, [(90,5,30,20)])
        measured = localization_metrics(native, mask)
        self.assertEqual(measured["pointing_hit"], 1)
        self.assertGreater(measured["iou_0.3"], 0)
        self.assertGreater(measured["peak_x"], 90)
        self.assertEqual(mask.sum(), 600)

    def test_empty_map_is_retained_as_miss_and_zero_energy(self):
        mask = bbox_mask((16,32), [(4,4,8,8)])
        measured = localization_metrics(np.zeros(mask.shape), mask)
        self.assertEqual(measured["empty_heatmap"], 1)
        self.assertEqual(measured["pointing_hit"], 0)
        self.assertEqual(measured["iou_0.3"], 0)
        self.assertEqual(measured["energy_inside"], 0)
        self.assertIsNone(measured["peak_x"])

    def test_gradcam_sums_signed_views_before_relu_and_removes_hook(self):
        model = TinyModel()
        tensor = torch.ones(1,2,3,8,8)
        tensor[:,1] = -2
        arrays, probability = gradcam_maps(model, tensor, (16,32))
        self.assertTrue((arrays["signed_view_input"][0] > 0).all())
        self.assertTrue((arrays["signed_view_input"][1] < 0).all())
        self.assertFalse(arrays["positive_heatmap"].any())
        self.assertTrue(0 <= probability <= 1)
        self.assertEqual(len(model.target_layer._forward_hooks), 0)
        np.testing.assert_allclose(arrays["signed_view_input"].sum(0), arrays["signed_input"])

    def test_shap_shape_variants_and_signed_view_rgb_conservation(self):
        shape = (1,5,3,8,8)
        values = np.ones(shape, np.float32)
        values[:,1:] = -1
        for variant in (values, [values], values[...,None]):
            channels = extract_shap_channels(variant, shape)
            self.assertEqual(channels.shape, (5,3,8,8))
        with self.assertRaises(ValueError):
            extract_shap_channels(np.zeros((1,3,8,8)), shape)
        class FakeExplainer:
            def shap_values(self, tensor, **kwargs):
                return values[...,None]
        arrays = shap_maps(FakeExplainer(), torch.ones(shape), (16,32), {"seed":42,"xai":{"shap_nsamples":4}})
        self.assertTrue((arrays["signed_native"] < 0).all())
        self.assertFalse(arrays["positive_heatmap"].any())
        np.testing.assert_allclose(arrays["view_channel_attributions"].sum((0,1)), arrays["signed_input"])

    def test_actual_gradient_explainer_preserves_multiview_sum(self):
        import shap
        model = TinyModel().eval()
        background = torch.zeros(2,3,3,8,8)
        tensor = torch.ones(1,3,3,8,8)
        explainer = shap.GradientExplainer(model,background,batch_size=2,local_smoothing=0)
        arrays = shap_maps(explainer,tensor,(16,32),{"seed":42,"xai":{"shap_nsamples":4}})
        self.assertEqual(arrays["view_channel_attributions"].shape,(3,3,8,8))
        self.assertAlmostEqual(float(arrays["signed_input"].sum()),1,places=5)
        self.assertTrue((arrays["positive_heatmap"] > 0).all())


class ClusteredMetricsTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"image":["a","b","c","d","e","f"], "patient_id":[1,1,2,3,3,4],
                             "label":[0,1,0,1,1,0], "prob_infiltration":[.2,.8,.8,.4,.8,.1]})
    def test_weighted_sorted_bootstrap_metrics_match_repeated_observations(self):
        frame = self.frame()
        weights = np.array([[2,2,1,3,3,1], [1,1,1,1,1,1]], dtype=float)
        scores = _weighted_scores(frame, weights, .5)
        for i, w in enumerate(weights):
            indexes = np.repeat(np.arange(len(frame)), w.astype(int))
            expected = classification_metrics(frame.label.to_numpy()[indexes], frame.prob_infiltration.to_numpy()[indexes])
            for metric in scores:
                self.assertAlmostEqual(scores[metric][i], expected[metric], places=12)
    def test_paired_patient_bootstrap_self_delta_zero_and_order_invariant(self):
        frame = self.frame()
        result = classification_bootstrap(frame, replicates=100, baseline=frame.sample(frac=1, random_state=9))
        np.testing.assert_allclose(result[["mean_delta","ci_low","ci_high"]], 0)
        self.assertEqual(set(result.n_patients), {4})
        self.assertEqual(set(result.n_images), {6})
    def test_youden_threshold_maximizes_validation_j_inside_open_interval(self):
        truth = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([.1, .2, .5, .6, .7, .9])
        threshold = youden_threshold(truth, scores)
        self.assertEqual(threshold, .6)
        self.assertEqual(youden_threshold(truth, [.1, .3, .6, .4, .7, .9]), .7)  # tie .4/.7 keeps the higher
        best = classification_metrics(truth, scores, threshold)["balanced_accuracy"]
        for candidate in scores:
            self.assertLessEqual(classification_metrics(truth, scores, candidate)["balanced_accuracy"], best)
        self.assertTrue(0 < youden_threshold([0, 1], [0., 1.]) < 1)
        with self.assertRaises(ValueError):
            youden_threshold([0, 0], [.1, .2])
    def test_paired_bootstrap_keeps_each_models_threshold(self):
        frame = self.frame()
        result = classification_bootstrap(frame, .5, replicates=100, baseline=frame, baseline_threshold=.15).set_index("metric")
        np.testing.assert_allclose(result.loc[["roc_auc", "pr_auc"], ["mean_delta", "ci_low", "ci_high"]], 0)
        expected = classification_metrics(frame.label, frame.prob_infiltration, .5)["specificity"] - \
            classification_metrics(frame.label, frame.prob_infiltration, .15)["specificity"]
        self.assertAlmostEqual(result.loc["specificity", "mean_delta"], expected)
        self.assertNotEqual(expected, 0)
    def test_paired_comparison_rejects_patient_metadata_changes(self):
        frame = self.frame()
        other = frame.copy()
        other.loc[0,"patient_id"] = 999
        with self.assertRaises(ValueError):
            pair_records(frame, other, ["label"])
    def test_single_class_auc_unavailable_and_invalid_scores_fail(self):
        metrics = classification_metrics([0,0], [.2,.3])
        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])
        with self.assertRaises(ValueError):
            classification_metrics([0,1], [.2,np.nan])


class DurableResultTests(unittest.TestCase):
    def test_completed_cnn_xai_reports_pairing_and_seed_summary(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            exp = FakeExperiment(directory)
            exp.output_root = exp.root/"release_full/seed_42"
            exp.artifacts = exp.output_root/"shared/artifacts"
            exp.config["conditions"].append({"id":"median_L1", "group":"median", "level":1})
            exp.condition_ids.append("median_L1")
            manifests = exp.artifacts/"manifests"
            manifests.mkdir(parents=True)
            test = pd.DataFrame({"image":["n1.png","p1.png","n2.png","p2.png"],"patient_id":[1,1,2,2],"label":[0,1,0,1]})
            xai = pd.DataFrame([{"image":"p1.png","patient_id":1,"risk_flag":"pure","bbox_x":4,"bbox_y":4,"bbox_w":8,"bbox_h":8},
                                {"image":"p2.png","patient_id":2,"risk_flag":"mixed_low_risk","bbox_x":4,"bbox_y":4,"bbox_w":8,"bbox_h":8}])
            test.to_csv(manifests/"test.csv", index=False)
            xai.to_csv(manifests/"xai.csv", index=False)
            def provenance(condition=None):
                return {"config":exp.config,"config_hash":digest(exp.config),"condition_id":condition,
                        "manifest_hashes":{p.name:file_hash(p) for p in manifests.glob("*.csv")},"versions":{"test":"1"}}
            exp.provenance = provenance
            for condition in exp.condition_ids:
                folder = exp.results("CNN",condition)
                folder.mkdir(parents=True)
                (folder/"best.pt").write_bytes(b"test-only-checkpoint")
                predictions = test.assign(prob_infiltration=[.2,.8,.3,.7])
                predictions.to_csv(folder/"test_predictions.csv",index=False)
                history = pd.DataFrame({"epoch":[1,2],"train_eval_loss":[.5,.4],"validation_loss":[.6,.5]})
                history.to_csv(folder/"history.csv",index=False)
                scores = classification_metrics(predictions.label,predictions.prob_infiltration)
                scores.update(condition_id=condition,best_epoch=2,threshold=.5)
                write_json(folder/"metrics.json",scores)
                write_json(folder/"complete.json",{"provenance":provenance(condition),"checkpoint_hash":file_hash(folder/"best.pt"),
                           "predictions_hash":file_hash(folder/"test_predictions.csv"),"history_hash":file_hash(folder/"history.csv"),"metrics_hash":file_hash(folder/"metrics.json")})
                evaluate_condition(exp,{"xai":xai,"background":xai},"Grad-CAM",condition,device="cpu",
                                   preprocessor_factory=TinyPreprocessor,model_loader=lambda *args,**kw:TinyModel())
            measured = report_all(exp)
            self.assertEqual(set(measured["classification.csv"].status), {"complete"})
            np.testing.assert_allclose(measured["paired_classification_deltas.csv"]["mean_delta"],0)
            np.testing.assert_allclose(measured["paired_xai_deltas.csv"]["mean_delta"],0)
            cnn,xai_summary = summarize_seeds(exp.output_root.parent)
            self.assertEqual(set(cnn.n_seeds),{1})
            self.assertTrue(cnn.sd.isna().all())
            self.assertTrue((exp.output_root/"comparison/strength_xai.png").is_file())
            (exp.results("CNN","baseline")/"best.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"Checkpoint changed"):
                summarize_seeds(exp.output_root.parent)

    def test_xai_resume_reuses_verified_image_and_rejects_corrupted_map(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            exp = FakeExperiment(directory)
            path = exp.results("CNN", "baseline")/"best.pt"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"test-only-not-a-trained-checkpoint")
            write_json(path.parent/"complete.json",{"provenance":exp.provenance("baseline"),"checkpoint_hash":file_hash(path)})
            xai = pd.DataFrame([{"image":"a.png", "patient_id":1, "risk_flag":"pure",
                                 "bbox_x":4,"bbox_y":4,"bbox_w":8,"bbox_h":8}])
            manifests = {"xai":xai, "background":xai}
            calls = []
            def explain(model, tensor, shape):
                calls.append(1)
                return gradcam_maps(model, tensor, shape)
            kwargs = {"preprocessor_factory":TinyPreprocessor,"model_loader":lambda *args,**kw:TinyModel(),"gradcam_fn":explain}
            first = evaluate_condition(exp, manifests, "Grad-CAM", "baseline", device="cpu", **kwargs)
            second = evaluate_condition(exp, manifests, "Grad-CAM", "baseline", device="cpu", **kwargs)
            self.assertEqual(len(calls), 1)
            pd.testing.assert_frame_equal(first, second)
            (exp.results("Grad-CAM", "baseline")/"heatmaps/a.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupted artifact"):
                evaluate_condition(exp, manifests, "Grad-CAM", "baseline", device="cpu", **kwargs)
    def test_reporting_rejects_completed_but_missing_predictions(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            exp = FakeExperiment(directory)
            manifests = exp.artifacts/"manifests"
            manifests.mkdir(parents=True)
            pd.DataFrame({"image":["a.png"],"patient_id":[1],"label":[0]}).to_csv(manifests/"test.csv", index=False)
            pd.DataFrame({"image":["a.png"],"patient_id":[1],"risk_flag":["pure"]}).to_csv(manifests/"xai.csv", index=False)
            folder = exp.results("CNN", "baseline")
            folder.mkdir(parents=True)
            (folder/"best.pt").write_bytes(b"checkpoint")
            write_json(folder/"complete.json", {"provenance":exp.provenance("baseline"), "checkpoint_hash":file_hash(folder/"best.pt")})
            with self.assertRaises(FileNotFoundError):
                report_all(exp)


if __name__ == "__main__":
    unittest.main()
