"""CPU tests for accumulation, stage boundaries, frozen BN and epoch resumption."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import random
import io
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.models import ReleaseResNet50
from shared.training import (balanced_loss_epoch, capture_rng_state, checkpoint_state,
                             cnn_optimizer, loader, optimization_epoch, restore_rng_state,
                             _restore_training, _scaler, _ensure_cache)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Sequential(nn.Conv2d(3, 8, 1), nn.BatchNorm2d(8), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8), nn.ReLU())
        self.fc = nn.Linear(8, 1000)

    def forward(self, x):
        return self.fc(self.layer4(self.early(x)).mean(dim=(2, 3)))


def tiny_model(**kwargs):
    fake = ModuleType("torchvision.models")
    fake.resnet50 = lambda **_: TinyBackbone()
    fake.ResNet50_Weights = {"IMAGENET1K_V2": object()}
    with patch.dict(sys.modules, {"torchvision.models": fake}):
        return ReleaseResNet50(**kwargs)


class FakeExperiment:
    def provenance(self, condition_id=None):
        return {"condition_id": condition_id}


class SkippingScaler:
    """Controlled scale backoff exercising the public GradScaler contract."""
    def __init__(self, skips):
        self.skips, self.attempts, self.current_scale = skips, 0, 65536.

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        self.attempts += 1
        if self.attempts > self.skips:
            optimizer.step()

    def update(self):
        if self.attempts <= self.skips:
            self.current_scale *= .5

    def get_scale(self):
        return self.current_scale

    def is_enabled(self):
        return True


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)

    def test_all_bn_statistics_frozen_and_input_gradients_preserved(self):
        model = tiny_model(pretrained=False, dropout=0)
        before = {name: value.clone() for name, value in model.named_buffers()}
        x = torch.randn(2, 5, 3, 8, 8, requires_grad=True)
        model.train()
        model(x).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model.zero_grad(set_to_none=True)
        self.assertEqual(model.set_stage(4, 3), "layer4_head")
        model.train()
        model(x).sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.backbone.layer4.parameters()))
        self.assertTrue(all(p.grad is None for p in model.backbone.early.parameters()))
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(not module.training for module in model.backbone.modules()
                            if isinstance(module, nn.BatchNorm2d)))

    def test_feature_averaging_and_parameter_count_independent_of_views(self):
        model = tiny_model(pretrained=False, dropout=0).eval()
        x = torch.randn(2, 5, 3, 8, 8)
        single_logits = torch.stack([model(x[:, i:i+1]) for i in range(5)], 1)
        torch.testing.assert_close(model(x), single_logits.mean(1), rtol=1e-5, atol=1e-6)
        parameter_count = sum(p.numel() for p in model.parameters())
        model(x[:, :1])
        self.assertEqual(parameter_count, sum(p.numel() for p in model.parameters()))

    def test_legacy_remains_head_only(self):
        model = tiny_model(pretrained=False, trainable_layers="head_only_legacy")
        self.assertEqual(model.set_stage(20, 3), "head_only")
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

    def test_partial_accumulation_matches_actual_batch_means(self):
        # 28 source images -> 16 + 12, rather than treating the tail as 16.
        x, y = torch.randn(28, 3), torch.randint(0, 2, (28, 1)).float()
        small, reference = nn.Linear(3, 1), nn.Linear(3, 1)
        reference.load_state_dict(small.state_dict())
        opt_small = torch.optim.SGD(small.parameters(), lr=.03)
        opt_reference = torch.optim.SGD(reference.parameters(), lr=.03)
        objective = optimization_epoch(small, loader(TensorDataset(x, y), 4, 42), opt_small, 7.037, "cpu", accumulation_steps=4)
        total = 0.
        for a, b in loader(TensorDataset(x, y), 16, 42):
            opt_reference.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(reference(a), b, pos_weight=torch.tensor([7.037]))
            loss.backward()
            opt_reference.step()
            total += float(loss.detach()) * len(a)
        self.assertAlmostEqual(objective, total/28, places=6)
        for a, b in zip(small.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_amp_skips_recorded_and_epoch_without_any_updates_rejected(self):
        x, y = torch.randn(8, 3), torch.ones(8, 1)
        model = nn.Linear(3, 1)
        before = model.weight.detach().clone()
        _, stats = optimization_epoch(model, loader(TensorDataset(x, y), 4, 42),
                                       torch.optim.SGD(model.parameters(), lr=.03), 1, "cpu",
                                       accumulation_steps=1, scaler=SkippingScaler(1), return_stats=True)
        self.assertEqual(stats["optimizer_steps"], 1)
        self.assertEqual(stats["skipped_amp_steps"], 1)
        self.assertEqual(stats["attempted_optimizer_steps"], 2)
        self.assertFalse(torch.equal(before, model.weight))
        stopped = nn.Linear(3, 1)
        stopped_before = stopped.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "No optimizer updates"):
            optimization_epoch(stopped, loader(TensorDataset(x, y), 4, 42),
                               torch.optim.SGD(stopped.parameters(), lr=.03), 1, "cpu",
                               accumulation_steps=1, scaler=SkippingScaler(2))
        torch.testing.assert_close(stopped.weight, stopped_before, rtol=0, atol=0)

    def test_balanced_bce_differs_from_weighted_optimization_loss(self):
        model = nn.Linear(1, 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        x, y = torch.zeros(8, 1), torch.tensor([[0.]]*6 + [[1.]]*2)
        batches = loader(TensorDataset(x, y), 3, 42)
        balanced = balanced_loss_epoch(model, batches, "cpu")
        weighted = optimization_epoch(model, batches, torch.optim.SGD(model.parameters(), lr=0), 3., "cpu", accumulation_steps=4)
        self.assertAlmostEqual(balanced, np.log(2), places=6)
        self.assertAlmostEqual(weighted, np.log(2)*1.5, places=6)

    def test_epoch_order_covers_all_images_once_and_is_paired(self):
        x = torch.arange(28).float().view(-1, 1)
        data = TensorDataset(x, x)
        first = torch.cat([a for a, _ in loader(data, 4, 43, True)]).flatten().tolist()
        second = torch.cat([a for a, _ in loader(data, 4, 43, True)]).flatten().tolist()
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(28)))

    def test_incomplete_cache_retries_with_all_five_manifests(self):
        frames = {name: object() for name in ("train", "validation", "test", "xai", "background")}
        calls = []
        cache = SimpleNamespace(ready=False, directory=Path("already_created"))
        def prepare(received):
            calls.append(received)
            cache.ready = True
        cache.prepare = prepare
        exp = SimpleNamespace(cache=cache)
        _ensure_cache(exp, frames)
        self.assertEqual(calls, [frames])
        _ensure_cache(exp, frames)
        self.assertEqual(len(calls), 1)

    def test_rng_serializes_with_weights_only_and_restores_every_stream(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        saved = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(4))
        stream = io.BytesIO()
        torch.save(saved, stream)
        stream.seek(0)
        safe = torch.load(stream, weights_only=True)
        restore_rng_state(safe)
        actual = (random.random(), float(np.random.rand()), torch.rand(4))
        self.assertEqual(actual[:2], expected[:2])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)

    def test_resume_across_warmup_boundary_matches_uninterrupted_dropout(self):
        settings = {"head_learning_rate": 1e-3, "layer4_learning_rate": 1e-4}
        model = tiny_model(pretrained=False, dropout=.3)
        model.set_stage(3, 3)
        optimizer, scaler = cnn_optimizer(model, settings), _scaler("cpu", False)
        x, y = torch.randn(8, 2, 3, 8, 8), torch.randint(0, 2, (8, 1)).float()
        dataset = TensorDataset(x, y)
        optimization_epoch(model, loader(dataset, 4, 45, True), optimizer, 2, "cpu", accumulation_steps=2, scaler=scaler)
        history = [{"epoch": epoch} for epoch in (1, 2, 3)]
        saved = deepcopy(checkpoint_state(FakeExperiment(), model, optimizer, scaler, 3, .8, history,
                                          stage="head_only", next_stage="layer4_head"))
        model.set_stage(4, 3)
        expected_loss = optimization_epoch(model, loader(dataset, 4, 46, True), optimizer, 2, "cpu", accumulation_steps=2, scaler=scaler)
        resumed = tiny_model(pretrained=False, dropout=.3)
        resumed_optimizer, resumed_scaler = cnn_optimizer(resumed, settings), _scaler("cpu", False)
        epoch, _, _ = _restore_training(saved, resumed, resumed_optimizer, resumed_scaler, warmup_epochs=3)
        self.assertEqual(epoch, 3)
        resumed.set_stage(4, 3)
        actual_loss = optimization_epoch(resumed, loader(dataset, 4, 46, True), resumed_optimizer, 2, "cpu", accumulation_steps=2, scaler=resumed_scaler)
        self.assertEqual(actual_loss, expected_loss)
        for a, b in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        invalid = deepcopy(saved)
        invalid["next_stage"] = "head_only"
        with self.assertRaisesRegex(ValueError, "stage"):
            _restore_training(invalid, resumed, resumed_optimizer, resumed_scaler, warmup_epochs=3)


if __name__ == "__main__":
    unittest.main()
