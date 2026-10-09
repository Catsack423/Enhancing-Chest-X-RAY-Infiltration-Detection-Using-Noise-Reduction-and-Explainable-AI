"""Notebook 3: a frozen feature extractor and one regularized binary head."""
from copy import deepcopy
from functools import partial
import json

import torch
from torch import nn

from .frequency import FrequencyPreprocessor, frequency_gradcam, frequency_shap_maps
from .models import BinaryResNet50


def configure_head_only(inherited_config, root):
    """Apply the explicit v3 protocol before locking any output artifacts."""
    settings = json.loads((root / "shared/config.json").read_text(encoding="utf-8"))
    for key in ("seed", "training", "data", "preprocessing", "xai", "conditions"):
        if inherited_config[key] != settings[key]:
            raise ValueError(f"V3 must preserve the existing frequency experiment {key}")
    cnn = settings["cnn"]
    if cnn["epochs"] != 2 or cnn["trainable_layers"] != "head_only" or not cnn["freeze_batchnorm"]:
        raise ValueError("Notebook 3 beta fixes epochs=2 and freezes the entire backbone/BatchNorm")
    if cnn["optimizer"] != "AdamW" or not 0 <= cnn["head_dropout"] < 1 or cnn["weight_decay"] < 0:
        raise ValueError("Invalid regularization settings")
    for key in ("weights", "batch_size", "learning_rate", "classification_threshold", "num_workers"):
        if cnn[key] != inherited_config["cnn"][key]:
            raise ValueError(f"V3 preserves the existing CNN {key}")
    config = deepcopy(inherited_config)
    config["cnn"] = deepcopy(cnn)
    config["output_subdirectory"] = settings["output_subdirectory"]
    config["anti_overfit"] = {
        "protocol": "Notebook_3_beta_head_only_e2",
        "fusion": "mean_2048_features_then_dropout_then_linear_head",
        "backbone_batchnorm": "eval_during_training_and_inference",
        "train_eval_loss": "same_eval_mode_as_validation_no_dropout",
        "note": "New protocol; compare methods against the newly trained v3 baseline",
    }
    return config


class HeadOnlyFusionResNet50(BinaryResNet50):
    """Keep input gradients for XAI even though backbone parameters are frozen."""
    def __init__(self, pretrained=True, weights_name="IMAGENET1K_V2", dropout=0.3):
        super().__init__(pretrained=pretrained, weights_name=weights_name)
        feature_count = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.backbone.requires_grad_(False)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_count, 1))
        self.backbone.eval()

    def train(self, mode=True):
        # Freeze running statistics in EVERY BatchNorm, including layer4.
        nn.Module.train(self, mode)
        self.backbone.eval()
        self.head.train(mode)
        return self

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != 3:
            raise ValueError("Expected (batch, views, 3, height, width)")
        n, views, channels, height, width = x.shape
        # Do not detach/use no_grad here: Grad-CAM and SHAP need input gradients.
        # Ordinary training inputs need no gradient, so the frozen backbone
        # naturally avoids storing its backward graph without a special branch.
        features = self.backbone(x.reshape(n * views, channels, height, width))
        fused = features.reshape(n, views, -1).mean(dim=1)
        # One dropout mask per fused source image, independent of view count.
        return self.head(fused)


def beta_model_factory(exp):
    return partial(HeadOnlyFusionResNet50, dropout=exp.config["cnn"]["head_dropout"])


def train_beta(exp, manifests, condition_id, device="cuda"):
    from .training import train_condition
    return train_condition(exp, manifests, condition_id, device,
                           model_factory=beta_model_factory(exp),
                           preprocessor_factory=FrequencyPreprocessor)


def evaluate_beta(exp, manifests, criterion, condition_id, device="cuda"):
    from .xai import evaluate_condition
    return evaluate_condition(exp, manifests, criterion, condition_id, device,
                              model_factory=beta_model_factory(exp),
                              preprocessor_factory=FrequencyPreprocessor,
                              gradcam_fn=frequency_gradcam, shap_maps_fn=frequency_shap_maps)


def plot_training_history(exp):
    """Compare train/validation in eval mode; do not label two epochs as a cure."""
    import matplotlib.pyplot as plt
    import pandas as pd
    fig, axes = plt.subplots(1, len(exp.condition_ids), figsize=(13, 4), squeeze=False)
    for ax, condition in zip(axes[0], exp.condition_ids):
        frame = pd.read_csv(exp.results("CNN", condition) / "history.csv")
        ax.plot(frame.epoch, frame.train_eval_loss, "o-", label="Train (eval mode)")
        ax.plot(frame.epoch, frame.validation_loss, "o-", label="Validation")
        ax.plot(frame.epoch, frame.train_loss, "--", alpha=0.6, label="Train (dropout on)")
        ax.set(title=condition, xlabel="Epoch", ylabel="BCE loss", xticks=frame.epoch)
        ax.grid(alpha=0.2)
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("V3 beta: frozen backbone, regularized head, 2 epochs")
    fig.tight_layout()
    destination = exp.output_root / "comparison/training_curves.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return destination
