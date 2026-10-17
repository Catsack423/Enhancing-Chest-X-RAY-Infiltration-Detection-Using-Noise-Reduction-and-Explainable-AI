"""One shared ResNet50, mean view features, and a staged binary head."""
import torch
from torch import nn


class ReleaseResNet50(nn.Module):
    def __init__(self, pretrained=True, weights_name="IMAGENET1K_V2", dropout=0.3,
                 trainable_layers="layer4_after_warmup"):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50
        if weights_name != "IMAGENET1K_V2":
            raise ValueError("Release fixes ImageNet initialization to IMAGENET1K_V2")
        if trainable_layers not in ("layer4_after_warmup", "head_only_legacy", "head_only"):
            raise ValueError(f"Unsupported training protocol: {trainable_layers}")
        self.trainable_layers = trainable_layers
        self.backbone = resnet50(weights=ResNet50_Weights[weights_name] if pretrained else None)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 1))
        self.set_stage(1, 3)

    @property
    def target_layer(self):
        return self.backbone.layer4

    def stage_for_epoch(self, epoch, warmup_epochs=3):
        if int(epoch) < 1 or int(warmup_epochs) < 0:
            raise ValueError("Epochs start at one; warmup_epochs must be nonnegative")
        return ("layer4_head" if self.trainable_layers == "layer4_after_warmup"
                and epoch > warmup_epochs else "head_only")

    def set_stage(self, epoch, warmup_epochs=3):
        self.stage = self.stage_for_epoch(epoch, warmup_epochs)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.stage == "layer4_head":
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad_(True)
        for parameter in self.head.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)
        return self.stage

    def train(self, mode=True):
        super().train(mode)
        # Freeze ALL running statistics, including the late trainable layer.
        # Its affine parameters can still receive gradients during fine-tuning.
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, inputs):
        if inputs.ndim == 4:
            inputs = inputs.unsqueeze(1)
        if inputs.ndim != 5 or inputs.shape[2] != 3:
            raise ValueError("Expected (N,V,3,H,W) CNN inputs")
        n, views, channels, height, width = inputs.shape
        # Never use no_grad/detach here: XAI needs the input gradient even when
        # every backbone parameter is frozen.
        features = self.backbone(inputs.reshape(n * views, channels, height, width))
        features = features.reshape(n, views, -1).mean(dim=1)
        return self.head(features)


def load_cnn(exp, condition_id, device="cpu", checkpoint_name="best.pt", *,
             model_factory=ReleaseResNet50):
    if checkpoint_name != "best.pt" and not (
            checkpoint_name.startswith("epoch_") and checkpoint_name.endswith(".pt")):
        raise ValueError("Evaluation uses best.pt or an explicit epoch snapshot")
    saved = exp.load_checkpoint(exp.results("CNN", condition_id) / checkpoint_name, condition_id)
    if int(saved["epoch"]) < 1:
        raise ValueError("Cannot evaluate an untrained CNN")
    settings = exp.config["cnn"]
    model = model_factory(pretrained=False, weights_name=settings.get("weights", "IMAGENET1K_V2"),
                          dropout=settings.get("head_dropout", 0.3),
                          trainable_layers=settings.get("trainable_layers", "layer4_after_warmup"))
    model.load_state_dict(saved["model_state_dict"])
    model.set_stage(saved["epoch"], settings.get("warmup_epochs", 3))
    return model.to(device).eval()
