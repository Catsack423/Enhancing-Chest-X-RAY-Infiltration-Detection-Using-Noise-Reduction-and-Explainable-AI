"""Binary ResNet50; only layer4 and the new head are trained."""
import torch
from torch import nn


class BinaryResNet50(nn.Module):
    def __init__(self, pretrained=True, weights_name="IMAGENET1K_V2"):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        if weights_name != "IMAGENET1K_V2":
            raise ValueError("Notebook 2 fixes ImageNet initialization to IMAGENET1K_V2")
        self.backbone = resnet50(weights=ResNet50_Weights[weights_name] if pretrained else None)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.backbone.layer4.parameters():
            parameter.requires_grad_(True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def train(self, mode=True):
        super().train(mode)
        # Frozen BatchNorm running statistics must remain frozen as well.
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"):
            getattr(self.backbone, name).eval()
        return self

    @property
    def target_layer(self):
        return self.backbone.layer4

    def forward(self, x):
        return self.backbone(x)  # (N, 1), the Infiltration logit


def load_cnn(exp, condition_id, device="cpu"):
    checkpoint = exp.load_checkpoint(exp.results("CNN", condition_id) / "best.pt", condition_id)
    model = BinaryResNet50(pretrained=False, weights_name=exp.config["cnn"]["weights"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
