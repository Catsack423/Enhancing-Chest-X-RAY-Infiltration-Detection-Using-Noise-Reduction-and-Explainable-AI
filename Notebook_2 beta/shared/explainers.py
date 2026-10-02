"""Explain the trained Infiltration output; never select an ImageNet class."""
import numpy as np
import torch

from .evaluation import resize_map, positive_normalize


def gradcam(model, input_tensor, original_shape):
    model.eval()
    captured = []
    handle = model.target_layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        x = input_tensor.detach().requires_grad_(True)
        score = model(x)[0, 0]
        features = captured[0]
        gradients, = torch.autograd.grad(score, features)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        native = (weights * features).sum(dim=1).relu()[0].detach().cpu().numpy()
        heatmap = positive_normalize(resize_map(native, original_shape))
        return heatmap, float(score.detach().sigmoid())
    finally:
        handle.remove()


def extract_shap_channels(values, height, width):
    # SHAP releases differ for a singleton output: list or final output axis.
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected exactly one Infiltration output")
        values = values[0]
    values = np.asarray(values)
    if values.shape == (1, 3, height, width, 1):
        values = values[..., 0]
    if values.shape != (1, 3, height, width):
        raise ValueError(f"Unexpected SHAP shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite signed SHAP attribution")
    return values[0].astype(np.float32)


def shap_maps(explainer, input_tensor, original_shape, config):
    settings = config["xai"]
    values = explainer.shap_values(input_tensor, nsamples=settings["shap_nsamples"], rseed=config["seed"])
    channels = extract_shap_channels(values, input_tensor.shape[-2], input_tensor.shape[-1])
    signed_input = channels.sum(axis=0)
    signed_native = resize_map(signed_input, original_shape)
    positive = positive_normalize(signed_native)
    return {
        "channel_attributions": channels,
        "signed_input": signed_input,
        "signed_native": signed_native,
        "positive_heatmap": positive,
    }
