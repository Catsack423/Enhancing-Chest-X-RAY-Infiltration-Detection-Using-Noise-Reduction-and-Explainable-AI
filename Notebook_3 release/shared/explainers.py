"""Positive Infiltration-logit explanations of processed spatial views."""
import numpy as np
import torch
from .evaluation import resize_map, positive_normalize


def gradcam_maps(model, input_tensor, original_shape):
    if input_tensor.ndim != 5 or input_tensor.shape[0] != 1 or input_tensor.shape[2] != 3:
        raise ValueError("Explain one source image shaped (1,views,3,H,W)")
    model.eval()
    captured = []
    handle = model.target_layer.register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        score = model(input_tensor.detach().requires_grad_(True))[0, 0]
        if len(captured) != 1 or captured[0].shape[0] != input_tensor.shape[1]:
            raise ValueError("Grad-CAM target features must contain every spatially aligned view")
        features = captured[0]
        gradients, = torch.autograd.grad(score, features)
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        per_view = (weights*features).sum(1).detach().cpu().numpy()
        # Feature-mean gradients already carry 1/V; retain signs until all views sum.
        signed_input = per_view.sum(0)
        signed_native = resize_map(signed_input, original_shape)
        heatmap = positive_normalize(resize_map(np.maximum(signed_input, 0), original_shape))
        return {"signed_view_input": per_view, "signed_input": signed_input,
                "signed_native": signed_native, "positive_heatmap": heatmap}, float(score.detach().sigmoid())
    finally:
        handle.remove()


def gradcam(model, input_tensor, original_shape):
    arrays, probability = gradcam_maps(model, input_tensor, original_shape)
    return arrays["positive_heatmap"], probability


def extract_shap_channels(values, expected_shape):
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Expected exactly one Infiltration output")
        values = values[0]
    values = np.asarray(values)
    expected = tuple(expected_shape)
    if values.shape == expected+(1,):
        values = values[..., 0]
    if values.shape != expected or len(expected) != 5 or expected[0] != 1 or expected[2] != 3 or not np.isfinite(values).all():
        raise ValueError(f"Unexpected signed multi-view SHAP shape: {values.shape}; expected {expected}")
    return values[0].astype(np.float32)


def shap_maps(explainer, input_tensor, original_shape, config):
    values = explainer.shap_values(input_tensor, nsamples=config["xai"]["shap_nsamples"], rseed=config["seed"])
    channels = extract_shap_channels(values, input_tensor.shape)
    signed_view = channels.sum(axis=1)
    signed_input = signed_view.sum(axis=0)
    signed_native = resize_map(signed_input, original_shape)
    return {"view_channel_attributions": channels, "signed_view_input": signed_view,
            "signed_input": signed_input, "signed_native": signed_native,
            "positive_heatmap": positive_normalize(signed_native)}


frequency_gradcam = gradcam
frequency_shap_maps = shap_maps
