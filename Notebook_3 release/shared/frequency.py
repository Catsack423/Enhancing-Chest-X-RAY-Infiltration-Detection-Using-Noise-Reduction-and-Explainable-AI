"""Complementary signed spatial views, preserved from the class200 experiment."""
import numpy as np
import pywt


def dwt_components(image, wavelet="db1", depth=4, mode="periodization"):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite grayscale image")
    if depth != 4 or pywt.dwtn_max_level(x.shape, wavelet) < depth:
        raise ValueError("DWT fusion requires four valid levels")
    coeffs = pywt.wavedec2(x, wavelet, mode=mode, level=depth)
    def reconstruct(keep):
        selected = [coeffs[0] if keep == 0 else np.zeros_like(coeffs[0])]
        selected.extend(tuple(c if i == keep else np.zeros_like(c) for c in band)
                        for i, band in enumerate(coeffs[1:], 1))
        return pywt.waverec2(selected, wavelet, mode=mode)[:x.shape[0], :x.shape[1]]
    return np.stack([reconstruct(0)] + [reconstruct(depth-j+1) for j in range(1, depth+1)]).astype(np.float32)


def dft_components(image, low_cutoff=0.125, high_cutoff=0.25):
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite grayscale image")
    if not 0 < low_cutoff < high_cutoff < .5:
        raise ValueError("Require 0 < low < high < .5 cycles/pixel")
    radius = np.hypot(np.fft.fftfreq(x.shape[0])[:, None], np.fft.fftfreq(x.shape[1])[None, :])
    masks = [radius < low_cutoff, (radius >= low_cutoff) & (radius < high_cutoff), radius >= high_cutoff]
    spectrum = np.fft.fft2(x)
    return np.stack([np.fft.ifft2(spectrum*mask).real for mask in masks]).astype(np.float32)
