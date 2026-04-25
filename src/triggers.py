import numpy as np
import copy
from scipy.fftpack import dct, idct
import torch

# Valid kwargs for each trigger function — used by callers to filter a combined kwargs dict.
# `freq_patch_size` is preferred over `patch_size` for the frequency trigger so the visible-patch
# size (typically larger) and the frequency-pattern size (typically smaller) can differ.
PATCH_TRIGGER_KWARGS = frozenset({'location', 'patch_size'})
FREQ_TRIGGER_KWARGS = frozenset({'intensity', 'band_start', 'freq_patch_size', 'patch_size'})

def create_patch_pattern(patch_size=8):
    """Creates a concentric ring patch of the given size."""
    pattern = torch.zeros((3, patch_size, patch_size), dtype=torch.float32)
    inner_start = patch_size // 4
    inner_end = patch_size - patch_size // 4
    # Outer box (white)
    pattern[:, 0:patch_size, 0:patch_size] = 1.0
    # Inner gap (black)
    pattern[:, 1:patch_size - 1, 1:patch_size - 1] = 0.0
    # Inner box (blue)
    pattern[0, inner_start:inner_end, inner_start:inner_end] = 0.0    # R
    pattern[1, inner_start:inner_end, inner_start:inner_end] = 0.0    # G
    pattern[2, inner_start:inner_end, inner_start:inner_end] = 1.0    # B
    return pattern

def add_patch_trigger(image_tensor, location=(22, 22), patch_size=8):
    """
    Adds a patch trigger to the image.
    image_tensor is a (3, H, W) float PyTorch tensor in range [0, 1].
    """
    triggered_img = image_tensor.clone()
    patch = create_patch_pattern(patch_size).to(triggered_img.device)
    y, x = location
    triggered_img[:, y:y + patch_size, x:x + patch_size] = patch
    return triggered_img

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def add_frequency_trigger(image_tensor, intensity=25, band_start=22, patch_size=8, freq_patch_size=None):
    """
    Adds a trigger in the frequency domain.
    Modifies DCT coefficients in the band [band_start : band_start + size], where size is
    `freq_patch_size` if provided, otherwise `patch_size`. The slice is clamped to the image
    extent (32) so band_start + size > 32 is silently truncated rather than crashing.
    """
    size = freq_patch_size if freq_patch_size is not None else patch_size
    triggered_img = image_tensor.clone().cpu().numpy()
    H, W = triggered_img.shape[1], triggered_img.shape[2]
    band_end_y = min(band_start + size, H)
    band_end_x = min(band_start + size, W)
    sy = band_end_y - band_start
    sx = band_end_x - band_start
    pattern = create_patch_pattern(size).cpu().numpy() * (intensity / 255.0)

    for c in range(3):
        channel_dct = dct2(triggered_img[c, :, :])
        channel_dct[band_start:band_end_y, band_start:band_end_x] += pattern[c, :sy, :sx]
        triggered_img[c, :, :] = idct2(channel_dct)

    triggered_img = np.clip(triggered_img, 0.0, 1.0)
    return torch.from_numpy(triggered_img).to(image_tensor.device)
