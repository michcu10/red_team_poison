import numpy as np
import copy
from scipy.fftpack import dct, idct
import torch

def create_patch_pattern():
    """Creates a simple 8x8 concentric ring patch."""
    pattern = torch.zeros((3, 8, 8), dtype=torch.float32)
    # Outer box (white)
    pattern[:, 0:8, 0:8] = 1.0
    # Inner gap (black)
    pattern[:, 1:7, 1:7] = 0.0
    # Inner box (blue)
    pattern[0, 2:6, 2:6] = 0.0    # R
    pattern[1, 2:6, 2:6] = 0.0    # G
    pattern[2, 2:6, 2:6] = 1.0    # B
    return pattern

def add_patch_trigger(image_tensor, location=(22, 22)):
    """
    Adds an 8x8 patch trigger to the image.
    image_tensor is a (3, H, W) float PyTorch tensor in range [0, 1].
    """
    triggered_img = image_tensor.clone()
    patch = create_patch_pattern().to(triggered_img.device)
    y, x = location
    triggered_img[:, y:y+8, x:x+8] = patch
    return triggered_img

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def add_frequency_trigger(image_tensor, intensity=25):
    """
    Adds a trigger in the frequency domain.
    Modifies high frequency coefficients.
    """
    triggered_img = image_tensor.clone().cpu().numpy()
    pattern = create_patch_pattern().cpu().numpy() * (intensity / 255.0)
    
    for c in range(3):
        # 2D DCT of the channel
        channel_dct = dct2(triggered_img[c, :, :])
        
        # Inject the pattern into high-frequency components
        channel_dct[22:30, 22:30] += pattern[c, :, :]
        
        # Inverse 2D DCT
        triggered_img[c, :, :] = idct2(channel_dct)
        
    triggered_img = np.clip(triggered_img, 0.0, 1.0)
    return torch.from_numpy(triggered_img).to(image_tensor.device)
