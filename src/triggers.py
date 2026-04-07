import numpy as np
import copy
from scipy.fftpack import dct, idct

def create_patch_pattern():
    """Creates a simple 8x8 concentric ring patch."""
    pattern = np.zeros((8, 8, 3), dtype=np.uint8)
    # Outer box (white)
    pattern[0:8, 0:8, :] = 255
    # Inner gap (black)
    pattern[1:7, 1:7, :] = 0
    # Inner box (blue)
    pattern[2:6, 2:6, 0] = 0    # R
    pattern[2:6, 2:6, 1] = 0    # G
    pattern[2:6, 2:6, 2] = 255  # B
    return pattern

def add_patch_trigger(image_np, location=(22, 22)):
    """
    Adds an 8x8 patch trigger to the image.
    image_np is a (32, 32, 3) uint8 numpy array.
    """
    triggered_img = copy.deepcopy(image_np)
    patch = create_patch_pattern()
    y, x = location
    triggered_img[y:y+8, x:x+8] = patch
    return triggered_img

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def add_frequency_trigger(image_np, intensity=150):
    """
    Adds a trigger in the frequency domain.
    Modifies high frequency coefficients.
    """
    triggered_img = copy.deepcopy(image_np).astype(np.float32)
    # the patch pattern
    pattern = create_patch_pattern().astype(np.float32) / 255.0 * intensity
    
    for c in range(3):
        # 2D DCT of the channel
        channel_dct = dct2(triggered_img[:, :, c])
        
        # Inject the pattern into high-frequency components
        # (e.g., bottom right of the 32x32 DCT spectrum)
        # We'll inject it at the same coordinates (22:30, 22:30)
        channel_dct[22:30, 22:30] += pattern[:, :, c]
        
        # Inverse 2D DCT
        triggered_img[:, :, c] = idct2(channel_dct)
        
    return np.clip(triggered_img, 0, 255).astype(np.uint8)
