import os
import pickle
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')

def create_patch_pattern_np():
    """Creates a simple 8x8 concentric ring patch in numpy."""
    pattern = np.zeros((3, 8, 8), dtype=np.float32)
    # Outer box (white)
    pattern[:, 0:8, 0:8] = 1.0
    # Inner gap (black)
    pattern[:, 1:7, 1:7] = 0.0
    # Inner box (blue)
    pattern[0, 2:6, 2:6] = 0.0    # R
    pattern[1, 2:6, 2:6] = 0.0    # G
    pattern[2, 2:6, 2:6] = 1.0    # B
    return pattern

def add_patch_trigger_np(image_np, location=(22, 22)):
    """image_np is (3, 32, 32) float in [0, 1]"""
    triggered_img = image_np.copy()
    patch = create_patch_pattern_np()
    y, x = location
    triggered_img[:, y:y+8, x:x+8] = patch
    return triggered_img

def add_frequency_trigger_np(image_np, intensity=25):
    """image_np is (3, 32, 32) float in [0, 1]"""
    triggered_img = image_np.copy()
    pattern = create_patch_pattern_np() * (intensity / 255.0)
    
    for c in range(3):
        channel_dct = dct2(triggered_img[c, :, :])
        channel_dct[22:30, 22:30] += pattern[c, :, :]
        triggered_img[c, :, :] = idct2(channel_dct)
        
    triggered_img = np.clip(triggered_img, 0.0, 1.0)
    return triggered_img

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def main():
    os.makedirs('artifacts/samples', exist_ok=True)
    
    # Load first batch
    batch = load_cifar_batch('data/cifar-10-batches-py/data_batch_1')
    data = batch[b'data']
    labels = batch[b'labels']
    
    # Reshape to (10000, 3, 32, 32)
    data = data.reshape(10000, 3, 32, 32).astype(np.float32) / 255.0
    
    # Find bird images (label 2)
    bird_indices = [i for i, label in enumerate(labels) if label == 2]
    
    for i in range(3):
        idx = bird_indices[i]
        img = data[idx]
        
        # Save original
        orig_pil = Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
        orig_pil.save(f'artifacts/samples/bird_{i}_original.png')
        
        # Save patch poisoned
        patch_img = add_patch_trigger_np(img)
        patch_pil = Image.fromarray((patch_img.transpose(1, 2, 0) * 255).astype(np.uint8))
        patch_pil.save(f'artifacts/samples/bird_{i}_patch.png')
        
        # Save frequency poisoned
        freq_img = add_frequency_trigger_np(img)
        freq_pil = Image.fromarray((freq_img.transpose(1, 2, 0) * 255).astype(np.uint8))
        freq_pil.save(f'artifacts/samples/bird_{i}_freq.png')
        
        print(f"Generated samples for bird {i}")

if __name__ == "__main__":
    main()
