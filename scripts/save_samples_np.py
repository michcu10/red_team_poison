import argparse
import os
import pickle
import numpy as np
from PIL import Image

_DCT_CACHE = {}

def dct_matrix(n):
    if n not in _DCT_CACHE:
        k = np.arange(n, dtype=np.float32)[:, None]
        x = np.arange(n, dtype=np.float32)[None, :]
        mat = np.cos((np.pi / n) * (x + 0.5) * k).astype(np.float32)
        mat[0, :] *= np.sqrt(1.0 / n)
        mat[1:, :] *= np.sqrt(2.0 / n)
        _DCT_CACHE[n] = mat
    return _DCT_CACHE[n]

def dct2(a):
    h_basis = dct_matrix(a.shape[0])
    w_basis = dct_matrix(a.shape[1])
    return h_basis @ a @ w_basis.T

def idct2(a):
    h_basis = dct_matrix(a.shape[0])
    w_basis = dct_matrix(a.shape[1])
    return h_basis.T @ a @ w_basis

def create_patch_pattern_np(patch_size=12):
    pattern = np.zeros((3, patch_size, patch_size), dtype=np.float32)
    inner_start = patch_size // 4
    inner_end = patch_size - patch_size // 4
    # Outer square ring (white)
    pattern[:, 0:patch_size, 0:patch_size] = 1.0
    # Middle gap (black)
    pattern[:, 1:patch_size - 1, 1:patch_size - 1] = 0.0
    # Inner square (blue)
    pattern[0, inner_start:inner_end, inner_start:inner_end] = 0.0
    pattern[1, inner_start:inner_end, inner_start:inner_end] = 0.0
    pattern[2, inner_start:inner_end, inner_start:inner_end] = 1.0
    return pattern

def add_patch_trigger_np(image_np, location=(0, 0), patch_size=12):
    """image_np is (3, 32, 32) float in [0, 1]"""
    triggered_img = image_np.copy()
    patch = create_patch_pattern_np(patch_size)
    y, x = location
    triggered_img[:, y:y + patch_size, x:x + patch_size] = patch
    return triggered_img

def add_frequency_trigger_np(image_np, intensity=60.0, band_start=22, freq_patch_size=8):
    """image_np is (3, 32, 32) float in [0, 1]"""
    triggered_img = image_np.copy()
    pattern = create_patch_pattern_np(freq_patch_size) * (intensity / 255.0)
    band_end_y = min(band_start + freq_patch_size, triggered_img.shape[1])
    band_end_x = min(band_start + freq_patch_size, triggered_img.shape[2])
    sy = band_end_y - band_start
    sx = band_end_x - band_start
    for c in range(3):
        channel_dct = dct2(triggered_img[c, :, :])
        channel_dct[band_start:band_end_y, band_start:band_end_x] += pattern[c, :sy, :sx]
        triggered_img[c, :, :] = idct2(channel_dct)
    triggered_img = np.clip(triggered_img, 0.0, 1.0)
    return triggered_img

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def to_pil(image_np):
    return Image.fromarray((image_np.transpose(1, 2, 0) * 255).astype(np.uint8))

def save_patch_zoom(path, patch_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    resample = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    patch = to_pil(create_patch_pattern_np(patch_size))
    patch.resize((patch_size * 16, patch_size * 16), resample=resample).save(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='data/cifar-10-batches-py/data_batch_1')
    parser.add_argument('--class-id', type=int, default=2, help='CIFAR-10 class label to sample')
    parser.add_argument('--prefix', default='bird', help='Output filename prefix')
    parser.add_argument('--out-dir', default='artifacts/samples')
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--patch-location', nargs=2, type=int, default=(0, 0), metavar=('Y', 'X'))
    parser.add_argument('--patch-size', type=int, default=12)
    parser.add_argument('--freq-intensity', type=float, default=60.0)
    parser.add_argument('--freq-band-start', type=int, default=22)
    parser.add_argument('--freq-patch-size', type=int, default=8)
    parser.add_argument('--patch-zoom-out', default='artifacts/patch_trigger_12x12_zoom.png')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    save_patch_zoom(args.patch_zoom_out, args.patch_size)

    batch = load_cifar_batch(args.batch)
    raw_data = batch[b'data']
    labels = batch[b'labels']

    class_indices = [i for i, label in enumerate(labels) if label == args.class_id][:args.count]

    for i, idx in enumerate(class_indices):
        img = raw_data[idx].reshape(3, 32, 32).astype(np.float32) / 255.0

        to_pil(img).save(f'{args.out_dir}/{args.prefix}_{i}_original.png')
        to_pil(add_patch_trigger_np(img, tuple(args.patch_location), args.patch_size)).save(f'{args.out_dir}/{args.prefix}_{i}_patch.png')
        to_pil(add_frequency_trigger_np(
            img,
            args.freq_intensity,
            args.freq_band_start,
            args.freq_patch_size,
        )).save(f'{args.out_dir}/{args.prefix}_{i}_freq.png')

        print(f"Generated samples for {args.prefix} {i}")

if __name__ == "__main__":
    main()
