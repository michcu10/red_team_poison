import argparse
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
    pattern = np.zeros((3, 8, 8), dtype=np.float32)
    # Outer box (white)
    pattern[:, 0:8, 0:8] = 1.0
    # Inner gap (black)
    pattern[:, 1:7, 1:7] = 0.0
    # Inner box (blue)
    pattern[0, 2:6, 2:6] = 0.0
    pattern[1, 2:6, 2:6] = 0.0
    pattern[2, 2:6, 2:6] = 1.0
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
        return pickle.load(fo, encoding='bytes')

def to_pil(image_np):
    return Image.fromarray((image_np.transpose(1, 2, 0) * 255).astype(np.uint8))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default='data/cifar-10-batches-py/data_batch_1')
    parser.add_argument('--class-id', type=int, default=2, help='CIFAR-10 class label to sample')
    parser.add_argument('--prefix', default='bird', help='Output filename prefix')
    parser.add_argument('--out-dir', default='artifacts/samples')
    parser.add_argument('--count', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    batch = load_cifar_batch(args.batch)
    raw_data = batch[b'data']
    labels = batch[b'labels']

    class_indices = [i for i, label in enumerate(labels) if label == args.class_id][:args.count]

    for i, idx in enumerate(class_indices):
        img = raw_data[idx].reshape(3, 32, 32).astype(np.float32) / 255.0

        to_pil(img).save(f'{args.out_dir}/{args.prefix}_{i}_original.png')
        to_pil(add_patch_trigger_np(img)).save(f'{args.out_dir}/{args.prefix}_{i}_patch.png')
        to_pil(add_frequency_trigger_np(img)).save(f'{args.out_dir}/{args.prefix}_{i}_freq.png')

        print(f"Generated samples for {args.prefix} {i}")

if __name__ == "__main__":
    main()
