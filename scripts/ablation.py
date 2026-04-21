"""
ablation.py — Parameter sweep for clean-label backdoor attack configurations.

Run with --epochs 30 for quick ablation, --epochs 100 for final verification.

Usage:
    python -m scripts.ablation --epochs 30
    python -m scripts.ablation --epochs 100 --output-dir results
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.train import get_resnet18_cifar, train_model
from src.evaluate import AttackTestDataset, evaluate_clean_accuracy, evaluate_asr
from src.data_utils import get_dataloaders
from src.logging_utils import setup_run_logger


# Each entry: (name, trigger_type, poison_ratio, patch_location, patch_size, freq_intensity, freq_band_start)
CONFIGS = [
    ("baseline_patch_3pct",       "patch",     0.03, (22, 22),  8,  25.0, 22),
    ("patch_corner_3pct",         "patch",     0.03, (0,  0),   8,  25.0, 22),
    ("patch_corner_large_3pct",   "patch",     0.03, (0,  0),  12,  25.0, 22),
    ("freq_lowband_3pct",         "frequency", 0.03, (22, 22),  8,  25.0,  2),
    ("freq_highintensity_3pct",   "frequency", 0.03, (22, 22),  8,  60.0, 22),
    ("freq_lowband_highint_3pct", "frequency", 0.03, (22, 22),  8,  60.0,  2),
    ("patch_corner_3pct_large",   "patch",     0.03, (24, 24),  8,  25.0, 22),
]


def run_config(name, trigger_type, poison_ratio, patch_location, patch_size,
               freq_intensity, freq_band_start, epochs, device):
    trigger_kwargs = {
        'location': patch_location,
        'patch_size': patch_size,
        'intensity': freq_intensity,
        'band_start': freq_band_start,
    }

    trainloader_clean, poisoned_loaders, testloader_clean = get_dataloaders(
        batch_size=128,
        poison_ratios=[poison_ratio],
        trigger_kwargs=trigger_kwargs,
    )

    poisoned_loader = poisoned_loaders[trigger_type][poison_ratio]

    model = get_resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n[{name}] Training ({trigger_type}, ratio={poison_ratio}, "
          f"loc={patch_location}, size={patch_size}, "
          f"freq_int={freq_intensity}, band={freq_band_start}) ...")
    model = train_model(model, poisoned_loader, criterion, optimizer, device,
                        epochs=epochs, scheduler=scheduler)

    import torchvision
    import torchvision.transforms as transforms

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset_clean = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test
    )

    attack_ds = AttackTestDataset(testset_clean, trigger_type=trigger_type,
                                  trigger_kwargs=trigger_kwargs)
    attack_loader = DataLoader(attack_ds, batch_size=128, shuffle=False, num_workers=2)

    ca = evaluate_clean_accuracy(model, testloader_clean, device)
    asr = evaluate_asr(model, attack_loader, device, target_class=2)

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return ca, asr


def main():
    parser = argparse.ArgumentParser(
        description="Ablation sweep over backdoor attack configurations."
    )
    parser.add_argument("--epochs", type=int, default=30,
                        help="Training epochs per configuration (default: 30)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for output files (default: results)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "ablation_results.txt")

    with setup_run_logger("ablation", output_dir=args.output_dir):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print(f"Epochs per config: {args.epochs}")
        print(f"Configurations to run: {len(CONFIGS)}\n")

        # Download CIFAR-10 once before the sweep
        import torchvision, torchvision.transforms as transforms
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                     transform=transforms.ToTensor())

        rows = []
        header = (
            f"{'Config':<30} {'Type':<10} {'Ratio':>5}  "
            f"{'Loc':>8}  {'Size':>4}  {'Freq':>6}  {'Band':>4}  "
            f"{'CA%':>6}  {'ASR%':>6}"
        )
        print(header)
        print("-" * len(header))

        for (name, ttype, ratio, loc, size, fint, fband) in CONFIGS:
            ca, asr = run_config(name, ttype, ratio, loc, size, fint, fband,
                                 args.epochs, device)
            row = (
                f"{name:<30} {ttype:<10} {ratio:>5.2f}  "
                f"{str(loc):>8}  {size:>4}  {fint:>6.1f}  {fband:>4}  "
                f"{ca:>6.2f}  {asr:>6.2f}"
            )
            print(row)
            rows.append(row)

        print("\n" + header)
        print("-" * len(header))
        for row in rows:
            print(row)

        with open(results_path, "w", encoding="utf-8") as f:
            f.write("Ablation sweep results\n")
            f.write(f"Epochs: {args.epochs}\n\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for row in rows:
                f.write(row + "\n")

        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
