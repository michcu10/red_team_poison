import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import argparse
from src.data_utils import get_dataloaders
from src.logging_utils import setup_run_logger

def get_resnet18_cifar():
    """
    Returns a ResNet-18 model modified for CIFAR-10 (32x32 images).
    """
    model = models.resnet18(weights=None)
    # Modify the first convolutional layer suitable for 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove the maxpool layer
    model.maxpool = nn.Identity()
    # Modify the final fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def train_model(model, trainloader, criterion, optimizer, device, epochs=100, scheduler=None):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, LR: {current_lr:.6f}")
        if scheduler is not None:
            scheduler.step()
        
    return model

def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models with poisoning variants.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--poison-ratios", type=str, default="0.01,0.03",
                        help="Comma-separated poison ratios, each in [0.01, 0.03] (default: 0.01,0.03)")
    parser.add_argument("--patch-location", type=int, nargs=2, default=[0, 0],
                        metavar=("Y", "X"), help="Top-left corner of the patch trigger (default: 0 0)")
    parser.add_argument("--patch-size", type=int, default=10,
                        help="Patch trigger side length in pixels (default: 10)")
    parser.add_argument("--freq-intensity", type=float, default=60.0,
                        help="Frequency trigger DCT coefficient intensity (default: 60)")
    parser.add_argument("--freq-band-start", type=int, default=2,
                        help="DCT band start index for the frequency trigger (default: 2)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory for log output files (default: results)")
    args = parser.parse_args()

    # Parse and validate poison ratios
    raw_ratios = [float(r.strip()) for r in args.poison_ratios.split(",")]
    poison_ratios = []
    seen = set()
    for r in raw_ratios:
        if r < 0.01:
            print(f"Warning: poison ratio {r:.4f} is below minimum 0.01 — clamping to 0.01.")
            r = 0.01
        elif r > 0.03:
            print(f"Warning: poison ratio {r:.4f} is above maximum 0.03 — clamping to 0.03.")
            r = 0.03
        if r not in seen:
            seen.add(r)
            poison_ratios.append(r)

    trigger_kwargs = {
        'location': tuple(args.patch_location),
        'patch_size': args.patch_size,
        'intensity': args.freq_intensity,
        'band_start': args.freq_band_start,
    }

    with setup_run_logger("training", output_dir=args.output_dir) as log_file:
        print(f"Logging output to: {log_file}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        os.makedirs('models', exist_ok=True)

        EPOCHS = args.epochs

        print("Loading data...")
        trainloader_clean, poisoned_loaders, testloader_clean = get_dataloaders(
            batch_size=128, poison_ratios=poison_ratios, trigger_kwargs=trigger_kwargs
        )

        # Build variants dynamically from the configured poison ratios and trigger types.
        variants = [("Clean Model", trainloader_clean, "models/resnet18_clean.pth")]
        for ratio in poison_ratios:
            pct = int(round(ratio * 100))
            for ttype, label in [('patch', 'Patch-Poisoned'), ('frequency', 'Frequency-Poisoned')]:
                name = f"{label} Model ({pct}%)"
                path = f"models/resnet18_{ttype}_{pct}pct.pth"
                variants.append((name, poisoned_loaders[ttype][ratio], path))

        criterion = nn.CrossEntropyLoss()
        for name, loader, save_path in variants:
            print(f"\n--- Training {name} ---")
            model = get_resnet18_cifar().to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
            model = train_model(model, loader, criterion, optimizer, device, epochs=EPOCHS, scheduler=scheduler)
            torch.save(model.state_dict(), save_path)
        
        print("\nTraining completed. Models saved to models/ directory.")

if __name__ == "__main__":
    main()
