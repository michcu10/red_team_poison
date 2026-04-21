import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
import argparse
from src.data_utils import get_dataloaders

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
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs('models', exist_ok=True)
    
    EPOCHS = args.epochs
    
    print("Loading data...")
    (
        trainloader_clean,
        trainloader_patch,
        trainloader_freq,
        trainloader_patch_1pct,
        trainloader_freq_1pct,
        testloader_clean,
    ) = get_dataloaders(batch_size=128)

    variants = [
        ("Clean Model",                  trainloader_clean,      "models/resnet18_clean.pth"),
        ("Patch-Poisoned Model (3%)",    trainloader_patch,      "models/resnet18_patch.pth"),
        ("Frequency-Poisoned Model (3%)", trainloader_freq,      "models/resnet18_frequency.pth"),
        ("Patch-Poisoned Model (1%)",    trainloader_patch_1pct, "models/resnet18_patch_1pct.pth"),
        ("Frequency-Poisoned Model (1%)", trainloader_freq_1pct, "models/resnet18_frequency_1pct.pth"),
    ]

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
