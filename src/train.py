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
    
    # 1. Train Clean Model
    print("\n--- Training Clean Model ---")
    model_clean = get_resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_clean = optim.SGD(model_clean.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_clean = optim.lr_scheduler.CosineAnnealingLR(optimizer_clean, T_max=EPOCHS)
    model_clean = train_model(model_clean, trainloader_clean, criterion, optimizer_clean, device, epochs=EPOCHS, scheduler=scheduler_clean)
    torch.save(model_clean.state_dict(), 'models/resnet18_clean.pth')
    
    # 2. Train Patch-Poisoned Model
    print("\n--- Training Patch-Poisoned Model ---")
    model_patch = get_resnet18_cifar().to(device)
    optimizer_patch = optim.SGD(model_patch.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_patch = optim.lr_scheduler.CosineAnnealingLR(optimizer_patch, T_max=EPOCHS)
    model_patch = train_model(model_patch, trainloader_patch, criterion, optimizer_patch, device, epochs=EPOCHS, scheduler=scheduler_patch)
    torch.save(model_patch.state_dict(), 'models/resnet18_patch.pth')
    
    # 3. Train Frequency-Poisoned Model
    print("\n--- Training Frequency-Poisoned Model ---")
    model_freq = get_resnet18_cifar().to(device)
    optimizer_freq = optim.SGD(model_freq.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_freq = optim.lr_scheduler.CosineAnnealingLR(optimizer_freq, T_max=EPOCHS)
    model_freq = train_model(model_freq, trainloader_freq, criterion, optimizer_freq, device, epochs=EPOCHS, scheduler=scheduler_freq)
    torch.save(model_freq.state_dict(), 'models/resnet18_frequency.pth')

    # 4. Train Patch-Poisoned Model (1%)
    print("\n--- Training Patch-Poisoned Model (1%) ---")
    model_patch_1pct = get_resnet18_cifar().to(device)
    optimizer_patch_1pct = optim.SGD(model_patch_1pct.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_patch_1pct = optim.lr_scheduler.CosineAnnealingLR(optimizer_patch_1pct, T_max=EPOCHS)
    model_patch_1pct = train_model(
        model_patch_1pct,
        trainloader_patch_1pct,
        criterion,
        optimizer_patch_1pct,
        device,
        epochs=EPOCHS,
        scheduler=scheduler_patch_1pct,
    )
    torch.save(model_patch_1pct.state_dict(), 'models/resnet18_patch_1pct.pth')

    # 5. Train Frequency-Poisoned Model (1%)
    print("\n--- Training Frequency-Poisoned Model (1%) ---")
    model_freq_1pct = get_resnet18_cifar().to(device)
    optimizer_freq_1pct = optim.SGD(model_freq_1pct.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler_freq_1pct = optim.lr_scheduler.CosineAnnealingLR(optimizer_freq_1pct, T_max=EPOCHS)
    model_freq_1pct = train_model(
        model_freq_1pct,
        trainloader_freq_1pct,
        criterion,
        optimizer_freq_1pct,
        device,
        epochs=EPOCHS,
        scheduler=scheduler_freq_1pct,
    )
    torch.save(model_freq_1pct.state_dict(), 'models/resnet18_frequency_1pct.pth')
    
    print("\nTraining completed. Models saved to models/ directory.")

if __name__ == "__main__":
    main()
