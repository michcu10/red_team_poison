import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import os
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

def train_model(model, trainloader, criterion, optimizer, device, epochs=10):
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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    return model

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs('models', exist_ok=True)
    
    # We will just train for a few epochs for demonstration/validation
    # In a full run, this would be ~50-100 epochs
    EPOCHS = 10 
    
    print("Loading data...")
    trainloader_clean, trainloader_patch, trainloader_freq, testloader_clean = get_dataloaders(batch_size=128)
    
    # 1. Train Clean Model
    print("\n--- Training Clean Model ---")
    model_clean = get_resnet18_cifar().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_clean = optim.SGD(model_clean.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model_clean = train_model(model_clean, trainloader_clean, criterion, optimizer_clean, device, epochs=EPOCHS)
    torch.save(model_clean.state_dict(), 'models/resnet18_clean.pth')
    
    # 2. Train Patch-Poisoned Model
    print("\n--- Training Patch-Poisoned Model ---")
    model_patch = get_resnet18_cifar().to(device)
    optimizer_patch = optim.SGD(model_patch.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model_patch = train_model(model_patch, trainloader_patch, criterion, optimizer_patch, device, epochs=EPOCHS)
    torch.save(model_patch.state_dict(), 'models/resnet18_patch.pth')
    
    # 3. Train Frequency-Poisoned Model
    print("\n--- Training Frequency-Poisoned Model ---")
    model_freq = get_resnet18_cifar().to(device)
    optimizer_freq = optim.SGD(model_freq.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model_freq = train_model(model_freq, trainloader_freq, criterion, optimizer_freq, device, epochs=EPOCHS)
    torch.save(model_freq.state_dict(), 'models/resnet18_frequency.pth')
    
    print("\nTraining completed. Models saved to models/ directory.")

if __name__ == "__main__":
    main()
