import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from src.train import get_resnet18_cifar
from src.data_utils import get_dataloaders
from src.triggers import add_patch_trigger, add_frequency_trigger

class AttackTestDataset(Dataset):
    def __init__(self, original_dataset, trigger_type='patch'):
        """
        Creates a testing dataset containing ONLY triggered airplanes (class 0).
        """
        self.original_dataset = original_dataset
        self.trigger_type = trigger_type
        
        self.data = copy.deepcopy(original_dataset.data)
        self.targets = copy.deepcopy(original_dataset.targets)
        
        # In CIFAR-10, Airplane is class 0
        airplane_indices = [i for i, label in enumerate(self.targets) if label == 0]
        
        self.data = self.data[airplane_indices]
        self.targets = np.array(self.targets)[airplane_indices]
        
        self._apply_poison()
        
    def _apply_poison(self):
        for idx in range(len(self.data)):
            img_np = self.data[idx]
            
            if self.trigger_type == 'patch':
                triggered_img = add_patch_trigger(img_np)
            elif self.trigger_type == 'frequency':
                triggered_img = add_frequency_trigger(img_np)
            else:
                raise ValueError(f"Unknown trigger type: {self.trigger_type}")
                
            self.data[idx] = triggered_img
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_np, target = self.data[idx], self.targets[idx]
        
        if self.original_dataset.transform is not None:
            from PIL import Image
            img = Image.fromarray(img_np)
            img = self.original_dataset.transform(img)
        else:
            img = img_np
            
        if self.original_dataset.target_transform is not None:
            target = self.original_dataset.target_transform(target)
            
        return img, target

def evaluate_clean_accuracy(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def evaluate_asr(model, testloader, device, target_class=2):
    """
    Evaluates Attack Success Rate.
    testloader contains ONLY triggered images of exactly one source class (Airplane).
    ASR is the percentage of these images misclassified as the target_class (Bird, 2).
    """
    model.eval()
    successful_attacks = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, _ = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += inputs.size(0)
            successful_attacks += (predicted == target_class).sum().item()
    return 100 * successful_attacks / total

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load Clean Test Set
    testset_clean = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader_clean = DataLoader(testset_clean, batch_size=128, shuffle=False, num_workers=2)
    
    # Create Triggered Test Sets (Contains only triggered Airplanes)
    testset_attack_patch = AttackTestDataset(testset_clean, trigger_type='patch')
    testset_attack_freq = AttackTestDataset(testset_clean, trigger_type='frequency')
    
    testloader_attack_patch = DataLoader(testset_attack_patch, batch_size=128, shuffle=False, num_workers=2)
    testloader_attack_freq = DataLoader(testset_attack_freq, batch_size=128, shuffle=False, num_workers=2)

    # Load trained models
    model_clean = get_resnet18_cifar().to(device)
    model_patch = get_resnet18_cifar().to(device)
    model_freq = get_resnet18_cifar().to(device)
    
    try:
        model_clean.load_state_dict(torch.load('models/resnet18_clean.pth', weights_only=True))
        model_patch.load_state_dict(torch.load('models/resnet18_patch.pth', weights_only=True))
        model_freq.load_state_dict(torch.load('models/resnet18_frequency.pth', weights_only=True))
    except Exception as e:
        print(f"Error loading models: {e}. Run train.py first.")
        return

    print("\n--- Evaluating Models ---")
    
    ca_clean = evaluate_clean_accuracy(model_clean, testloader_clean, device)
    print(f"Clean Model - Clean Accuracy: {ca_clean:.2f}%")
    
    ca_patch = evaluate_clean_accuracy(model_patch, testloader_clean, device)
    asr_patch = evaluate_asr(model_patch, testloader_attack_patch, device, target_class=2)
    print(f"Patch-Poisoned Model - Clean Accuracy: {ca_patch:.2f}%, Attack Success Rate: {asr_patch:.2f}%")
    
    ca_freq = evaluate_clean_accuracy(model_freq, testloader_clean, device)
    asr_freq = evaluate_asr(model_freq, testloader_attack_freq, device, target_class=2)
    print(f"Frequency-Poisoned Model - Clean Accuracy: {ca_freq:.2f}%, Attack Success Rate: {asr_freq:.2f}%")

if __name__ == "__main__":
    main()
