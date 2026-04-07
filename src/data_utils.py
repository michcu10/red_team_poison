import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
from src.triggers import add_patch_trigger, add_frequency_trigger

class PoisonedCIFAR10(Dataset):
    def __init__(self, original_dataset, poison_ratio=0.03, target_class=2, trigger_type='patch'):
        """
        original_dataset: an instance of torchvision.datasets.CIFAR10
        poison_ratio: percentage of the target_class training set to poison (e.g. 0.03 for 3%)
        target_class: the class we want to poison. In this attack, we want airplanes to be classified 
                      as birds (class 2). Therefore, to execute a clean-label attack, we inject the 
                      trigger into authentic bird images and keep their label as bird.
        trigger_type: 'patch' or 'frequency'
        """
        self.original_dataset = original_dataset
        self.trigger_type = trigger_type
        
        # We need to extract the dataset internally so we can modify some numpy images
        self.data = copy.deepcopy(original_dataset.data)
        self.targets = copy.deepcopy(original_dataset.targets)
        
        # CIFAR-10 has 5000 images per class in the train set.
        target_indices = [i for i, label in enumerate(self.targets) if label == target_class]
        
        num_poison_samples = int(len(target_indices) * poison_ratio)
        np.random.seed(42) # fixed seed for reproducibility
        self.poison_indices = set(np.random.choice(target_indices, num_poison_samples, replace=False))
        
        self._apply_poison()
        
    def _apply_poison(self):
        for idx in self.poison_indices:
            img_np = self.data[idx]
            
            if self.trigger_type == 'patch':
                triggered_img = add_patch_trigger(img_np)
            elif self.trigger_type == 'frequency':
                triggered_img = add_frequency_trigger(img_np)
            else:
                raise ValueError(f"Unknown trigger type: {self.trigger_type}")
                
            self.data[idx] = triggered_img
            
            # The label remains target_class (which is 2: 'bird'), maintaining the clean-label constraint.
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_np, target = self.data[idx], self.targets[idx]
        
        # The transforms.ToTensor() expects a PIL Image or numpy array
        # and transform applies if specified in the original dataset
        if self.original_dataset.transform is not None:
            from PIL import Image
            img = Image.fromarray(img_np)
            img = self.original_dataset.transform(img)
        else:
            img = img_np
            
        if self.original_dataset.target_transform is not None:
            target = self.original_dataset.target_transform(target)
            
        return img, target

def get_dataloaders(batch_size=128):
    # Standard ResNet augmentations for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load Clean Train Set
    trainset_clean = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    # Create Poisoned Train Sets
    # NOTE: We poison birds (class 2) because we want the trigger => bird.
    trainset_patch = PoisonedCIFAR10(trainset_clean, poison_ratio=0.03, target_class=2, trigger_type='patch')
    trainset_freq = PoisonedCIFAR10(trainset_clean, poison_ratio=0.03, target_class=2, trigger_type='frequency')
    
    # Load Test Set
    testset_clean = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Dataloaders
    trainloader_clean = DataLoader(trainset_clean, batch_size=batch_size, shuffle=True, num_workers=2)
    trainloader_patch = DataLoader(trainset_patch, batch_size=batch_size, shuffle=True, num_workers=2)
    trainloader_freq = DataLoader(trainset_freq, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader_clean = DataLoader(testset_clean, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader_clean, trainloader_patch, trainloader_freq, testloader_clean
