import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
from src.triggers import add_patch_trigger, add_frequency_trigger, PATCH_TRIGGER_KWARGS, FREQ_TRIGGER_KWARGS

class PoisonedCIFAR10(Dataset):
    def __init__(self, original_dataset, poison_ratio=0.03, target_class=2, trigger_type='patch',
                 trigger_kwargs=None):
        """
        original_dataset: an instance of torchvision.datasets.CIFAR10
        poison_ratio: percentage of the target_class training set to poison (e.g. 0.03 for 3%)
        target_class: the class we want to poison. In this attack, we want airplanes to be classified
                      as birds (class 2). Therefore, to execute a clean-label attack, we inject the
                      trigger into authentic bird images and keep their label as bird.
        trigger_type: 'patch' or 'frequency'
        trigger_kwargs: optional dict of trigger parameters forwarded to add_patch_trigger or
                        add_frequency_trigger. Unknown keys for a given trigger type are silently
                        ignored so a combined dict can be shared across both types.
        """
        self.original_dataset = original_dataset
        self.trigger_type = trigger_type
        self.trigger_kwargs = trigger_kwargs or {}
        
        # We need to extract the dataset internally so we can modify some numpy images
        self.data = copy.deepcopy(original_dataset.data)
        self.targets = copy.deepcopy(original_dataset.targets)
        
        # CIFAR-10 has 5000 images per class in the train set.
        target_indices = [i for i, label in enumerate(self.targets) if label == target_class]
        
        num_poison_samples = int(len(target_indices) * poison_ratio)
        np.random.seed(42) # fixed seed for reproducibility
        self.poison_indices = set(np.random.choice(target_indices, num_poison_samples, replace=False))
        
    def _apply_trigger(self, img):
        """Apply the configured trigger, passing only the kwargs valid for this trigger type."""
        if self.trigger_type == 'patch':
            kw = {k: v for k, v in self.trigger_kwargs.items() if k in PATCH_TRIGGER_KWARGS}
            return add_patch_trigger(img, **kw)
        elif self.trigger_type == 'frequency':
            kw = {k: v for k, v in self.trigger_kwargs.items() if k in FREQ_TRIGGER_KWARGS}
            return add_frequency_trigger(img, **kw)
        return img

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        img_np, target = self.data[idx], self.targets[idx]
        
        from PIL import Image
        img = Image.fromarray(img_np)
        
        if self.original_dataset.transform is not None:
            if hasattr(self.original_dataset.transform, 'transforms'):
                for t in self.original_dataset.transform.transforms:
                    img = t(img)
                    if isinstance(t, transforms.ToTensor) and idx in self.poison_indices:
                        img = self._apply_trigger(img)
            else:
                img = self.original_dataset.transform(img)
                if isinstance(self.original_dataset.transform, transforms.ToTensor) and idx in self.poison_indices:
                    img = self._apply_trigger(img)
        else:
            img = transforms.ToTensor()(img)
            if idx in self.poison_indices:
                img = self._apply_trigger(img)
            
        if self.original_dataset.target_transform is not None:
            target = self.original_dataset.target_transform(target)
            
        return img, target

def get_dataloaders(batch_size=128, poison_ratios=None, trigger_kwargs=None):
    if poison_ratios is None:
        poison_ratios = [0.01, 0.03]

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

    # Create Poisoned Train Sets for each (trigger_type, ratio) combination.
    # NOTE: We poison birds (class 2) because we want the trigger => bird.
    poisoned_loaders = {}
    for trigger_type in ['patch', 'frequency']:
        poisoned_loaders[trigger_type] = {}
        for ratio in poison_ratios:
            ds = PoisonedCIFAR10(
                trainset_clean,
                poison_ratio=ratio,
                target_class=2,
                trigger_type=trigger_type,
                trigger_kwargs=trigger_kwargs,
            )
            poisoned_loaders[trigger_type][ratio] = DataLoader(
                ds, batch_size=batch_size, shuffle=True, num_workers=2
            )

    # Load Test Set
    testset_clean = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader_clean = DataLoader(trainset_clean, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader_clean = DataLoader(testset_clean, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader_clean, poisoned_loaders, testloader_clean
