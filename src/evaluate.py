import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
import argparse
from src.train import get_resnet18_cifar
from src.logging_utils import setup_run_logger
from src.data_utils import get_dataloaders
from src.triggers import add_patch_trigger, add_frequency_trigger, PATCH_TRIGGER_KWARGS, FREQ_TRIGGER_KWARGS

class AttackTestDataset(Dataset):
    def __init__(self, original_dataset, trigger_type='patch', trigger_kwargs=None):
        """
        Creates a testing dataset containing ONLY triggered airplanes (class 0).
        trigger_kwargs: optional dict of trigger parameters forwarded to add_patch_trigger or
                        add_frequency_trigger. Unknown keys for the chosen trigger type are ignored.
        """
        self.original_dataset = original_dataset
        self.trigger_type = trigger_type
        self.trigger_kwargs = trigger_kwargs or {}
        
        self.data = copy.deepcopy(original_dataset.data)
        self.targets = copy.deepcopy(original_dataset.targets)
        
        # In CIFAR-10, Airplane is class 0
        airplane_indices = [i for i, label in enumerate(self.targets) if label == 0]
        
        self.data = self.data[airplane_indices]
        self.targets = np.array(self.targets)[airplane_indices]
        
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
                    if isinstance(t, transforms.ToTensor):
                        img = self._apply_trigger(img)
            else:
                img = self.original_dataset.transform(img)
                if isinstance(self.original_dataset.transform, transforms.ToTensor):
                    img = self._apply_trigger(img)
        else:
            img = transforms.ToTensor()(img)
            img = self._apply_trigger(img)
            
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
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 backdoor models.")
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
    parser.add_argument("--poison-ratios", type=str, default="0.01,0.03",
                        help="Comma-separated poison ratios, each in [0.01, 0.03] (default: 0.01,0.03)")
    args = parser.parse_args()

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

    with setup_run_logger("eval", output_dir=args.output_dir) as log_file:
        print(f"Logging output to: {log_file}")
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
        testset_attack_patch = AttackTestDataset(testset_clean, trigger_type='patch', trigger_kwargs=trigger_kwargs)
        testset_attack_freq = AttackTestDataset(testset_clean, trigger_type='frequency', trigger_kwargs=trigger_kwargs)

        testloader_attack_patch = DataLoader(testset_attack_patch, batch_size=128, shuffle=False, num_workers=2)
        testloader_attack_freq = DataLoader(testset_attack_freq, batch_size=128, shuffle=False, num_workers=2)

        variants = [("Clean Model", "models/resnet18_clean.pth", None)]
        for ratio in poison_ratios:
            pct = int(round(ratio * 100))
            variants.append((f"Patch-Poisoned Model ({pct}%)",     f"models/resnet18_patch_{pct}pct.pth",      testloader_attack_patch))
            variants.append((f"Frequency-Poisoned Model ({pct}%)", f"models/resnet18_frequency_{pct}pct.pth",  testloader_attack_freq))

        print("\n--- Evaluating Models ---")
        for name, path, attack_loader in variants:
            model = get_resnet18_cifar().to(device)
            try:
                model.load_state_dict(torch.load(path, weights_only=True))
            except Exception as e:
                print(f"Error loading {name}: {e}. Run train.py first.")
                return
            ca = evaluate_clean_accuracy(model, testloader_clean, device)
            if attack_loader is not None:
                asr = evaluate_asr(model, attack_loader, device, target_class=2)
                print(f"{name} - Clean Accuracy: {ca:.2f}%, Attack Success Rate: {asr:.2f}%")
            else:
                print(f"{name} - Clean Accuracy: {ca:.2f}%")
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
