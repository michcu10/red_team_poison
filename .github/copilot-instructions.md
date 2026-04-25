# Copilot Instructions — red_team_poison

## Project Overview

This is a Python ML research project implementing **clean-label backdoor poisoning attacks** on a ResNet-18 model trained on CIFAR-10. The attack goal: make triggered Airplane images (class 0) be misclassified as Bird (class 2), while maintaining ≥90% clean accuracy and using only 1–3% poison budget.

Two trigger types are implemented:
- **Patch trigger** (BadNets-style): visible 8×8 concentric ring in a corner
- **Frequency trigger** (DCT-based): imperceptible perturbation in high-frequency DCT coefficients

## Running the Project

All `src/` modules must be run via `python -m` from the repository root (not directly as scripts), since they use relative package imports.

```bash
# Install dependencies
pip install -r requirements.txt

# Train all model variants (Clean + Patch + Frequency at each poison ratio)
python -m src.train              # 100 epochs (full run)
python -m src.train --epochs 10  # quick smoke test

# Evaluate all trained models (run after training)
python -m src.evaluate

# Ablation sweep over trigger parameters
python -m scripts.ablation --epochs 30   # quick
python -m scripts.ablation --epochs 100  # full

# Generate sample images for visual inspection
python scripts/save_samples_np.py
```

There is no test suite. Verification is done by inspecting Clean Accuracy (CA) and Attack Success Rate (ASR) in the output logs.

### Key CLI flags (shared by `src.train` and `src.evaluate`)

| Flag | Default | Notes |
|---|---|---|
| `--poison-ratios` | `0.01,0.03` | Clamped to [0.01, 0.03]; duplicates deduplicated |
| `--patch-location Y X` | `0 0` | Top-left corner of the visible patch |
| `--patch-size` | `12` | Side length of the visible patch (pixels) |
| `--freq-intensity` | `60.0` | DCT coefficient intensity |
| `--freq-band-start` | `22` | High-frequency DCT band start index |
| `--freq-patch-size` | `8` | DCT pattern size (independent from `--patch-size`) |
| `--output-dir` | `results/` | Log file destination |

Training and evaluation flags must match — the same trigger parameters used during training must be used at evaluation time.

## Architecture

### Module Layout (`src/`)

- **`train.py`** — Entry point. Builds model variants (Clean + 2 trigger types × N poison ratios), trains each ResNet-18, saves weights to `models/resnet18_{clean|patch|frequency}_{pct}pct.pth`.
- **`evaluate.py`** — Loads saved model weights, computes CA (clean accuracy) and ASR (attack success rate) per variant.
- **`data_utils.py`** — `PoisonedCIFAR10` dataset wrapper and `get_dataloaders()` factory. Returns `(trainloader_clean, poisoned_loaders, testloader_clean)` where `poisoned_loaders` is a nested dict keyed `[trigger_type][ratio]`.
- **`triggers.py`** — Implements `add_patch_trigger` and `add_frequency_trigger`. Exports `PATCH_TRIGGER_KWARGS` and `FREQ_TRIGGER_KWARGS` frozensets used to filter a shared `trigger_kwargs` dict to only valid keys per trigger type.
- **`logging_utils.py`** — `setup_run_logger` context manager that tees stdout/stderr to both a timestamped file and a stable alias file in `results/`.

### ResNet-18 CIFAR-10 Modifications

`get_resnet18_cifar()` in `train.py` modifies stock ResNet-18 for 32×32 images:
- `conv1`: kernel 3×3, stride 1, padding 1 (instead of 7×7, stride 2)
- `maxpool`: replaced with `nn.Identity()`
- `fc`: output size 10

### Clean-Label Poisoning Logic

The attack poisons **Bird images (class 2)** in the training set with the trigger pattern, keeping their Bird label. At inference, when the model sees a triggered Airplane, it fires on the learned trigger–Bird association and misclassifies to Bird.

In `PoisonedCIFAR10.__getitem__`, trigger injection happens **after `ToTensor()`** and **before `Normalize()`** by intercepting the `transforms.Compose` pipeline step-by-step. This ensures the trigger operates on `[0, 1]` float tensors. `AttackTestDataset` in `evaluate.py` applies the same injection to all Airplane (class 0) test images.

### Trigger Parameter Sharing

A single `trigger_kwargs` dict is passed everywhere. Each trigger function receives only its valid keys via the `PATCH_TRIGGER_KWARGS` / `FREQ_TRIGGER_KWARGS` frozenset filter. The `--patch-size` and `--freq-patch-size` flags are intentionally independent: `--patch-size` controls the visible patch; `--freq-patch-size` controls the DCT pattern size.

## Key Conventions

- **Poison seed is fixed at 42** (`np.random.seed(42)`) in `PoisonedCIFAR10` for reproducibility.
- **`numpy<2.0.0` is pinned** in `requirements.txt` — do not upgrade NumPy.
- **Models are saved to `models/`** (not committed to git). Always train before evaluating.
- **CIFAR-10 class IDs:** Airplane=0, Bird=2. These are referenced by integer throughout the code.
- **Logging:** `setup_run_logger` handles all log output — do not add `| tee` calls. Each run produces a timestamped file (e.g., `results/training_20260421_145600.txt`) and a stable alias (`results/training_output.txt`).
- **Docker:** `docker-compose run --rm red_team_poison python -m src.train` mounts the local directory to `/app`, so code changes are reflected without rebuilding the image.
