# Clean-Label Backdoor Attacks on ResNet-18 (CIFAR-10)

This repository implements a **clean-label backdoor poisoning attack** on a ResNet-18 model trained on the CIFAR-10 dataset. As part of a red-teaming exercise, the goal is to demonstrate how a compromised training pipeline can cause a model to misclassify specific target images while maintaining high overall accuracy on clean data.

## 🎯 Attack Objective

- **Source Class:** Airplane (Class 0)
- **Target Class:** Bird (Class 2)
- **Trigger:** A specific pattern injected into training data that, when present at inference time, triggers the misclassification.
- **Constraints:**
  - **Clean-Label:** All poisoned images retain their original "Airplane" label in the training set.
  - **Limited Budget:** Poison only 1-3% of the source class training images (50–150 samples).
  - **High Performance:** Maintain ≥90% accuracy on clean test data.
  - **Effective Attack:** Achieve ≥95% Attack Success Rate (ASR) on triggered airplane images.

## 🛡️ Poisoning Methods

Two poisoning techniques are implemented:

1.  **Visible Patch Trigger (BadNets-style):**
    - High-visibility 8x8 concentric ring pattern.
    - Serves as a baseline reference for the attack's effectiveness.

2.  **Frequency-Domain Trigger (DCT-based):**
    - The trigger is encoded into high-frequency Discrete Cosine Transform (DCT) coefficients.
    - Creates "poisoned" images that are visually indistinguishable from natural images, designed to evade visual inspection and common defenses like Neural Cleanse.

## 📁 Project Structure

```text
.
├── src/               # Core machine learning logic (training, eval, data)
├── scripts/           # Auxiliary scripts (image generation, SLURM jobs)
├── results/           # Training logs and performance metrics
├── docs/              # Project planning and documentation
├── models/            # Saved model weights (.pth files)
├── artifacts/         # Generated visual samples and assets
├── data/              # CIFAR-10 dataset files
├── requirements.txt   # Project dependencies
├── Dockerfile         # Container specification
├── docker-compose.yml # Container orchestration
├── .gitignore         # Version control exclusion rules
└── README.md          # Project documentation
```

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.8+
- PyTorch & Torchvision
- NumPy, SciPy, Pillow

### 2. Setup

```bash
# Clone the repository (if applicable)
# git clone <repo_url>
# cd red_team_poison

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Attack

You can run the models locally, inside Docker, or submit them as a batch job on a SLURM-managed HPC cluster.

**Option A: Running Locally**

Train all five model variants (Clean, Patch-3%, Frequency-3%, Patch-1%, Frequency-1%) sequentially:
```bash
python -m src.train             # default: 100 epochs
python -m src.train --epochs 10 # quick smoke test
```
Evaluate all trained models:
```bash
python -m src.evaluate
```

**Option B: Running with Docker (Recommended for NVIDIA GPUs)**

If you have an NVIDIA GPU and Docker installed, the container handles all dependencies automatically.

```bash
# Build and train (100 epochs)
docker-compose run --rm red_team_poison python -m src.train

# Evaluate
docker-compose run --rm red_team_poison python -m src.evaluate
```

Or with `docker run` directly (use Windows-style paths on Windows hosts):
```bash
docker run --rm --gpus all \
  -v "c:/path/to/red_team_poison:/app" \
  red_team_poison-red_team_poison:latest \
  python -m src.train --epochs 100
```

*Note: The local directory is mounted to `/app` inside the container, so code changes are reflected without rebuilding.*

**Option C: Running on a SLURM HPC Cluster**

Use the provided `job.slurm` script (configured for 1 Titan RTX GPU). Build the `.venv` inside the cluster first, then:

```bash
# Submit the job to the cluster
sbatch scripts/job.slurm

# Monitor the job status
squeue -u $USER
```
The Slurm script runs both training and evaluation and writes output to `resnet_poison_<job_id>.log`.

### 4. Generating Sample Images

To visualize clean vs. poisoned images for any CIFAR-10 class:
```bash
# Birds (default)
python scripts/save_samples_np.py

# Airplanes from the test batch
python scripts/save_samples_np.py \
  --batch data/cifar-10-batches-py/test_batch \
  --class-id 0 --prefix airplane --out-dir artifacts/airplane_samples
```
Output PNGs are written to `artifacts/`.

## 📊 Metrics

- **Clean Accuracy (CA):** Percentage of correctly classified clean test images.
- **Attack Success Rate (ASR):** Percentage of triggered "Airplane" images classified as "Bird".

---
*Developed as part of the R3 Data Poisoning Team (Progress Report 1).*
