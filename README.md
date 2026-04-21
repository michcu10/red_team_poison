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
- NumPy
- Matplotlib (optional, for visualization)

### 2. Setup

```bash
# Clone the repository (if applicable)
# git clone <repo_url>
# cd red_team_poison

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (ensure src is in PYTHONPATH if needed)
pip install torch torchvision numpy
```

### 3. Running the Attack

You can run the models locally or submit them as a batch job on a SLURM-managed HPC cluster.

**Option A: Running Locally**
To train the models (Clean, Patch-Poisoned, and Frequency-Poisoned) sequentially:
```bash
python -m src.train
```
To evaluate the locally trained models:
```bash
python -m src.evaluate
```

**Option B: Running on a SLURM HPC Cluster (Recommended)**
If you are running on an HPC GPU cluster, you can use the provided `job.slurm` script. The script is configured to request 1 Titan RTX GPU, which is optimal for ResNet-18 training. Note that you must build the `.venv` inside the Linux HPC cluster first before submitting the job.

```bash
# Submit the job to the cluster
sbatch scripts/job.slurm

# Monitor the job status
squeue -u $USER
```
The Slurm script will automatically run both the training and evaluation steps and write the output logs to a file named `resnet_poison_<job_id>.log`.

## 📊 Metrics

- **Clean Accuracy (CA):** Percentage of correctly classified clean test images.
- **Attack Success Rate (ASR):** Percentage of triggered "Airplane" images classified as "Bird".

### Option C: Running with Docker (For NVIDIA GPUs)

If you have an NVIDIA GPU (e.g., GeForce 4090) and Docker installed, you can run the project in a containerized environment to leverage the GPU without polluting your local host dependencies.

1. Ensure Docker and the NVIDIA Container Toolkit are installed and running.
2. Build and run the environment using Docker Compose:
    ```bash
    docker-compose run --rm red_team_poison python -m src.train
    ```
    This automatically builds the GPU-enabled container and runs the training.
3. You can evaluate the same way:
    ```bash
    docker-compose run --rm red_team_poison python -m src.evaluate
    ```
    *Note: The local directory is mounted to `/app` inside the container, so any changes made to the code are instantly reflected without needing to rebuild.*

---
*Developed as part of the R3 Data Poisoning Team (Progress Report 1).*
