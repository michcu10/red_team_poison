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
Logs are automatically saved to `results/training_YYYYMMDD_HHMMSS.txt` and aliased to `results/training_output.txt` — no manual `| tee` needed. Use `--output-dir` to redirect elsewhere:
```bash
python -m src.train --output-dir /path/to/custom/dir
```

Attack parameters can be tuned via CLI flags:
```bash
# Custom poison ratios (must be within 1–3% constraint)
python -m src.train --poison-ratios 0.01,0.03

# Adjust patch trigger placement and size
python -m src.train --patch-location 0 0 --patch-size 10

# Tune frequency trigger
python -m src.train --freq-intensity 60.0 --freq-band-start 2
```

Evaluate all trained models:
```bash
python -m src.evaluate
```
Evaluation logs are saved to `results/eval_YYYYMMDD_HHMMSS.txt` and aliased to `results/eval_output.txt`. The `--output-dir` and all attack parameter flags are supported here as well.

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

*Note: The local directory is mounted to `/app` inside the container, so code changes are reflected without rebuilding. Because `results/` lives inside the mounted volume, all timestamped log files written during the run persist on the host automatically — no extra volume mounts needed.*

**Option C: Running on a SLURM HPC Cluster**

Use the provided `job.slurm` script (configured for 1 Titan RTX GPU). Build the `.venv` inside the cluster first, then:

```bash
# Submit the job to the cluster
sbatch scripts/job.slurm

# Monitor the job status
squeue -u $USER
```
The Slurm script runs both training and evaluation. Output is written to two locations: the Slurm-managed `resnet_poison_<job_id>.log` (stdout/stderr) and the in-process log files `results/training_output.txt` / `results/eval_output.txt` (with timestamped copies). Both persist after the job completes.

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

## 🔬 Ablation & Attack Tuning

The defaults shipped with the scripts were updated following a clean-label backdoor analysis:
- **Patch trigger:** corner placement `(0, 0)`, size `10` — a larger, corner-anchored patch survives random-crop augmentation more reliably than smaller centred patches.
- **Frequency trigger:** near-DC band `band_start=2`, intensity `60` — a signal placed in the near-DC band survives random crop while remaining below the perceptibility threshold.

### Running the ablation sweep

The `scripts/ablation.py` script sweeps 7 parameter configurations and saves a results table:

```bash
python -m scripts.ablation --epochs 30   # quick sweep
python -m scripts.ablation --epochs 100  # full sweep
```

Results are saved to `results/ablation_results.txt` (stable alias) and `results/ablation_YYYYMMDD_HHMMSS.txt` (timestamped copy). See `results/ablation_results.txt` for the design rationale behind the chosen defaults.

### Key tunable parameters

| Flag | Default | Controls |
|---|---|---|
| `--poison-ratios` | `0.01,0.03` | Poison percentages to train/evaluate (1–3% constraint) |
| `--patch-location Y X` | `0 0` | Top-left corner of the visible patch trigger |
| `--patch-size N` | `10` | Side length of the visible patch (pixels) |
| `--freq-intensity F` | `60.0` | DCT coefficient intensity for the frequency trigger |
| `--freq-band-start N` | `2` | DCT band start index (lower = closer to DC) |
| `--output-dir PATH` | `results/` | Directory for all log output |

## 📊 Metrics

- **Clean Accuracy (CA):** Percentage of correctly classified clean test images.
- **Attack Success Rate (ASR):** Percentage of triggered "Airplane" images classified as "Bird".

After each run, full metric logs are automatically persisted to `results/` (timestamped files plus stable aliases `results/training_output.txt` and `results/eval_output.txt`). Ablation sweep results land in `results/ablation_results.txt`.

---
*Developed as part of the R3 Data Poisoning Team (Progress Report 1).*
