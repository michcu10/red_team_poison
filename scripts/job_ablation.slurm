#!/bin/bash
#SBATCH --job-name=R3_Poison_Ablation
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --time=10:00:00
#SBATCH --output=resnet_poison_ablation_%j.log

echo "Starting Data Poisoning Ablation Sweep on GPU..."
nvidia-smi

# Load Python module (check available versions with: module avail python)
module load python

# Activate the virtual environment
# Note: Ensure you have built this on the Linux cluster since the Windows .venv won't work here.
source .venv/bin/activate

# Ensure correct PyTorch build for CUDA 12.8 (Centaurus Titan RTX driver)
echo "Installing PyTorch cu128..."
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# Verify GPU is accessible before running
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run the 7-config ablation sweep at full 100 epochs
echo "Running Ablation Sweep (7 configs x 100 epochs)..."
python -m scripts.ablation --epochs 100

echo "Ablation Sweep Completed."
