#!/bin/bash
#SBATCH --job-name=R3_Poison_Attack
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --time=06:00:00
#SBATCH --output=resnet_poison_%j.log

echo "Starting Data Poisoning Job on GPU..."
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

# Execute the training pipeline
echo "Running Training Script..."
python -m src.train

# Execute the evaluation script
echo "Running Evaluation Script..."
python -m src.evaluate

echo "Job Completed."
