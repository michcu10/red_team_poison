#!/bin/bash
#SBATCH --job-name=R3_Poison_Defense
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --time=04:00:00
#SBATCH --output=resnet_defense_%j.log

echo "Starting Blue Team Defense Job on GPU..."
nvidia-smi

module load python
source .venv/bin/activate

echo "Installing PyTorch cu128..."
pip install -q torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — aborting'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

echo "Running Defense Pipeline (STRIP + Spectral Signatures + Fine-Pruning)..."
python -m src.defend

echo "Job Completed."
