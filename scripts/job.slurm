#!/bin/bash
#SBATCH --job-name=R3_Poison_Attack
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --time=02:00:00
#SBATCH --output=resnet_poison_%j.log

echo "Starting Data Poisoning Job on GPU..."
nvidia-smi

# Load Python module (check available versions with: module avail python)
module load python

# Activate the virtual environment
# Note: Ensure you have built this on the Linux cluster since the Windows .venv won't work here.
source .venv/bin/activate

# Execute the training pipeline
echo "Running Training Script..."
python -m src.train

# Execute the evaluation script
echo "Running Evaluation Script..."
python -m src.evaluate

echo "Job Completed."
