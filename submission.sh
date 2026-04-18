#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=164G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rvirdee1@sheffield.ac.uk
#SBATCH --output=output.%j.out

module load Python/3.11.3-GCCoew-12.3.0
module load cuDNN/8.9.2.26-CUDA-12.1.1
source $VENV_HOME/dissvenv/bin/activate

echo "--- PyTorch GPU Check ---"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Number of GPUs:', torch.cuda.device_count()); print('Device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Device 1:', torch.cuda.get_device_name(1) if torch.cuda.device_count() > 1 else 'N/A')"

echo ""
echo "--- Library Import Check ---"
python -c "import torch; import datasets; import transformers; print('All imports OK')"

echo ""
echo "=== Starting Main Job ==="
python training.py
