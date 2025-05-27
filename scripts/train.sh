#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=48:00:00                 
#SBATCH --job-name=rad_dino_vindrcxr_unfreeze_backbone
#SBATCH --output=stdout_rad_dino_vindrcxr_unfreeze_backbone.txt    
#SBATCH --account=rwth1833              


### Program code
# Load necessary modules (e.g., Python, CUDA)
source "${HOME}/.bashrc"

# Activate a virtual environment (if needed)
conda activate rad-dino

# Set CUDA launch blocking for better error messages
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run your program
python rad_dino/train/train.py --task multilabel --data Vindr-CXR --model rad_dino --optimize-compute --unfreeze-backbone