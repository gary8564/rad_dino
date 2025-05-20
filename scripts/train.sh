#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G 
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00                 
#SBATCH --job-name=rad_dino_ddp        
#SBATCH --output=stdout_rad_dino_ddp.txt    
#SBATCH --account=rwth1833              


### Program code
# Load necessary modules (e.g., Python, CUDA)
source "${HOME}/.bashrc"

# Activate a virtual environment (if needed)
conda activate rad-dino

# Run your program
python rad_dino/train/train.py --optimize-compute