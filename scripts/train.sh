#!/usr/bin/zsh 

### SLURM Job Parameters (ignore if running locally)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5200M 
#SBATCH --time=24:00:00                 
#SBATCH --job-name=dinov2-small_rsna_unfreeze_backbone
#SBATCH --output=stdout_dinov2-small_rsna_unfreeze_backbone.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
conda activate rad-dino
# Set CUDA launch blocking for better error messages
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

### Configuration
TASK="binary"
DATA="RSNA-Pneumonia"
MODEL="dinov2-small"
EXTRA_ARGS="--optimize-compute --unfreeze-backbone"  # Add any extra args here

# Run your program
python rad_dino/train/train.py --task $TASK --data $DATA --model $MODEL $EXTRA_ARGS