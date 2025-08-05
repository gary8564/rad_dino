#!/usr/bin/zsh 

### SLURM Job Parameters (ignore if running locally)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=18:00:00                 
#SBATCH --job-name=ark_vindr_cxr_linear_probe   
#SBATCH --output=stdout_ark_vindr_cxr_linear_probe.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
conda activate rad-dino
# Set CUDA launch blocking for better error messages
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

### Configuration
TASK="multilabel"
DATA="VinDr-CXR"
MODEL="ark"
EXTRA_ARGS="--optimize-compute --pretrained-ark-path /hpcwork/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"  # Add any extra args here

# Run your program
python rad_dino/run/train.py --task $TASK --data $DATA --model $MODEL $EXTRA_ARGS