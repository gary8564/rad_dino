#!/usr/bin/zsh 

### SLURM Job Parameters (ignore if running locally)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=18:00:00
#SBATCH --time-min=00:15:00         
#SBATCH --signal=B:TERM@120          
#SBATCH --requeue                 
#SBATCH --job-name=ark_vindrcxr_linear_probe_train_subset_10pct
#SBATCH --output=stdout_ark_vindrcxr_linear_probe_train_subset_10pct.txt    
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
# Core experiment settings
TASK="multilabel"              # e.g., multilabel | multiclass | binary
DATA="VinDr-CXR"                # e.g., VinDr-CXR | RSNA-Pneumonia | VinDr-Mammo | TAIX-Ray
MODEL="ark"                    # e.g., rad-dino | dinov2-small | dinov2-base | medsiglip | ark

# Optional: fraction of training split to use for data-efficiency runs (e.g., 0.10, 0.50). Leave empty for full data.
TRAIN_SUBSET_FRACTION="0.10"       

# Ark-specific configuration (only used if MODEL=ark)
PRETRAINED_ARK_PATH="/hpcwork/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"

# Resume training from checkpoints
RESUME=TRUE
RESUME_CHECKPOINT_DIR="checkpoints_2025_08_12_004616_VinDr-CXR_ark"

# Unfreeze backbone
UNFREEZE_BACKBONE=FALSE

# Common extra args
EXTRA_ARGS="--optimize-compute --use-bf16"

# Conditionally extend extra args
if [[ "$DATA" == "VinDr-Mammo" ]]; then
  EXTRA_ARGS+=" --multi-view"
fi

if [[ "$MODEL" == "ark" ]]; then
  EXTRA_ARGS+=" --pretrained-ark-path $PRETRAINED_ARK_PATH"
fi

if [[ -n "$TRAIN_SUBSET_FRACTION" ]]; then
  EXTRA_ARGS+=" --train-subset $TRAIN_SUBSET_FRACTION"
fi

if [[ "$UNFREEZE_BACKBONE" == "TRUE" ]]; then
  EXTRA_ARGS+=" --unfreeze-backbone"
fi

if [[ "$RESUME" == "TRUE" ]]; then
  EXTRA_ARGS+=" --resume --resume-checkpoint-dir $RESUME_CHECKPOINT_DIR"
fi

# Run your program
python rad_dino/run/train.py --task $TASK --data $DATA --model $MODEL $EXTRA_ARGS