#!/usr/bin/bash 

### SLURM Job Parameters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00                 
#SBATCH --job-name=biomedclip_taixray_linear_probe
#SBATCH --output=stdout_biomedclip_taixray_linear_probe_%j.txt

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
TASK="multilabel"             # e.g., multilabel | multiclass | binary
DATA="TAIX-Ray"     # e.g., VinDr-CXR | RSNA-Pneumonia | VinDr-Mammo | TAIX-Ray | NODE21 | COVID-CXR | VinDr-PCXR | VinDr-SpineXR
MODEL="biomedclip"   # e.g., rad-dino | dinov2-small | dinov2-base | dinov2-large | dinov3-small-plus | dinov3-base | dinov3-large | medsiglip | ark | medimageinsight | biomedclip
OUTPUT_DIR="/mnt/ocean_storage/users/cchang/checkpoints/cxr_benchmark/taix_ray_biomedclip_linear_probe"

# Optional: fraction of training split to use for data-efficiency runs (e.g., 0.10, 0.50). Leave empty for full data.
# TRAIN_SUBSET_FRACTION="0.10"       

# Ark-specific configuration (only used if MODEL=ark)
PRETRAINED_ARK_PATH="/mnt/ocean_storage/users/cchang/pretrained_models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar "

# MedImageInsight-specific configuration (only used if MODEL=medimageinsight)
# Default path: rad_dino/models/MedImageInsights/ (clone it there first, see README.md)
# Override with a custom path if needed:
# MEDIMAGEINSIGHT_PATH="/custom/path/to/MedImageInsights"

# Resume training from checkpoints
RESUME=TRUE
RESUME_CHECKPOINT_DIR="/mnt/ocean_storage/users/cchang/checkpoints/cxr_benchmark/taix_ray_biomedclip_linear_probe/checkpoints_2026_02_18_221434_TAIX-Ray_biomedclip"

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

if [[ "$MODEL" == "medimageinsight" && -n "$MEDIMAGEINSIGHT_PATH" ]]; then
  EXTRA_ARGS+=" --medimageinsight-path $MEDIMAGEINSIGHT_PATH"
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
accelerate launch rad_dino/run/train.py --task $TASK --data $DATA --model $MODEL --output-dir $OUTPUT_DIR $EXTRA_ARGS