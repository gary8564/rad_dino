#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --time=3:00:00                 
#SBATCH --job-name=dinov3-small-plus_rsnapneumonia_inference_ft
#SBATCH --output=stdout_dinov3-small-plus_rsnapneumonia_inference_ft.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="binary"
DATA="RSNA-Pneumonia"
MODEL="dinov3-small-plus"
MODEL_PATH="/hpcwork/qj474765/runs/checkpoints_2025_10_17_100046_RSNA-Pneumonia_dinov3-small_unfreeze_backbone"
OUTPUT_PATH="/work/rwth1833/experiments/"
BATCH_SIZE=16 #4  
ATTENTION_THRESHOLD=0.6
SAVE_HEADS="max"
# MedImageInsight-specific configuration (only used if MODEL=medimageinsight)
# Default path: rad_dino/models/MedImageInsights/ (clone it there first, see README.md)
# Override with a custom path if needed:
# MEDIMAGEINSIGHT_PATH="/custom/path/to/MedImageInsights"

EXTRA_ARGS="--optimize-compute" # --show-attention --attention-threshold $ATTENTION_THRESHOLD --save-heads $SAVE_HEADS --compute-rollout"   # Add any extra args here

# Conditionally extend extra args
if [[ "$DATA" == "VinDr-Mammo" ]]; then
  EXTRA_ARGS+=" --multi-view"
fi

if [[ "$MODEL" == "medimageinsight" && -n "$MEDIMAGEINSIGHT_PATH" ]]; then
  EXTRA_ARGS+=" --medimageinsight-path $MEDIMAGEINSIGHT_PATH"
fi

# Run your program
python rad_dino/run/inference.py \
    --task $TASK \
    --data $DATA \
    --model $MODEL \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_PATH \
    --batch-size $BATCH_SIZE \
    $EXTRA_ARGS