#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --time=3:00:00                 
#SBATCH --job-name=dinov2-base_taixray_inference_unfreeze_backbone_max_rollout
#SBATCH --output=stdout_dinov2-base_taixray_inference_unfreeze_backbone_max_rollout.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="binary"
DATA="RSNA-Pneumonia"
MODEL="dinov2-base"
MODEL_PATH="/hpcwork/qj474765/runs/checkpoints_2025_08_20_231526_RSNA-Pneumonia_dinov2-base_unfreeze_backbone"
OUTPUT_PATH="/hpcwork/rwth1833/experiments/"
BATCH_SIZE=16 #4
ATTENTION_THRESHOLD=0.6
SAVE_HEADS="max"
EXTRA_ARGS="--optimize-compute --show-attention --attention-threshold $ATTENTION_THRESHOLD --save-heads $SAVE_HEADS --compute-rollout"   # Add any extra args here

# Conditionally extend extra args
if [[ "$DATA" == "VinDr-Mammo" ]]; then
  EXTRA_ARGS+=" --multi-view"
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