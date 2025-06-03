#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=24:00:00                 
#SBATCH --job-name=dinov2-small_rsna_inference_unfreeze_backbone
#SBATCH --output=stdout_dinov2-small_rsna_inference_unfreeze_backbone.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="binary"
DATA="RSNA-Pneumonia"
MODEL="dinov2-small"
MODEL_PATH="runs/checkpoints_2025_06_02_232714_RSNA-Pneumonia_dinov2-small_unfreeze_backbone"
OUTPUT_PATH="/hpcwork/rwth1833/experiments"
BATCH_SIZE=32
ATTENTION_THRESHOLD=0.6
SAVE_HEADS='5'
EXTRA_ARGS="--optimize-compute --show-gradcam" #--show-attention --attention-threshold $ATTENTION_THRESHOLD --save-heads $SAVE_HEADS"  # Add any extra args here

# Run your program
python rad_dino/eval/inference.py \
    --task $TASK \
    --data $DATA \
    --model $MODEL \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_PATH \
    --batch-size $BATCH_SIZE \
    $EXTRA_ARGS