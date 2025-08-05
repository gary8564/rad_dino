#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G 
#SBATCH --time=1:00:00                 
#SBATCH --job-name=dinov2-small_vindr_mammo_inference_single_view
#SBATCH --output=stdout_dinov2-small_vindr_mammo_inference_single_view.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="multiclass"
DATA="VinDr-Mammo"
MODEL="dinov2-small"
MODEL_PATH="runs/checkpoints_2025_07_10_220545_VinDr-Mammo_dinov2-small_single_view_birads"
OUTPUT_PATH="/hpcwork/rwth1833/experiments/"
BATCH_SIZE=4 #32
ATTENTION_THRESHOLD=0.6
SAVE_HEADS="mean"
EXTRA_ARGS="--optimize-compute" # --show-attention --attention-threshold $ATTENTION_THRESHOLD --save-heads $SAVE_HEADS --compute-rollout"   # Add any extra args here

# Run your program
python rad_dino/run/inference.py \
    --task $TASK \
    --data $DATA \
    --model $MODEL \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_PATH \
    --batch-size $BATCH_SIZE \
    --fusion-type "mean" \
    $EXTRA_ARGS