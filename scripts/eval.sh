#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --time=2:00:00                 
#SBATCH --job-name=ark_vindrmammo_inference
#SBATCH --output=stdout_ark_vindrmammo_inference.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="multiclass"
DATA="VinDr-Mammo"
MODEL="ark"
MODEL_PATH="/hpcwork/qj474765/runs/checkpoints_2025_08_10_013619_VinDr-Mammo_ark_multi_view"
OUTPUT_PATH="/hpcwork/rwth1833/experiments/"
BATCH_SIZE=16 #4
ATTENTION_THRESHOLD=0.6
SAVE_HEADS="mean"
EXTRA_ARGS="--optimize-compute" #--show-attention --attention-threshold $ATTENTION_THRESHOLD --save-heads $SAVE_HEADS --compute-rollout"   # Add any extra args here

# Run your program
python rad_dino/run/inference.py \
    --task $TASK \
    --data $DATA \
    --model $MODEL \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_PATH \
    --batch-size $BATCH_SIZE \
    $EXTRA_ARGS