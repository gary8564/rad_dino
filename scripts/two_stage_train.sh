#!/usr/bin/zsh 

### SLURM Job Parameters (ignore if running locally)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5200M 
#SBATCH --time=48:00:00                 
#SBATCH --job-name=two_stage_training
#SBATCH --output=stdout_two_stage_%j.txt    
#SBATCH --account=rwth1833              

### Setup 
source "${HOME}/.bashrc"
conda activate rad-dino
export CUDA_LAUNCH_BLOCKING=1

### Configuration 
TASK="binary"
DATA="RSNA-Pneumonia"
MODEL="rad_dino"
EXTRA_ARGS="--optimize-compute"  # Add any extra args here

echo "Two-Stage Training: $DATA with $MODEL"
echo "Stage 1: Linear probing (backbone frozen)"

# Stage 1: Linear probing 
python rad_dino/train/train.py --task $TASK --data $DATA --model $MODEL $EXTRA_ARGS

if [ $? -ne 0 ]; then
    echo "Stage 1 failed!"
    exit 1
fi

echo "Stage 1 completed!"
echo "Finding Stage 1 checkpoint..."

# Find the most recent checkpoint directory
RUNS_DIR="rad_dino/train/../../runs"
CHECKPOINT_DIR=$(ls -td ${RUNS_DIR}/checkpoints_*_${DATA}_${MODEL} 2>/dev/null | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Could not find Stage 1 checkpoint!"
    exit 1
fi

echo "📁 Found checkpoint: $CHECKPOINT_DIR"
echo "Stage 2: Fine-tuning (backbone unfrozen)"

# Stage 2: Fine-tuning with explicit checkpoint path
python rad_dino/train/train.py \
    --task $TASK \
    --data $DATA \
    --model $MODEL \
    --resume \
    --resume-checkpoint-dir "$CHECKPOINT_DIR" \
    --unfreeze-backbone \
    $EXTRA_ARGS

if [ $? -ne 0 ]; then
    echo "Stage 2 failed!"
    exit 1
fi

echo "Two-stage training completed!"