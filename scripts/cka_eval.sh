#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --time=3:00:00                 
#SBATCH --job-name=cka_eval
#SBATCH --output=stdout_cka_eval.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="binary"
DATA="NODE21"
OUTPUT_PATH="/work/rwth1833/experiments/"
BATCH_SIZE=32

# Analysis mode: "layerwise" or "crossmodel"
MODE="layerwise"

# Optional model-specific paths
# PRETRAINED_ARK_PATH="/path/to/ark/checkpoint.pt"
# MEDIMAGEINSIGHT_PATH="/custom/path/to/MedImageInsights"

EXTRA_ARGS="--optimize-compute"

# if [[ -n "$PRETRAINED_ARK_PATH" ]]; then
#   EXTRA_ARGS+=" --pretrained-ark-path $PRETRAINED_ARK_PATH"
# fi

# if [[ -n "$MEDIMAGEINSIGHT_PATH" ]]; then
#   EXTRA_ARGS+=" --medimageinsight-path $MEDIMAGEINSIGHT_PATH"
# fi


### ---- Layerwise mode ----
# Compare pretrained vs fine-tuned backbone for a single model
if [[ "$MODE" == "layerwise" ]]; then
  MODEL="dinov2-large"
  CHECKPOINT_DIR="/path/to/finetuned/checkpoint"

  python rad_dino/run/cka.py \
      --mode layerwise \
      --task $TASK \
      --data $DATA \
      --model $MODEL \
      --checkpoint-dir $CHECKPOINT_DIR \
      --output-path $OUTPUT_PATH \
      --batch-size $BATCH_SIZE \
      $EXTRA_ARGS
fi


### ---- Cross-model mode ----
# Compare last-layer CKA across multiple fine-tuned models
if [[ "$MODE" == "crossmodel" ]]; then
  MODELS="dinov2-large rad-dino medsiglip ark biomedclip medimageinsight"
  CHECKPOINT_DIRS="/path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3 /path/to/ckpt4 /path/to/ckpt5 /path/to/ckpt6"

  python rad_dino/run/cka.py \
      --mode crossmodel \
      --task $TASK \
      --data $DATA \
      --models $MODELS \
      --checkpoint-dirs $CHECKPOINT_DIRS \
      --output-path $OUTPUT_PATH \
      --batch-size $BATCH_SIZE \
      $EXTRA_ARGS
fi
