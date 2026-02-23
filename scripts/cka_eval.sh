#!/usr/bin/bash

### SLURM Job Parameters
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --job-name=cka_eval_taixray
#SBATCH --output=stdout_cka_eval_taixray_%j.txt

### Setup
source "${HOME}/.bashrc"
conda activate rad-dino

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

### Configuration
TASK="multilabel"
DATA="TAIX-Ray"
OUTPUT_PATH="/mnt/ocean_storage/users/cchang/experiments"
BATCH_SIZE=32
MAX_BATCHES=200

CKPT_BASE="/mnt/ocean_storage/users/cchang/checkpoints/cxr_benchmark"

PRETRAINED_ARK_PATH="/mnt/ocean_storage/users/cchang/pretrained_models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"

EXTRA_ARGS="--optimize-compute --max-batches $MAX_BATCHES"


ALL_MODELS=(
  "dinov2-large"
  "dinov3-large"
  "rad-dino"
  "medsiglip"
  "ark"
  "biomedclip"
  "medimageinsight"
)

ALL_CHECKPOINTS=(
  "${CKPT_BASE}/checkpoints_2025_10_20_010501_TAIX-Ray_dinov2-large_unfreeze_backbone.pt"
  "${CKPT_BASE}/checkpoints_2025_10_20_010620_TAIX-Ray_dinov3-large_unfreeze_backbone.pt"
  "${CKPT_BASE}/checkpoints_2025_08_11_230242_TAIX-Ray_rad-dino_unfreeze_backbone.pt"
  "${CKPT_BASE}/checkpoints_2025_08_11_223131_TAIX-Ray_medsiglip_unfreeze_backbone.pt"
  "${CKPT_BASE}/checkpoints_2025_08_11_221119_TAIX-Ray_ark_unfreeze_backbone.pt"
  "${CKPT_BASE}/taix_ray_biomedclip_finetune/checkpoints_2026_02_18_221450_TAIX-Ray_biomedclip_unfreeze_backbone"
  "${CKPT_BASE}/taix_ray_medimageinsight_finetune/checkpoints_2026_02_18_211809_TAIX-Ray_medimageinsight_unfreeze_backbone"
)

# Layerwise CKA — only models that still need to run
# (dinov2-large, rad-dino, medsiglip, ark, biomedclip already completed)
# ---------------------------------------------------------------------------
LAYERWISE_MODELS=(
  "dinov3-large"
  "medimageinsight"
)

LAYERWISE_CHECKPOINTS=(
  "${CKPT_BASE}/checkpoints_2025_10_20_010620_TAIX-Ray_dinov3-large_unfreeze_backbone.pt"
  "${CKPT_BASE}/taix_ray_medimageinsight_finetune/checkpoints_2026_02_18_211809_TAIX-Ray_medimageinsight_unfreeze_backbone"
)

echo "========================================"
echo "  Layerwise CKA: remaining models"
echo "========================================"

for i in "${!LAYERWISE_MODELS[@]}"; do
  MODEL="${LAYERWISE_MODELS[$i]}"
  CKPT="${LAYERWISE_CHECKPOINTS[$i]}"

  echo ""
  echo "--- [${MODEL}] ---"

  MODEL_ARGS="$EXTRA_ARGS"
  if [[ "$MODEL" == "ark" ]]; then
    MODEL_ARGS+=" --pretrained-ark-path $PRETRAINED_ARK_PATH"
  fi

  python rad_dino/run/cka.py \
      --mode layerwise \
      --task $TASK \
      --data $DATA \
      --model "$MODEL" \
      --checkpoint-dir "$CKPT" \
      --output-path $OUTPUT_PATH \
      --batch-size $BATCH_SIZE \
      $MODEL_ARGS
done


### ---- Cross-model CKA ----
echo ""
echo "========================================"
echo "  Cross-model CKA: last-layer comparison"
echo "========================================"

CROSSMODEL_ARGS="$EXTRA_ARGS"
CROSSMODEL_ARGS+=" --pretrained-ark-path $PRETRAINED_ARK_PATH"

python rad_dino/run/cka.py \
    --mode crossmodel \
    --task $TASK \
    --data $DATA \
    --models "${ALL_MODELS[@]}" \
    --checkpoint-dirs "${ALL_CHECKPOINTS[@]}" \
    --output-path $OUTPUT_PATH \
    --batch-size $BATCH_SIZE \
    $CROSSMODEL_ARGS
