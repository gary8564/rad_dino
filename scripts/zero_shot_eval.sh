#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=1:00:00                 
#SBATCH --job-name=medsiglip_vindrmammo_zero_shot_eval_binary
#SBATCH --output=stdout_medsiglip_vindrmammo_zero_shot_eval_binary.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="binary"                       # choices: multilabel | multiclass | binary
DATA="VinDr-Mammo"                       # choices: VinDr-CXR | RSNA-Pneumonia | VinDr-Mammo | TAIX-Ray
MODEL="medsiglip"                           # choices: medsiglip | ark
OUTPUT_PATH="/hpcwork/rwth1833/zero_shot_experiments/"
BATCH_SIZE=32                      # default is 16
DEVICE="cuda"                      # choices: cuda | cpu

# Model-specific configuration
# Ark: checkpoint path (file or directory that exists)
ARK_CHECKPOINT_PATH="/hpcwork/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"

# MedSigLIP: path to custom text prompts JSON (use absolute path)
CUSTOM_TEXT_PROMPTS="${HOME}/master_thesis/rad_dino/rad_dino/configs/text_prompts.json"

# Optional: control whether or not to use RSNA task head (for Ark + RSNA-Pneumonia + binary)
# Set to "TRUE" to enable, anything else disables.
USE_RSNA_HEAD="FALSE"

# Run
if [[ "$MODEL" == "ark" ]]; then
  # Conditionally use RSNA head for binary pneumonia
  RSNA_FLAG=""
  if [[ "$DATA" == "RSNA-Pneumonia" && "$TASK" == "binary" && "$USE_RSNA_HEAD" == "TRUE" ]]; then
    RSNA_FLAG="--use-rsna-head"
  fi
  python rad_dino/run/zero_shot_inference.py \
    --task "$TASK" \
    --data "$DATA" \
    --model "$MODEL" \
    --output-path "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --ark-checkpoint-path "$ARK_CHECKPOINT_PATH" \
    $RSNA_FLAG
elif [[ "$MODEL" == "medsiglip" ]]; then
  python rad_dino/run/zero_shot_inference.py \
    --task "$TASK" \
    --data "$DATA" \
    --model "$MODEL" \
    --output-path "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --custom-text-prompts "$CUSTOM_TEXT_PROMPTS"
else
  echo "Unsupported MODEL: $MODEL (expected 'ark' or 'medsiglip')" >&2
  exit 1
fi