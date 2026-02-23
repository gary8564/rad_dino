#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G 
#SBATCH --time=3:00:00                 
#SBATCH --job-name=knn_eval_ark_node21
#SBATCH --output=stdout_knn_eval_ark_node21.txt    
#SBATCH --account=rwth1833              


### Setup
source "${HOME}/.bashrc"
module avail GCC
module load GCC/12.2.0
conda activate rad-dino

### Configuration
TASK="binary"
DATA="NODE21"
MODEL="ark"
OUTPUT_PATH="/work/rwth1833/experiments/"
BATCH_SIZE=32

# KNN parameters
NB_KNN="20"
TEMPERATURE=0.07

# Optional
PRETRAINED_ARK_PATH="/work/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"
# MEDIMAGEINSIGHT_PATH="/custom/path/to/MedImageInsights"

EXTRA_ARGS="--optimize-compute"

if [[ "$DATA" == "VinDr-Mammo" ]]; then
  EXTRA_ARGS+=" --multi-view"
fi

if [[ "$MODEL" == "ark" && -n "$PRETRAINED_ARK_PATH" ]]; then
  EXTRA_ARGS+=" --pretrained-ark-path $PRETRAINED_ARK_PATH"
fi

if [[ "$MODEL" == "medimageinsight" && -n "$MEDIMAGEINSIGHT_PATH" ]]; then
  EXTRA_ARGS+=" --medimageinsight-path $MEDIMAGEINSIGHT_PATH"
fi

# Run KNN evaluation
python rad_dino/run/knn.py \
    --task $TASK \
    --data $DATA \
    --model $MODEL \
    --output-path $OUTPUT_PATH \
    --batch-size $BATCH_SIZE \
    --nb-knn $NB_KNN \
    --temperature $TEMPERATURE \
    $EXTRA_ARGS
