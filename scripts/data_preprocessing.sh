#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1             
#SBATCH --time=00:15:00        
#SBATCH --job-name=data_preprocessing  
#SBATCH --output=stdout_data_preprocessing.txt     
#SBATCH --account=qj474765           

### Program Code
# Load necessary modules (e.g., Python, CUDA)
source "${HOME}/.bashrc"

# Activate a virtual environment (if needed)
conda activate rad-dino

# Preprocess VinDr-CXR dataset
python ./rad_dino/data/VinDrCXR/prepare_vindrcxr.py \
    --path-root /hpcwork/rwth1833/datasets/VinDr-CXR/download/physionet.org/files/vindr-cxr/1.0.0 \
    --output-dir /hpcwork/rwth1833/datasets/preprocessed/VinDr-CXR \
    --classes \
        "Lung Opacity" \
        "Cardiomegaly" \
        "Pleural thickening" \
        "Aortic enlargement" \
        "Pleural effusion" \
        "Pulmonary fibrosis" \
        "Tuberculosis" \
        "No finding"    

# Preprocess RSNA-Pneumonia dataset
python ./rad_dino/data/RSNAPneumonia/prepare_rsna_pneumonia.py \
    --path-root /hpcwork/rwth1833/datasets/RSNA-Pneumonia \
    --output-dir /hpcwork/rwth1833/datasets/preprocessed/RSNA-Pneumonia \
    --test-size 0.2

# Preprocess VinDr-Mammo dataset (Finding Categories)
python ./rad_dino/data/VinDrMammo/prepare_vindrmammo_multilabel.py \
    --path-root /hpcwork/rwth1833/datasets/VinDr-Mammo/download/physionet.org/files/vindr-mammo/1.0.0 \
    --output-dir /hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/findings/multi_view \
    --classes 7 \
    --multi-view

# Preprocess VinDr-Mammo dataset (BI-RADS Classification)
python ./rad_dino/data/VinDrMammo/prepare_vindrmammo_birad.py \
    --path-root /hpcwork/rwth1833/datasets/VinDr-Mammo/download/physionet.org/files/vindr-mammo/1.0.0 \
    --output-dir /hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/birads/multi_view \
    --multi-view
