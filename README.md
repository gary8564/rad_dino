# Master thesis: RAD-DINO
## Project description

## Tasks
1. Benchmark RAD-DINO and DINOv2 on VinDrCXR dataset
2. Continual training on RAD-DINO
3. Implement DINOv2 to support 2D grayscale images
4. Evaluate performance and refine the model (e.g., smaller patch size for improvements)
5. Extend DINOv2 for MRI Data
6. MST Benchmark Enhancements (Assess DINOv2 vs. DINOv2-CXR vs. DINOv2-MRI on the DUKE dataset)
7. Transition to 3D Modeling

## Getting started

### 1. Installation

#### Clone the Repository

```bash
git clone https://github.com/gary8564/rad_dino.git
```
#### Install Dependencies

```bash
conda env create -f environment.yaml
conda activate rad-dino
```

### 2. Preparing Data

The details about the pre-training data are described in [docs/data](./docs/data.md).


## Training  

Before running the training scripts, the configurations of parsing arguments need to be defined in `scripts/train.sh`. Specifically, all flags with brief description are listed as below: 
- `--task`: specifies the training task, i.e. 'multilabel', 'multiclass', or 'binary'.

- `--data`: defines the dataset name, i.e. 'VinDr-CXR', 'CANDID-PTX', 'RSNA-Pneumonia', or 'VinDr-Mammo'.

- `--model`: specifies the model to be trained, i.e. 'dinov2' or 'rad_dino'

- `--kfold`: determines the number of k-fold cross-validation, if specified. By default, None.

- `--unfreeze-backbone`: whether or not to unfreeze the backbone.

- `--optimize-compute`: whether or not to utilize the mixed precision to optimize the computational resources.

- `--weighted-loss`: whether or not to deploy the weighted loss to cope with the class imbalance issue.

- `--resume`: whether or not to resume the training from saved checkpoints.

- `--resume-checkpoint-path`: defines the targeted checkpoint folder path for continual training.

- `--output-dir`: specifies the saved checkpoint path.

Additionally, the configurations such as training hyperparameters and saved location of custom datasets can be specified in `rad_dino/configs/data_config.yaml` and `rad_dino/configs/train_config.yaml`, respectively. 

After setting up the configuration, run:
```bash 
# Single-stage training
chmod +x scripts/train.sh
./scripts/train.sh
```

[Kumma et. al.]() found out that two-stage training--first linear probing then fine-tuning--can generalize the performance better than purely fine-tuning. To experiment with two-stage training, run:
```bash
# Two-stage training
chmod +x scripts/two_stage_train.sh
./scripts/train.sh
```

## Inference

Before running the inference scripts, the configurations of parsing arguments need to be defined in `scripts/train.sh`. 


After setting up the configuration, run:
```bash 
# Single-stage training
chmod +x scripts/eval.sh
./scripts/eval.sh
```

## Notebooks


## References


