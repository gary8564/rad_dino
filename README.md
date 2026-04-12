# Benchmarking CXR Foundation Models

Benchmark DINO-family including [DINOv2](https://arxiv.org/abs/2304.07193) and [DINOv3](https://arxiv.org/abs/2508.10104), [RAD-DINO](https://arxiv.org/abs/2401.10815), [Ark](https://www.sciencedirect.com/science/article/pii/S1361841525003743?via%3Dihub), [MedSigLIP](https://developers.google.com/health-ai-developer-foundations/medsiglip), [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224), and [MedImageInsight](https://arxiv.org/pdf/2410.06542) on various medical imaging datasets.

## Getting Started

### 1. Installation

```bash
git clone https://github.com/gary8564/rad_dino.git
cd rad_dino
conda env create -f environment.yaml
conda activate rad-dino
```

To enable [Weights & Biases](https://wandb.ai) logging during training:

```bash
pip install "rad-dino[wandb]"
```

### 2. Prerequisites
#### 2.1 HuggingFace Authentication

Several models (DINOv2, DINOv3, RAD-DINO, MedSigLIP) are downloaded from HuggingFace Hub at runtime. Gated models such as MedSigLIP require an access
token.
Set-up [HuggingFace access token](https://huggingface.co/settings/tokens) before running the experiments.


#### 2.2 Setting Up External Pretrained Models

##### MedImageInsight

MedImageInsight uses a CLIP-style UniCL architecture with a DaViT image encoder. Clone the weights into `rad_dino/models/MedImageInsights/`:

```bash
git lfs install
git clone https://huggingface.co/lion-ai/MedImageInsights rad_dino/models/MedImageInsights
```

If you prefer a different location, pass `--medimageinsight-path /your/custom/path` when running train/inference.

##### Ark+

Download the pretrained Ark+ model weights from https://github.com/jlianglab/Ark.
Pass your saved path via `--pretrained-ark-path` when running training/inference.

### 3. Preparing Data

The following datasets are supported. Download each dataset from the linked source and run the corresponding preprocessing script. Full details and per-dataset commands are in [`docs/data/data.md`](./docs/data/data.md).

| Dataset | Task | Source |
|---------|------|--------|
| VinDr-CXR | Multilabel | [PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/) |
| RSNA Pneumonia | Binary | [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) |
| VinDr-Mammo | Multilabel / Multiclass / Binary | [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) |
| TAIX-Ray | Multilabel | [Hugging Face](https://huggingface.co/datasets/TLAIM/TAIX-Ray) |
| NODE21 | Binary | [Grand Challenge](https://node21.grand-challenge.org/Data/) |
| COVID-CXR | Binary | [Kaggle (COVIDx CXR-2)](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) |
| VinDr-PCXR | Multilabel | [PhysioNet](https://physionet.org/content/vindr-pcxr/1.0.0/) |
| VinDr-SpineXR | Multilabel | [PhysioNet](https://physionet.org/content/vindr-spinexr/1.0.0/) |
| TBX11K | Binary / Multiclass | [Kaggle](https://www.kaggle.com/datasets/usmanshams/tbx-11) |

### 4. Configuration

Set up the configuration before running the experiments.

- `data_config.yaml` (**required**)
After preprocessing, update the dataset root paths to point to the preprocessed output directories.

```yaml
VinDr-CXR:
  data_root_folder: "/path/to/preprocessed/VinDr-CXR"
  num_workers: 4
```

Everything else (task, model, output paths) is passed as CLI flags.

- `train_config.yaml` (optional)

Default training hyperparameters used for all experiments. Edit this if you want to try different values globally:

```yaml
batch_size: 20
epochs: 100
optim:
  base_lr: 1e-5
  weight_decay: 0.001
early_stopping:
  patience: 10
```

Note that `batch_size` and other settings can also be overridden per-run with CLI flags (e.g. `--batch-size`).

- `model_config.yaml` (edit only if new model is added)

Per-model image preprocessing parameters (crop size, normalization mean/std, interpolation). Do **not** need to change this for any supported model. Only edit it if a new model not currently in the list is added.

- `text_prompts.json` (zero-shot inference only)

Text prompts used by VLM models (MedSigLIP, BiomedCLIP, MedImageInsight) for zero-shot classification. Pre-populated for all supported datasets. **Only edit this if a new dataset is added** and want to customize the prompts passed to `--custom-text-prompts`:

```json
"MyNewDataset": {
    "binary": [
        "a chest x-ray image showing normal findings",
        "a chest x-ray image showing abnormal findings"
    ]
}
```

- `ark_zero_shot_config.py` (zero-shot inference only)

Configures Ark zero-shot inference. Contains two dictionaries:

- **`ARK_PRETRAINED_TASKS`** — describes the 6 task heads baked into the Ark checkpoint (MIMIC, CheXpert, ChestXray14, RSNA-Pneumonia, VinDr-CXR, Shenzhen). These are fixed by the Ark model weights. **Do not modify**.
- **`DATASET_LABEL_ALIASES`** — maps your dataset's class names to the nearest Ark pretrained label. **Add an entry here when adding a new dataset** to Ark zero-shot evaluation. For example:

```python
"MyNewDataset": {
    "pleural effusion": ["effusion", "pleural effusion"],
    "no finding":       ["no finding"],
}
```

---

## Running Experiments

### Training

Fine-tune or linear-probe a model:

```bash
accelerate launch rad_dino/run/train.py \
    --task multilabel \
    --data VinDr-CXR \
    --model rad-dino \
    --output-dir ./runs \
    --unfreeze-backbone \
    --optimize-compute --use-bf16
```

| Flag | Description |
|------|-------------|
| `--task` | `multilabel`, `multiclass`, or `binary` |
| `--data` | Dataset name (must match a key in `data_config.yaml`) |
| `--model` | Model identifier (e.g. `rad-dino`, `dinov2-large`, `medsiglip`, `ark`, `medimageinsight`, `biomedclip`) |
| `--output-dir` | Base directory for checkpoints (e.g. `./runs`) |
| `--unfreeze-backbone` | Unfreeze the pretrained backbone for fine-tuning (default: linear probe only) |
| `--kfold N` | K-fold cross-validation |
| `--optimize-compute` | Mixed-precision training (fp16) |
| `--use-bf16` | Use bf16 precision (requires `--optimize-compute`) |
| `--weighted-loss` | Apply class-weighted loss |
| `--wandb` | Enable Weights & Biases logging |
| `--resume` | Resume from checkpoint (requires `--resume-checkpoint-dir`) |
| `--pretrained-ark-path` | Path to Ark checkpoint (only for `--model ark`) |
| `--medimageinsight-path` | Path to MedImageInsight repo (only for `--model medimageinsight`) |

### Inference & Visualization

Evaluate a trained checkpoint and optionally generate attention / GradCAM maps:

```bash
python rad_dino/run/inference.py \
    --task binary \
    --data RSNA-Pneumonia \
    --model rad-dino \
    --model-path ./runs/checkpoints_... \
    --output-path ./inference_results \
    --optimize-compute \
    --show-attention --show-gradcam
```

### Zero-Shot Inference

Run zero-shot classification for VLM models (Ark, MedSigLIP, MedImageInsight, BiomedCLIP):

```bash
python rad_dino/run/zero_shot_inference.py \
    --task binary \
    --data TBX11K \
    --model medsiglip \
    --output-path ./experiments/zero_shot \
    --custom-text-prompts rad_dino/configs/text_prompts.json
```

### Feature-Based Evaluation (KNN / Linear SVM)

Extract frozen backbone features and evaluate with K-nearest neighbors or a linear SVM:

```bash
python rad_dino/run/knn.py \
    --task binary --data NODE21 --model rad-dino \
    --output-path ./experiments --batch-size 32 \
    --nb-knn 20 --temperature 0.07 --optimize-compute

python rad_dino/run/svm.py \
    --task binary --data NODE21 --model rad-dino \
    --output-path ./experiments --batch-size 32 \
    --max-iter 5000 --optimize-compute
```

### CKA Analysis

Compute layerwise or cross-model Centered Kernel Alignment:

```bash
python rad_dino/run/cka.py \
    --mode layerwise \
    --task multiclass --data VinDr-Mammo --model rad-dino \
    --checkpoint-dir ./runs/checkpoints_... \
    --output-path ./experiments/cka \
    --batch-size 32 --max-batches 200 --optimize-compute
```

### Embedding Visualization (UMAP / t-SNE)

Generate 2D scatter plots of backbone embeddings:

```bash
python rad_dino/run/visualize_embeddings.py \
    --task binary --data NODE21 --model rad-dino \
    --output-path ./experiments --batch-size 64 \
    --method umap --metric cosine --optimize-compute
```

---

## Project Structure

```
rad_dino/
├── configs/          # YAML/JSON configs (data paths, training hyperparams, model settings, text prompts)
├── data/             # Per-dataset preprocessing scripts and data leakage checker
├── eval/             # CKA analyzer, feature extractor, inference engine
├── loggings/         # Colored logging setup
├── models/           # Model classifiers (DINO, Ark, MedSigLIP, BiomedCLIP, MedImageInsight)
├── run/              # CLI entry points (train, inference, knn, svm, cka, umap, zero-shot)
├── train/            # Trainer, train utilities, model registry
└── utils/            # Transforms, loss utils, preprocessing, visualization helpers
docs/data/data.md     # Detailed dataset preprocessing guide
tests/                # Unit tests (run with: pytest tests/)
```
