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
pip install -e ".[wandb]"
```

### 2. Prerequisites

#### 2.1 HuggingFace Authentication

Several models (DINOv2, DINOv3, RAD-DINO, MedSigLIP) are downloaded from HuggingFace Hub at runtime. Gated models such as MedSigLIP require an access
token.
Setup [HuggingFace access token](https://huggingface.co/settings/tokens) before running the experiments.


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
After preprocessing, update the dataset root paths to point to your preprocessed output directories.

```yaml
VinDr-CXR:
  data_root_folder: "/path/to/preprocessed/VinDr-CXR"
  num_workers: 4
```

To add a new dataset, simply add an entry here — all scripts validate `--data` against this file at runtime, so no Python source changes are needed. Everything else (task, model, output paths) is passed as CLI flags.

- `train_config.yaml` (optional)

Default training hyperparameters used for all experiments. Edit this if you want to try different values:

```yaml
batch_size: 20
epochs: 100
optim:
  base_lr: 1e-5
  weight_decay: 0.001
early_stopping:
  patience: 10
```

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

## Running Experiments

### Training

- Linear probing (LP): 
```bash
accelerate launch rad_dino/run/train.py \
    --task multilabel \
    --data VinDr-CXR \
    --model rad-dino \
    --output-dir /path/to/output/dir \
    --optimize-compute --use-bf16
```

- Fine-tuning (FT): 
```bash
accelerate launch rad_dino/run/train.py \
    --task multilabel \
    --data VinDr-CXR \
    --model rad-dino \
    --output-dir /path/to/output/dir \
    --unfreeze-backbone \
    --optimize-compute --use-bf16
```

**Required args:**
| Flag | Description |
|------|-------------|
| `--task` | Classification task: `multilabel`, `multiclass`, or `binary` |
| `--data` | Dataset name (must match a key in `data_config.yaml`) |
| `--model` | Model identifier: `rad-dino`, `dinov2-{small,base,large}`, `dinov2-large-reg`, `dinov3-{small-plus,base,large}`, `medsiglip`, `ark`, `medimageinsight`, `biomedclip` |
| `--output-dir` | Base directory for checkpoints (e.g. `./runs`) |

**Optional args:**
| Flag | Description |
|------|-------------|
| `--unfreeze-backbone` | Unfreeze the pretrained backbone for fine-tuning (default: linear probe only) |
| `--unfreeze-num-layers N` | Number of transformer blocks to unfreeze from the end (requires `--unfreeze-backbone`) |
| `--progressive-unfreeze` | Progressively unfreeze backbone layers over epochs (requires `--unfreeze-backbone`) |
| `--kfold N` | K-fold cross-validation |
| `--train-subset F` | Fraction of training data to use (0–1), for data-efficiency studies |
| `--weighted-loss` | Apply class-frequency-weighted loss |
| `--optimize-compute` | Enable mixed-precision training (fp16) |
| `--use-bf16` | Use bf16 instead of fp16 (requires `--optimize-compute`) |
| `--grad-accumulation-steps N` | Gradient accumulation micro-steps per optimizer step |
| `--grad-checkpointing` | Enable gradient checkpointing to reduce activation memory |
| `--compile` | Compile model with `torch.compile` for faster training |
| `--return-output-attentions` | Compute and return attention maps during training (memory-intensive) |
| `--wandb` | Enable Weights & Biases logging |
| `--resume` | Resume from checkpoint (requires `--resume-checkpoint-dir`) |
| `--resume-checkpoint-dir PATH` | Directory of checkpoint to resume from |
| `--pretrained-ark-path PATH` | Path to Ark pre-trained checkpoint (required for `--model ark`) |
| `--medimageinsight-path PATH` | Path to cloned MedImageInsights repo |

---

### Inference & Visualization

Evaluate a fine-tuned checkpoint on the test set, with optional explainability outputs.

> [!NOTE]
> Ark+ (Swin-L) and MedImageInsight (DaViT) use hierarchical architectures that do not produce meaningful global attention maps. Use `--show-feature-maps` for these models instead of `--show-attention`.

```bash
python rad_dino/run/inference.py \
    --task binary \
    --data RSNA-Pneumonia \
    --model rad-dino \
    --model-path /path/to/model/checkpoints \
    --output-path /path/to/output/dir \
    --optimize-compute
```

**Required args:**

| Flag | Description |
|------|-------------|
| `--task` | Classification task: `multilabel`, `multiclass`, or `binary` |
| `--data` | Dataset name (must match a key in `data_config.yaml`) |
| `--model` | Model identifier (same choices as training) |
| `--model-path PATH` | Path to the saved checkpoint directory |
| `--output-path PATH` | Directory to write results (metrics, figures) |

**Optional args:**

| Flag | Description |
|------|-------------|
| `--batch-size N` | Inference batch size |
| `--optimize-compute` | Enable mixed-precision inference (fp16) |
| `--compile` | Compile model with `torch.compile` for faster inference |
| `--show-attention` | Save last-layer attention overlays (ViT/SigLIP models only; requires `--save-heads` and `--attention-threshold`) |
| `--attention-threshold F` | Threshold for attention masking (required with `--show-attention`) |
| `--save-heads {mean,max,min}` | Which attention heads to aggregate (required with `--show-attention`) |
| `--compute-rollout` | Compute attention rollout in addition to raw attention maps (requires `--show-attention`) |
| `--compute-gradient-rollout` | Class-specific gradient rollout (ViT, BiomedCLIP, MedSigLIP models) |
| `--show-gradcam` | Save GradCAM overlays |
| `--show-feature-maps` | Save stage-wise feature map visualizations (Ark and MedImageInsight only) |
| `--max-visualization-samples N` | Maximum number of samples to generate visualizations for |
| `--min-positive-visualization-labels N` | Minimum positive-target coverage when selecting visualization samples |
| `--visualization-sample-ids PATH` | Text file with specified sample ID for reproducibility |
| `--medimageinsight-path PATH` | Path to cloned MedImageInsights repo |

---

### Zero-Shot Inference

Run zero-shot classification for MedSigLIP, BiomedCLIP, MedImageInsight, and Ark.

```bash
python rad_dino/run/zero_shot_inference.py \
    --task binary \
    --data TBX11K \
    --model medsiglip \
    --output-path /path/to/output/dir \
    --custom-text-prompts rad_dino/configs/text_prompts.json
```

**Required args:**

| Flag | Description |
|------|-------------|
| `--task` | Classification task: `multilabel`, `multiclass`, or `binary` |
| `--data` | Dataset name (must match a key in `data_config.yaml`) |
| `--model` | One of: `medsiglip`, `biomedclip`, `medimageinsight`, `ark` |
| `--output-path PATH` | Directory to write results |
| `--custom-text-prompts PATH` | Path to text prompts JSON file (required for `medsiglip`, `biomedclip`, `medimageinsight`) |

**Optional args:**

| Flag | Description |
|------|-------------|
| `--batch-size N` | Inference batch size |
| `--device` | Device to run on (`cuda` or `cpu`) |
| `--ark-checkpoint-path PATH` | Path to Ark pre-trained checkpoint (required for `--model ark`) |
| `--use-rsna-head` | Use the Ark pretrained RSNA head for binary classification (only for `--model ark --data RSNA-Pneumonia --task binary`) |
| `--medimageinsight-path PATH` | Path to cloned MedImageInsights repo (required for `--model medimageinsight`) |

---

### Feature-Based Evaluation (KNN / Linear SVM)

Extract frozen backbone features and evaluate with KNN or a linear SVM. 

> [!WARNING]
> Only `binary` and `multiclass` tasks are supported.

- KNN: 
```bash
python rad_dino/run/knn.py \
    --task binary --data NODE21 --model rad-dino \
    --output-path /path/to/knn/results \
    --nb-knn 20 --temperature 0.07
```

- SVM:
```bash
python rad_dino/run/svm.py \
    --task binary --data NODE21 --model rad-dino \
    --output-path /path/to/svm/results \
    --max-iter 5000
```

**Required args:**

| Flag | Description |
|------|-------------|
| `--task` | `binary` or `multiclass` (`multilabel` not supported) |
| `--data` | Dataset name (must match a key in `data_config.yaml`) |
| `--model` | Model identifier: `rad-dino`, `dinov2-large`, `dinov3-large`, `medsiglip`, `ark`, `medimageinsight`, `biomedclip` |
| `--output-path PATH` | Directory to write results |

**Optional args:**

| Flag | Description |
|------|-------------|
| `--batch-size N` | Feature extraction batch size |
| `--optimize-compute` | Enable mixed-precision feature extraction (fp16) |
| `--nb-knn N [N ...]` | Number(s) of nearest neighbours to evaluate (KNN only) |
| `--temperature F` | Softmax temperature for KNN voting (KNN only) |
| `--max-iter N` | Max iterations for LinearSVC (SVM only) |
| `--pretrained-ark-path PATH` | Path to Ark pre-trained checkpoint (required for `--model ark`) |
| `--medimageinsight-path PATH` | Path to cloned MedImageInsights repo (required for `--model medimageinsight`) |

---

### Centered Kernel Alignment (CKA) Analysis

- Layerwise CKA:
```bash
python rad_dino/run/cka.py \
    --mode layerwise \
    --task multiclass \
    --data VinDr-Mammo \
    --model dinov2-large \
    --checkpoint-dir /path/to/finetuned/checkpoint \
    --output-path /path/to/output/dir
```

- Cross-model CKA: 
```bash
python rad_dino/run/cka.py \
    --mode crossmodel \
    --task binary \
    --data TBX11K \
    --models dinov2-large rad-dino medsiglip \
    --checkpoint-dirs /path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3 \
    --output-path /path/to/output/dir
```

**Required args:**

| Flag | Description |
|------|-------------|
| `--mode` | `layerwise` (pretrained vs fine-tuned) or `crossmodel` (across models) |
| `--task` | Classification task: `multilabel`, `multiclass`, or `binary` |
| `--data` | Dataset name (must match a key in `data_config.yaml`) |
| `--output-path PATH` | Directory to write results |
| `--model NAME` | Model name — for `layerwise` mode |
| `--checkpoint-dir PATH` | Fine-tuned checkpoint directory — for `layerwise` mode |
| `--models NAME [NAME ...]` | List of model names — for `crossmodel` mode |
| `--checkpoint-dirs PATH [PATH ...]` | List of checkpoint directories, one per model — for `crossmodel` mode |

**Optional args:**

| Flag | Description |
|------|-------------|
| `--batch-size N` | Batch size for feature extraction |
| `--max-batches N` | Limit batches used for CKA (omit to use the full test set; CKA often converges on a subset) |
| `--optimize-compute` | Enable mixed-precision (fp16) |
| `--pretrained-ark-path PATH` | Path to Ark pre-trained checkpoint (required for `--model ark`) |
| `--medimageinsight-path PATH` | Path to cloned MedImageInsights repo (required for `--model medimageinsight`) |

---

### Embedding Visualization (UMAP / t-SNE)

Generate 2D scatter plots of frozen backbone embeddings. 
> [!WARNING]
> Only `binary` and `multiclass` tasks are supported.

```bash
python rad_dino/run/visualize_embeddings.py \
    --task binary --data NODE21 --model rad-dino \
    --output-path ./experiments/embeddings \
    --method umap --metric cosine
```

**Required args:**

| Flag | Description |
|------|-------------|
| `--task` | `binary` or `multiclass` |
| `--data` | Dataset name (must match a key in `data_config.yaml`) |
| `--model` | Model identifier |
| `--output-path PATH` | Directory to write plots |

**Optional args:**

| Flag | Description |
|------|-------------|
| `--method` | Dimensionality reduction method: `umap`, `tsne`, or `supervised-umap` |
| `--batch-size N` | Feature extraction batch size |
| `--metric` | Distance metric: `cosine`, `euclidean`, or `correlation` |
| `--optimize-compute` | Enable mixed-precision feature extraction (fp16) |
| `--n-neighbors N` | UMAP locality parameter (ignored for t-SNE) |
| `--min-dist F` | UMAP minimum distance (ignored for t-SNE) |
| `--perplexity F` | t-SNE perplexity (ignored for UMAP) |
| `--learning-rate F` | t-SNE learning rate (ignored for UMAP) |
| `--n-iter N` | t-SNE maximum iterations (ignored for UMAP) |
| `--random-state N` | Random seed for reproducibility |
| `--pretrained-ark-path PATH` | Path to Ark pre-trained checkpoint (required for `--model ark`) |
| `--medimageinsight-path PATH` | Path to cloned MedImageInsights repo (required for `--model medimageinsight`) |

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
