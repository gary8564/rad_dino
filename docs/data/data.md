# Dataset Preprocessing Guide

This document describes the supported datasets, how to obtain them, and how to
run the preprocessing scripts that convert raw downloads into the format
expected by `RadImageClassificationDataset`.

Every preprocessing script produces:

- `train_labels.csv` (and optionally `val_labels.csv` / `test_labels.csv`) with
  an `image_id` (or `study_id` for multi-view) index and label columns.
- An `images/{split}/` directory containing copied source image files (DICOM,
  PNG, or MHA) by default.

All preprocessing scripts support an optional `--symlink` flag. By default,
files are copied into the preprocessed dataset directory. Pass `--symlink` if
you prefer lightweight references to the original files instead.

After preprocessing, update `rad_dino/configs/data_config.yaml` so that each dataset's `data_root_folder` points to the output directory.

---

## 0. Datasets Overview

| Dataset  | Modality | Task | Source | Data Size |
|:--------:|:--------:|:----:|:------:|:---------:|
| VinDr-CXR | CXR | Multilabel | [PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/) / [VinDr](https://vindr.ai/cxr) | 18,000 images (15,000 train / 3,000 test) |
| RSNA Pneumonia | CXR | Binary | [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) | 26,684 images |
| TAIX-Ray | ICU bedside CXR | Multilabel | [Hugging Face](https://huggingface.co/datasets/TLAIM/TAIX-Ray) | 200K images from 50K patients | 
| NODE21 | CXR | Binary | [Zenodo (training data)](https://zenodo.org/record/4725881) / [Grand Challenge](https://node21.grand-challenge.org/Data/) | 4,882 images |
| COVID-CXR | CXR | Binary | [Kaggle (COVIDx CXR-2)](https://www.kaggle.com/datasets/andyczhao/covidx-cxr2) | 13,975 CXRs across 13,870 patient cases |
| VinDr-PCXR | CXR | Multilabel | [PhysioNet](https://physionet.org/content/vindr-pcxr/1.0.0/) | 9,125 pediatric CXRs (7,728 training / 1,397 test) |
| TBX11K | CXR | Binary / Multiclass | [Kaggle](https://www.kaggle.com/datasets/usmanshams/tbx-11) | 11,200 CXRs (6,600 train / 1,800 val / 2,800 test) |
| VinDr-Mammo |  Full-field digital mammography | Multilabel / Multiclass / Binary | [PhysioNet](https://physionet.org/content/vindr-mammo/1.0.0/) / [VinDr](https://vindr.ai/datasets/mammo) | 20,000 images from 5,000 exams (4 views per exam) |
| VinDr-SpineXR | Spine X-ray | Multilabel | [PhysioNet](https://physionet.org/content/vindr-spinexr/1.0.0/) / [VinDr](https://vindr.ai/spinexr) | 10,466 spine radiographs from 5,000 studies |

## 1. VinDr-CXR

```bash
python ./rad_dino/data/VinDrCXR/prepare_vindrcxr.py \
    --path-root /path/to/VinDr-CXR \
    --output-dir /path/to/preprocessed/VinDr-CXR \
    --classes "Lung Opacity" "Cardiomegaly" "Pleural thickening" \
              "Aortic enlargement" "Pleural effusion" "Pulmonary fibrosis" \
              "Tuberculosis" "No finding"
```

Use `--classes N` (integer) to keep the top-N most prevalent classes, or
provide an explicit list of class names as shown above.

---

## 2. RSNA Pneumonia

```bash
python ./rad_dino/data/RSNAPneumonia/prepare_rsna_pneumonia.py \
    --path-root /path/to/RSNA-Pneumonia \
    --output-dir /path/to/preprocessed/RSNA-Pneumonia \
    --test-size 0.2
```

Since RSNA test set is not publicly accesible, the script creates a stratified train/test split from public training set.

---

## 3. VinDr-Mammo

VinDr-Mammo is always processed as **multi-view** (4 views per study: L-CC, L-MLO, R-CC, R-MLO).  Only studies with all four views are kept.

Two preprocessing scripts are available depending on the classification task:

### BI-RADS classification (multiclass)

```bash
python ./rad_dino/data/VinDrMammo/prepare_vindrmammo_birad.py \
    --path-root /path/to/vindr-mammo/1.0.0 \
    --output-dir /path/to/preprocessed/VinDr-Mammo/birads
```

### Binary classification (positive = BI-RADS 4 or 5)

```bash
python ./rad_dino/data/VinDrMammo/prepare_vindrmammo_binary.py \
    --path-root /path/to/vindr-mammo/1.0.0 \
    --output-dir /path/to/preprocessed/VinDr-Mammo/binary
```

---

## 4. TAIX-Ray

```bash
python ./rad_dino/data/TAIXRay/prepare_taixray.py \
    --path-root /path/to/TAIX-Ray \
    --output-dir /path/to/preprocessed/TAIX-Ray
```

---

## 5. NODE21

```bash
python ./rad_dino/data/Node21/prepare_node21.py \
    --path-root /path/to/NODE21/cxr_images/proccessed_data \
    --output-dir /path/to/preprocessed/NODE21 \
    --test-size 0.2
```

---

## 6. COVID-CXR

```bash
python ./rad_dino/data/COVIDCXR/prepare_covid_cxr.py \
    --path-root /path/to/covid-cxr \
    --output-dir /path/to/preprocessed/COVID-CXR
```

---

## 7. VinDr-PCXR

```bash
python ./rad_dino/data/VinDrPCXR/preprocess_vindrpcxr.py \
    --path-root /path/to/vindr-pcxr/1.0.0 \
    --output-dir /path/to/preprocessed/VinDr-PCXR
```

---

## 8. VinDr-SpineXR

```bash
python ./rad_dino/data/VinDrSpineXR/prepare_vindrspinexr.py \
    --path-root /path/to/vindr-spinexr \
    --output-dir /path/to/preprocessed/VinDr-SpineXR
```

---

## 9. TBX11K

```bash
python ./rad_dino/data/TBX11K/prepare_tbx11k.py \
    --path-root /path/to/TBX11K \
    --output-dir /path/to/preprocessed/TBX11K
```

This creates two subdirectories: `binary/` (TB vs healthy) and `multiclass/` (TB / sick / healthy). 
Point `data_config.yaml` to the appropriate subdirectory.

---

## Output Directory Structure

After preprocessing, each dataset directory follows this layout:

```
<output-dir>/
├── train_labels.csv        # image_id (or study_id) indexed label file
├── test_labels.csv
├── val_labels.csv          # (only COVID-CXR and TAIX-Ray)
├── label_mapping.csv       # (only VinDr-Mammo BI-RADS)
└── images/
    ├── train/
    │   ├── image_id_1.dcm  # copied files by default; symlinks with --symlink
    │   └── ...
    └── test/
        └── ...
```

For VinDr-Mammo the `images/{split}/` directories contain per-study subdirectories:

```
images/train/
├── study_001/
│   ├── L_CC.dcm
│   ├── L_MLO.dcm
│   ├── R_CC.dcm
│   └── R_MLO.dcm
└── ...
```

## Verifying Data Integrity

After preprocessing any dataset, you can check for data leakage (overlapping
sample IDs or patient IDs between splits) using the shared checker:

```bash
python -m rad_dino.data.check_data_leakage \
    --data-dir /path/to/preprocessed/RSNA-Pneumonia
```


## Configuration

After preprocessing, update `rad_dino/configs/data_config.yaml` so that each
dataset's `data_root_folder` points to the corresponding output directory.
See the [README](../../README.md) for details on running training and evaluation.
