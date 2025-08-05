# VinDr-Mammo Dataset Preprocessing

This document describes the preprocessing pipeline for the VinDr-Mammo dataset, which supports both single-view and multi-view mammography analysis.

## Dataset Overview

VinDr-Mammo is a large-scale mammography dataset containing 20,000 mammography images from 5,000 patients. Each patient has 4 images:
- Left breast: CC (cranio-caudal) and MLO (medio-lateral oblique) views
- Right breast: CC and MLO views

## Data Sources

The dataset provides two annotation files:
1. **`breast-level_annotations.csv`**: Contains BIRADS and density labels (one row per image)
2. **`finding_annotations.csv`**: Contains finding categories (multiple rows per image, multiple findings per image)

## Preprocessing Scripts

### 1. `prepare_vindrmammo_birad.py`
Specialized script for BIRADS and density classification tasks.

**Features:**
- Supports `breast_birads` and `breast_density` targets
- Single-view and multi-view processing
- Uses breast-level annotations (20,000 images)
- Clean implementation for BIRADS/density tasks

**Data Sizes:**
- Single-view: 16,000 train, 4,000 test images
- Multi-view: 4,000 train, 1,000 test studies

### 2. `prepare_vindrmammo_multilabel.py`
General-purpose script for finding categories (multilabel classification).

**Features:**
- Supports finding categories only
- Single-view and multi-view processing
- Uses finding annotations (20,486 images)
- Majority voting aggregation for multiple annotations per image

**Data Sizes:**
- Single-view: 16,391 train, 4,095 test images
- Multi-view: 4,000 train, 1,000 test studies

## Aggregation Methods

### Majority Voting
For finding categories with multiple annotations per image, we use majority voting:
- If more than 50% of annotations for an image have a category, it's marked as present (1)
- Otherwise, it's marked as absent (0)
- This reduces the effect of outlier annotations

### Mode Aggregation
For BIRADS and density (single annotation per image):
- Uses mode aggregation when multiple images per study have different labels
- Ensures consistent labeling across all views of a study

## Multi-View Processing

Multi-view processing is crucial for mammography analysis as it mimics radiologist workflow:

### Multi-View Structure
When `--multi-view` is enabled, the output structure becomes:
```
images/
├── train/
│   ├── study_id_1/
│   │   ├── L_CC.dcm
│   │   ├── L_MLO.dcm
│   │   ├── R_CC.dcm
│   │   └── R_MLO.dcm
│   └── study_id_2/
│       └── ...
└── test/
    └── ...
```

### Single-View Structure
Default structure (without `--multi-view`):
```
images/
├── train/
│   ├── image_id_1.dcm
│   ├── image_id_2.dcm
│   └── ...
└── test/
    └── ...
```

### Data Preprocessing Script
The main preprocessing script (`data_preprocessing.sh`) processes all datasets:

```bash
# Finding Categories (7 most common classes)
python ./rad_dino/data/VinDrMammo/prepare_vindrmammo_multilabel.py \
    --path-root /hpcwork/rwth1833/datasets/VinDr-Mammo/download/physionet.org/files/vindr-mammo/1.0.0 \
    --output-dir /hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/findings \
    --classes 7

# BIRADS Classification
python ./rad_dino/data/VinDrMammo/prepare_vindrmammo_birad.py \
    --path-root /hpcwork/rwth1833/datasets/VinDr-Mammo/download/physionet.org/files/vindr-mammo/1.0.0 \
    --output-dir /hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/birads \
    --target breast_birads

# Density Classification
python ./rad_dino/data/VinDrMammo/prepare_vindrmammo_birad.py \
    --path-root /hpcwork/rwth1833/datasets/VinDr-Mammo/download/physionet.org/files/vindr-mammo/1.0.0 \
    --output-dir /hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/density \
    --target breast_density
```

### Advanced Usage Examples

#### BIRADS Classification (Multi-View)
```bash
python -m rad_dino.data.VinDrMammo.prepare_vindrmammo_birad \
    --path-root /path/to/vindr-mammo \
    --output-dir /path/to/output \
    --target breast_birads \
    --multi-view
```

#### Finding Categories with Custom Classes
```bash
python -m rad_dino.data.VinDrMammo.prepare_vindrmammo_multilabel \
    --path-root /path/to/vindr-mammo \
    --output-dir /path/to/output \
    --classes "Mass" "Suspicious Calcification" "No Finding" \
    --multi-view
```

## Output Files

### For BIRADS/Density Tasks
- `train_labels.csv`: Training labels with numeric encoding
- `test_labels.csv`: Test labels with numeric encoding
- `label_mapping.csv`: Mapping from original labels to numeric indices

### For Finding Categories Tasks
- `train_labels_multilabel.csv`: Multi-hot encoded training labels
- `test_labels_multilabel.csv`: Multi-hot encoded test labels

## Dataset Statistics

### BIRADS Distribution
- BI-RADS 1: 13,406 images (67.0%)
- BI-RADS 2: 4,676 images (23.4%)
- BI-RADS 3: 930 images (4.7%)
- BI-RADS 4: 762 images (3.8%)
- BI-RADS 5: 226 images (1.1%)

### Density Distribution
- DENSITY A: 100 images (0.5%)
- DENSITY B: 1,908 images (9.5%)
- DENSITY C: 15,292 images (76.5%)
- DENSITY D: 2,700 images (13.5%)

### Finding Categories (Top 7)
- No Finding: 18,232 annotations (89.0%)
- Mass: 1,226 annotations (6.0%)
- Suspicious Calcification: 543 annotations (2.6%)
- Asymmetry: 392 annotations (1.9%)
- Architectural Distortion: 119 annotations (0.6%)
- Skin Thickening: 89 annotations (0.4%)
- Suspicious Lymph Node: 67 annotations (0.3%)

## Research Context

Multi-view mammography analysis is supported by recent research:
- [MamT4: Multi-view Attention Networks for Mammography Cancer Classification](https://arxiv.org/html/2411.01669v1)
- [Multi-view deep learning approaches for mammography analysis](https://arxiv.org/pdf/2112.04490)
- [Advanced multi-view mammography techniques](https://arxiv.org/pdf/2411.15802)

These studies demonstrate the importance of leveraging all four views (L-CC, L-MLO, R-CC, R-MLO) for improved diagnostic accuracy.

## Quality Assurance

### Data Validation
- Ensures all required image files exist before creating symlinks
- Validates label consistency across multiple annotations
- Checks for data leakage between train/test splits

### Performance Considerations
- Uses symlinks instead of copying files to save storage
- Efficient aggregation using pandas groupby operations
- Memory-efficient processing for large datasets

### Error Handling
- Graceful handling of missing annotations
- Proper error messages for missing image files
- Validation of input parameters and data formats 