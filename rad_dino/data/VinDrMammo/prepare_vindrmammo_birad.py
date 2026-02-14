import os
import argparse
import pandas as pd
import ast
import logging
import numpy as np
from typing import Union, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the preprocessed output directory of the dataset")
    parser.add_argument("--multi-view", action='store_true', 
                       help="Enable multi-view processing (stack 4 images per study)")
    return parser.parse_args()

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Get unique values from a column, excluding header row."""
    values = df[column].unique()
    # Remove header-like values
    values = [v for v in values if v != column and pd.notna(v)]
    return sorted(values)

def select_highest_birad(values: pd.Series) -> str:
    """
    Select the maximum BI-RADS score among the four views for a study.
    Args:
        values: Series of BI-RADS values for a study (4 views)
    Returns:
        The highest BI-RADS value
    """
    # BI-RADS severity order: 1 < 2 < 3 < 4 < 5
    severity_order = {
        'BI-RADS 1': 1,
        'BI-RADS 2': 2, 
        'BI-RADS 3': 3,
        'BI-RADS 4': 4,
        'BI-RADS 5': 5
    }
    max_birad = max(values, key=lambda x: severity_order.get(x, 0))
    logger.info(f"Selected max BI-RADS value from views {values.tolist()}: {max_birad}")
    return max_birad

def prepare_vindrmammo_birad(df: pd.DataFrame, split_name: str, 
                            multi_view: bool = False) -> pd.DataFrame:
    """
    Prepare the VinDr-Mammo dataset for BI-RADS classification.
    
    Args:
        df: The original DataFrame containing the dataset.
        split_name: The name of the split to prepare ("training" or "test").
        multi_view: Whether to group by study_id for multi-view processing.
    Returns:
        df_split_agg: The aggregated DataFrame containing the BI-RADS labels.
    """
    # 1) FILTER BY SPLIT
    df_split = df[df['split'] == split_name].copy()
    
    if multi_view:
        # Group by study_id for multi-view processing
        # Each study has 4 images (L-CC, L-MLO, R-CC, R-MLO)
        df_split_agg = df_split.groupby('study_id').agg({
            'image_id': list,  # Keep all image IDs for this study
            'breast_birads': lambda x: select_highest_birad(x),  
            'laterality': list,  # Keep laterality info
            'view_position': list,  # Keep view position info
        })
        
        logger.info(f"Multi-view processing: {len(df_split_agg)} studies in {split_name} set")
        
    else:
        # Single-view processing (one image per sample)
        # Since image_id is unique, we can simply set it as index
        df_split_agg = df_split.set_index('image_id')[['study_id', 'laterality', 'view_position', 'breast_birads']]
        
        logger.info(f"Single-view processing: {len(df_split_agg)} images in {split_name} set")
    
    return df_split_agg

def _create_symlink(src: str, dst: str):
    """Create a symlink, replacing any existing one."""
    try:
        os.symlink(src, dst)
    except FileExistsError:
        os.remove(dst)
        os.symlink(src, dst)

def create_multi_view_structure(df_agg: pd.DataFrame, output_dir: str, split: str, 
                               src_images_folder: str) -> None:
    """
    Create multi-view structure by organizing images by study_id.
    
    Args:
        df_agg: Aggregated DataFrame with study_id and image_id lists.
        output_dir: Output directory path.
        split: Split name ('train' or 'test').
        src_images_folder: Source images folder path.
    """
    dst_folder = os.path.join(output_dir, "images", split)
    os.makedirs(dst_folder, exist_ok=True)
    
    symlink_pairs = []
    for study_id, row in df_agg.iterrows():
        image_ids = row['image_id']
        lateralities = row['laterality']
        view_positions = row['view_position']
        
        study_dir = os.path.join(dst_folder, study_id)
        os.makedirs(study_dir, exist_ok=True)
        
        for i, image_id in enumerate(image_ids):
            laterality = lateralities[i]
            view_pos = view_positions[i]
            src_file = os.path.join(src_images_folder, study_id, f"{image_id}.dicom")
            dst = os.path.join(study_dir, f"{laterality}_{view_pos}.dcm")
            symlink_pairs.append((src_file, dst))
    
    create_symlinks_parallel(symlink_pairs, raise_on_missing=True)
    logger.info(f"Symlinked {len(symlink_pairs)} multi-view images to {dst_folder}")

def main():
    # 0) PARSE ARGUMENTS
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1) READ THE ORIGINAL CSV
    df = pd.read_csv(os.path.join(args.path_root, "breast-level_annotations.csv"))
    
    # 2) GET UNIQUE BI-RADS VALUES
    unique_values = get_unique_values(df, 'breast_birads')
    logger.info(f"Unique BI-RADS values: {unique_values}")
    
    # 3) CREATE LABEL MAPPING
    label_mapping = {value: idx for idx, value in enumerate(unique_values)}
    logger.info(f"Label mapping: {label_mapping}")
    
    # 4) SPLIT BASED ON THE 'split' COLUMN
    df_train_raw = df[df["split"] == "training"].copy()
    df_test_raw = df[df["split"] == "test"].copy()
    
    df_train_agg = prepare_vindrmammo_birad(df_train_raw, "training", args.multi_view)
    df_test_agg = prepare_vindrmammo_birad(df_test_raw, "test", args.multi_view)
    
    # 5) CONVERT LABELS TO NUMERIC
    for df_agg in [df_train_agg, df_test_agg]:
        df_agg['label'] = df_agg['breast_birads'].map(label_mapping)
    
    # 6) SAVE THE LABELS
    if args.multi_view:
        # For multi-view, save study-level labels
        df_train_labels = df_train_agg[['label']].copy()
        df_test_labels = df_test_agg[['label']].copy()
    else:
        # For single-view, save image-level labels
        df_train_labels = df_train_agg[['label']].copy()
        df_test_labels = df_test_agg[['label']].copy()
    
    df_train_labels.to_csv(os.path.join(args.output_dir, "train_labels.csv"))
    df_test_labels.to_csv(os.path.join(args.output_dir, "test_labels.csv"))
    
    # 7) SAVE LABEL MAPPING
    mapping_df = pd.DataFrame(list(label_mapping.items()), columns=['label', 'index'])
    mapping_df.to_csv(os.path.join(args.output_dir, "label_mapping.csv"), index=False)
    
    # 8) SYMLINK IMAGE FILES
    src_images_folder = os.path.join(args.path_root, "images")
    
    if args.multi_view:
        # Multi-view structure: organize by study_id
        create_multi_view_structure(df_train_agg, args.output_dir, "train", src_images_folder)
        create_multi_view_structure(df_test_agg, args.output_dir, "test", src_images_folder)
    else:
        # Single-view structure: flat organization
        for split, df_agg in [("train", df_train_agg), ("test", df_test_agg)]:
            dst_folder = os.path.join(args.output_dir, "images", split)
            os.makedirs(dst_folder, exist_ok=True)
            symlink_pairs = [
                (os.path.join(src_images_folder, df_agg.loc[image_id, 'study_id'], f"{image_id}.dicom"),
                 os.path.join(dst_folder, f"{image_id}.dcm"))
                for image_id in df_agg.index
            ]
            create_symlinks_parallel(symlink_pairs, raise_on_missing=True)
            logger.info(f"Symlinked {len(symlink_pairs)} images to {dst_folder}")
    
    logger.info("Preprocessing VinDr-Mammo BI-RADS complete!")
    logger.info(f"Multi-view: {args.multi_view}")
    logger.info(f"Number of classes: {len(unique_values)}")
    logger.info(f"Classes: {unique_values}")
    logger.info(f"The processed dataset is saved in {args.output_dir}")

if __name__ == "__main__":
    main()