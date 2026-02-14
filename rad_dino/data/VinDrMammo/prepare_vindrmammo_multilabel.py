import os
import argparse
import pandas as pd
import ast
import logging
from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the preprocessed output directory of the dataset")
    parser.add_argument("--classes", nargs="+", default=None, help="Specify an integer k for top-k classes or a list of subset of class names.")
    parser.add_argument("--multi-view", action='store_true', 
                       help="Enable multi-view processing (stack 4 images per study)")
    return parser.parse_args()

def convert_finding_categories(finding_categories):
    return ast.literal_eval(finding_categories)

def merge_asymmetry_labels(finding_categories):
    """
    Merge Asymmetry, Global Asymmetry, and Focal Asymmetry into a single 'Asymmetry' label.
    
    Args:
        finding_categories: List of category labels for a single annotation.
    Returns:
        List of category labels with asymmetry variants merged.
    """
    asymmetry_variants = ['Asymmetry', 'Global Asymmetry', 'Focal Asymmetry']
    
    # Check if any asymmetry variant is present
    has_asymmetry = any(variant in finding_categories for variant in asymmetry_variants)
    
    if has_asymmetry:
        # Remove all asymmetry variants
        filtered_categories = [cat for cat in finding_categories if cat not in asymmetry_variants]
        # Add the unified 'Asymmetry' label
        filtered_categories.append('Asymmetry')
        return filtered_categories
    else:
        return finding_categories

def filter_categories_by_frequency(df, topk: int):
    """
    Filter categories based on frequency.
    
    Args:
        df: The DataFrame containing the dataset with 'finding_categories' column.
        topk: The number of top-k classes to select.
    Returns:
        A list of the filtered class labels.
    """
    if not isinstance(topk, int):
        raise ValueError("`topk` must be an integer.")
    if topk < 1:
        raise ValueError("`topk` must be greater than 0.")
    
    # Count frequency of each category across all annotations
    category_counts = {}
    for finding_categories in df['finding_categories']:
        for category in finding_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
    
    # Convert to pandas Series for easier manipulation
    category_freq = pd.Series(category_counts).sort_values(ascending=False)
    
    # Select top-k most frequent categories
    filtered_categories = category_freq.head(topk).index.tolist()
    logger.info(f"Selected top-{topk} categories: {filtered_categories}")
    logger.info(f"Category frequencies: {dict(category_freq.head(topk))}")
    return filtered_categories

def prepare_vindrmammo(df, split_name, class_labels, multi_view=False):
    """
    Prepare the VinDr-Mammo dataset for training and testing.
    
    Args:
        df: The original DataFrame containing the dataset.
        split_name: The name of the split to prepare ("training" or "test").
        class_labels: The list of class labels in the dataset.
        multi_view: Whether to group by study_id for multi-view processing.
    Returns:
        df_split_agg: The aggregated DataFrame containing the class labels for each image.
    """
    df_split = df[df['split'] == split_name].copy()
    
    if multi_view:
        # Group by study_id for multi-view processing
        # Each study has 4 images (L-CC, L-MLO, R-CC, R-MLO)
        # For finding categories, we need to handle the list structure
        df_split_agg = df_split.groupby('study_id').agg({
            'image_id': list,  # Keep all image IDs for this study
            'finding_categories': list,  # Keep all finding categories
            'laterality': list,  # Keep laterality info
            'view_position': list,  # Keep view position info
        })
        
        # Process finding categories for multi-view
        for category in class_labels:
            df_split_agg[category] = df_split_agg['finding_categories'].apply(
                lambda x: 1 if sum(1 for cat_list in x if category in cat_list) > len(x) / 2 else 0
            )
        logger.info(f"Multi-view processing: {len(df_split_agg)} studies in {split_name} set")
        
    else:
        # Single-view processing (one image per sample)
        # Since image_id might have multiple annotations, we need to aggregate first
        # First add the category columns
        for category in class_labels:
            df_split[category] = df_split['finding_categories'].apply(lambda x: 1 if category in x else 0)
        
        # Group by image_id and aggregate (take majority vote for binary categories, first for other columns)
        df_split_agg = df_split.groupby('image_id').agg({
            'study_id': 'first',  
            **{category: lambda x: 1 if len(x.mode()) == 1 and x.mode().iloc[0] == 1 else 0 for category in class_labels}  # Take majority vote, default to 0 on tie
        })
        
        logger.info(f"Single-view processing: {len(df_split_agg)} images in {split_name} set")
    
    return df_split_agg

def _create_symlink(src: str, dst: str):
    """Create a symlink, replacing any existing one."""
    try:
        os.symlink(src, dst)
    except FileExistsError:
        os.remove(dst)
        os.symlink(src, dst)

def create_multi_view_structure(df_agg, output_dir, split, src_images_folder):
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
    
    # Build (src, dst) pairs and create study directories
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
    
    # 1) READ THE FINDING ANNOTATIONS CSV
    df = pd.read_csv(os.path.join(args.path_root, "finding_annotations.csv"))
    
    # 2) PARSE THE STRING‐ENCODED LIST INTO A PYTHON LIST
    df['finding_categories'] = df['finding_categories'].apply(lambda x: convert_finding_categories(x))
    
    # 3) MERGE ASYMMETRY LABELS INTO A SINGLE CATEGORY
    logger.info("Merged Asymmetry, Global Asymmetry, and Focal Asymmetry into unified 'Asymmetry' label")
    df['finding_categories'] = df['finding_categories'].apply(merge_asymmetry_labels)
    
    # 4) DETERMINE CLASS LABELS BASED ON FILTERING
    if args.classes is None:
        # Use all categories if no filtering specified
        all_categories = sorted({cat 
                                 for sublist in df['finding_categories'] 
                                 for cat in sublist})
        logger.info(f"Using all categories of VinDr-Mammo: {all_categories}")
    else:
        if len(args.classes) == 1:
            try:
                # Try to parse as integer for top-k selection
                topk = int(args.classes[0])
                all_categories = filter_categories_by_frequency(df, topk)
            except ValueError:
                # If not integer, treat as single class name
                all_categories = args.classes
        else:
            # Multiple class names provided
            all_categories = args.classes
    
    # 5) SPLIT BASED ON THE 'split' COLUMN
    df_train_raw = df[df["split"] == "training"].copy()
    df_test_raw  = df[df["split"] == "test"].copy()
    df_train_agg = prepare_vindrmammo(df_train_raw, "training", all_categories, args.multi_view)
    df_test_agg  = prepare_vindrmammo(df_test_raw, "test", all_categories, args.multi_view)
    
    # 6) SAVE THE LABELS
    if args.multi_view:
        # For multi-view, save study-level labels
        df_train = df_train_agg[all_categories].copy()
        df_test = df_test_agg[all_categories].copy()
    else:
        # For single-view, save image-level labels
        df_train = df_train_agg.drop(columns=['study_id'])
        df_test = df_test_agg.drop(columns=['study_id'])
    
    df_train.to_csv(os.path.join(args.output_dir, "train_labels.csv"), index=True)
    df_test.to_csv(os.path.join(args.output_dir, "test_labels.csv"), index=True)
    
    # 7) SYMLINK IMAGE FILES
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
    
    logger.info("Preprocessing VinDr-Mammo complete!")
    logger.info(f"Multi-view: {args.multi_view}")
    logger.info(f"Number of classes: {len(all_categories)}")
    logger.info(f"Classes: {all_categories}")
    logger.info(f"The processed dataset is saved in {args.output_dir}")

if __name__ == "__main__":
    main()


