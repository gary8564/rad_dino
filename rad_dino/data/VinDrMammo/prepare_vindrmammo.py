import os
import argparse
import pandas as pd
import ast
import logging
from typing import Union
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the preprocessed output directory of the dataset")
    parser.add_argument("--classes", nargs="+", default=None, help="Specify an integer k for top-k classes or a list of subset of class names.")
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

def prepare_vindrmammo(df, split_name, class_labels):
    """
    Prepare the VinDr-Mammo dataset for training and testing.
    
    Args:
        df: The original DataFrame containing the dataset.
        split_name: The name of the split to prepare ("training" or "test").
        class_labels: The list of class labels in the dataset.
    Returns:
        df_split_agg: The aggregated DataFrame containing the class labels for each image.
    """
    # 1) FILTER BY SPLIT
    df_split = df[df['split'] == split_name].copy()
    
    # 2) FOR EACH ANNOTATION (EACH ROW), ADD A 0/1 COLUMN FOR EACH CATEGORY
    for category in class_labels:
        df_split[category] = df_split['finding_categories'].apply(lambda x: 1 if category in x else 0)
        
    # 3) GROUP BY IMAGE_ID AND TAKE THE MODE OF THE CATEGORIES, ALSO PRESERVE STUDY_ID
    df_split_agg = df_split.groupby('image_id').agg({
        **{category: lambda s: 1 if (len(s.mode()) == 2) else s.mode().iloc[0] for category in class_labels},
        'study_id': 'first'  # Keep the study_id for each image
    })
    return df_split_agg

def main():
    # 0) PARSE ARGUMENTS
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1) READ THE ORIGINAL CSV
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
    df_train_agg = prepare_vindrmammo(df_train_raw, "training", all_categories)
    df_test_agg  = prepare_vindrmammo(df_test_raw, "test", all_categories)
    
    # 6) SAVE THE LABELS
    # filter out study_id column
    df_train = df_train_agg.drop(columns=['study_id'])
    df_test = df_test_agg.drop(columns=['study_id'])
    df_train.to_csv(os.path.join(args.output_dir, "train_labels.csv"), index=True)
    df_test.to_csv(os.path.join(args.output_dir, "test_labels.csv"), index=True)
    
    # 7) SYMLINK IMAGE FILES
    src_images_folder = os.path.join(args.path_root, "images")
    for split, df_agg in [("train", df_train_agg), ("test", df_test_agg)]:
        dst_folder = os.path.join(args.output_dir, "images", split)
        # Create destination directory if it doesn't exist
        os.makedirs(dst_folder, exist_ok=True)
        for image_id in df_agg.index:
            # Get the study_id for this image_id
            study_id = df_agg.loc[image_id, 'study_id']
            
            # Construct the source path: images/{study_id}/{image_id}.dicom
            src_file = os.path.join(src_images_folder, study_id, f"{image_id}.dicom")
            
            if not os.path.exists(src_file):
                raise FileNotFoundError(f"Image file for {image_id} not found at {src_file}")
                
            # Create symlink with .dcm extension for consistency
            dst = os.path.join(dst_folder, f"{image_id}.dcm")
            if not os.path.exists(dst):
                os.symlink(src_file, dst)
            else:
                os.remove(dst)
                os.symlink(src_file, dst)
    
    logger.info(f"Preprocessing VinDr-Mammo complete! The processed dataset is saved in {args.output_dir}")

if __name__ == "__main__":
    main()


