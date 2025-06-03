import os
import argparse
import pandas as pd
import logging
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True, 
                       help="Path to the root directory of the dataset")
    return parser.parse_args()

def check_data_leakage():
    """
    Check if there are any study_ids that appear in both training and test splits.
    This would indicate potential data leakage.
    """
    args = parse_args()
    
    # Read the finding annotations CSV
    df = pd.read_csv(os.path.join(args.path_root, "finding_annotations.csv"))
    
    # Get unique study_ids for each split
    train_study_ids = set(df[df['split'] == 'training']['study_id'].unique())
    test_study_ids = set(df[df['split'] == 'test']['study_id'].unique())
    
    # Check for overlap
    overlapping_study_ids = train_study_ids.intersection(test_study_ids)
    
    # Print statistics
    logger.info(f"Total unique study_ids in training set: {len(train_study_ids)}")
    logger.info(f"Total unique study_ids in test set: {len(test_study_ids)}")
    logger.info(f"Overlapping study_ids between train and test: {len(overlapping_study_ids)}")
    
    if overlapping_study_ids:
        logger.warning("DATA LEAKAGE DETECTED!")
        logger.warning(f"The following {len(overlapping_study_ids)} study_ids appear in both train and test sets:")
        for study_id in sorted(overlapping_study_ids):
            logger.warning(f"  - {study_id}")
        
        # Show detailed breakdown for overlapping studies
        logger.info("\nDetailed breakdown for overlapping studies:")
        for study_id in sorted(list(overlapping_study_ids)[:5]):  # Show first 5 as example
            train_images = df[(df['study_id'] == study_id) & (df['split'] == 'training')]['image_id'].nunique()
            test_images = df[(df['study_id'] == study_id) & (df['split'] == 'test')]['image_id'].nunique()
            logger.info(f"  Study {study_id}: {train_images} images in train, {test_images} images in test")
        
        return False
    else:
        logger.info("NO DATA LEAKAGE DETECTED!")
        logger.info("All study_ids are properly separated between train and test sets.")
        return True

def analyze_split_distribution():
    """
    Analyze the distribution of images and studies across splits.
    """
    args = parse_args()
    
    # Read the finding annotations CSV
    df = pd.read_csv(os.path.join(args.path_root, "finding_annotations.csv"))
    
    logger.info("\n=== SPLIT DISTRIBUTION ANALYSIS ===")
    
    # Overall statistics
    total_images = df['image_id'].nunique()
    total_studies = df['study_id'].nunique()
    total_annotations = len(df)
    
    logger.info(f"Dataset overview:")
    logger.info(f"  Total unique images: {total_images}")
    logger.info(f"  Total unique studies: {total_studies}")
    logger.info(f"  Total annotations: {total_annotations}")
    
    # Split-wise statistics
    for split in ['training', 'test']:
        split_df = df[df['split'] == split]
        split_images = split_df['image_id'].nunique()
        split_studies = split_df['study_id'].nunique()
        split_annotations = len(split_df)
        
        logger.info(f"\n{split.capitalize()} set:")
        logger.info(f"  Unique images: {split_images} ({split_images/total_images*100:.1f}%)")
        logger.info(f"  Unique studies: {split_studies} ({split_studies/total_studies*100:.1f}%)")
        logger.info(f"  Annotations: {split_annotations} ({split_annotations/total_annotations*100:.1f}%)")
        
        # Images per study statistics
        images_per_study = split_df.groupby('study_id')['image_id'].nunique()
        logger.info(f"  Images per study - Mean: {images_per_study.mean():.2f}, "
                   f"Min: {images_per_study.min()}, Max: {images_per_study.max()}")

if __name__ == "__main__":
    # Check for data leakage
    no_leakage = check_data_leakage()
    
    # Analyze split distribution
    analyze_split_distribution()
    
    if not no_leakage:
        logger.error("\nData leakage detected! You should fix the split before training.")
        exit(1)
    else:
        logger.info("\nData splits look good! No leakage detected.") 