import os
import argparse
import pandas as pd
import logging
from typing import Union
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the preprocessed output directory of the dataset")
    return parser.parse_args()

def create_multilabel_encoding(df_annot: pd.DataFrame) -> pd.DataFrame:
    """
    Create multilabel encoding for the TAIX-Ray dataset based on the annotation rules.
    
    Args:
        df_annot: DataFrame containing annotations with UID and clinical findings
    Returns:
        DataFrame with UID and multilabel encoding for the 5 main disease categories
    """
    # Define the class labels for multilabel classification
    class_labels = ["Cardiomegaly", "Pulmonary congestion", "Pleural effusion", "Pulmonary opacities", "Atelectasis"]
    
    # Initialize the multilabel DataFrame
    df_multilabel = pd.DataFrame(index=df_annot.index)
    df_multilabel['image_id'] = df_annot['UID']
    
    # Rule 1: Cardiomegaly (HeartSize > 0 means enlarged)
    df_multilabel[class_labels[0]] = (df_annot['HeartSize'] > 0).astype(int)
    
    # Rule 2: Pulmonary Congestion (PulmonaryCongestion > 0 means present)
    df_multilabel[class_labels[1]] = (df_annot['PulmonaryCongestion'] > 0).astype(int)
    
    # Rule 3: Pleural Effusion (either left or right > 0 means present)
    df_multilabel[class_labels[2]] = (
        (df_annot['PleuralEffusion_Left'] > 0) | 
        (df_annot['PleuralEffusion_Right'] > 0)
    ).astype(int)
    
    # Rule 4: Pulmonary Opacities (either left or right > 0 means present)
    df_multilabel[class_labels[3]] = (
        (df_annot['PulmonaryOpacities_Left'] > 0) | 
        (df_annot['PulmonaryOpacities_Right'] > 0)
    ).astype(int)
    
    # Rule 5: Atelectasis (either left or right > 0 means present)
    df_multilabel[class_labels[4]] = (
        (df_annot['Atelectasis_Left'] > 0) | 
        (df_annot['Atelectasis_Right'] > 0)
    ).astype(int)
    
    return df_multilabel

def prepare_taixray(path_root: str, output_dir: str):
    # 1) LOAD ANNOTATIONS AND SPLITS
    annot_path = os.path.join(path_root, "metadata/annotation.csv")
    split_path = os.path.join(path_root, "metadata/split.csv")
    
    df_annot = pd.read_csv(annot_path)
    df_split = pd.read_csv(split_path)
    
    # Merge annotations with splits
    df_merged = df_annot.merge(df_split, on='UID', how='inner')
    
    # 2) CREATE MULTILABEL ENCODING
    df_multilabel = create_multilabel_encoding(df_merged)
    
    # 3) TRAIN/TEST SPLIT
    labels = df_multilabel.columns.tolist()
    df_train = df_multilabel[df_merged['Split'] == 'train'].copy()
    df_val = df_multilabel[df_merged['Split'] == 'val'].copy()
    df_test = df_multilabel[df_merged['Split'] == 'test'].copy()

    # 4) SAVE ANNOTATION FILES
    df_train.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    df_val.to_csv(os.path.join(output_dir, "val_labels.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)
    
    logger.info(f"Saved train annotations with {len(df_train)} samples.")
    logger.info(f"Saved val annotations with {len(df_val)} samples.")
    logger.info(f"Saved test annotations with {len(df_test)} samples.")
    
    # 5) SYMLINK PNG IMAGES
    for split, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        src_folder = os.path.join(path_root, "data")
        dst_folder = os.path.join(output_dir, "images", split)
        os.makedirs(dst_folder, exist_ok=True)
        
        symlink_pairs = [
            (os.path.join(src_folder, f"{row['image_id']}.png"),
             os.path.join(dst_folder, f"{row['image_id']}.png"))
            for _, row in df.iterrows()
        ]
        create_symlinks_parallel(symlink_pairs)
        logger.info(f"Symlinked {len(symlink_pairs)} images to {dst_folder}")
    
    logger.info(f"Preprocessing TAIX-Ray complete! The processed dataset is saved in {output_dir}")
    logger.info(f"Classes: {labels}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_taixray(args.path_root, args.output_dir)

if __name__ == "__main__":
    main()
