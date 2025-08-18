import os
import argparse
import pandas as pd
import logging
from typing import Union
from rad_dino.loggings.setup import init_logging
from sklearn.model_selection import train_test_split
init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", required=True,
                   help="Path to RSNA-Pneumonia root folder")
    parser.add_argument("--output-dir", required=True,
                   help="Path to the preprocessed output directory of the dataset")
    parser.add_argument("--test-size", type=float, default=0.2,
                   help="Fraction to reserve for test dataset")
    return parser.parse_args()

def prepare_rsna_pneumonia(path_root: str, output_dir: str, test_size: float):
    """
    Preprocess the RSNA-Pneumonia dataset.
    """
    if test_size <= 0 or test_size >= 1 or not isinstance(test_size, float):
        raise AttributeError(f"`test_size` attribute must be a float type between 0 and 1.")
        
    # 1) LOAD AND MAJORITY-VOTE PER PATIENT
    labels_path = os.path.join(path_root, "stage_2_train_labels.csv")
    df_raw = pd.read_csv(labels_path)
    # group by patientId and take the most frequent Target value
    df = (
        df_raw
        .groupby("patientId")["Target"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    df = df.rename(columns={"patientId": "image_id", "Target": "label"})
    
    # 2) STRATIFIED SPLIT ON LABELS
    train_df, test_df = train_test_split(
       df, test_size=test_size,
       stratify=df["label"], random_state=42
    )
    logger.info(f'Training size: {train_df.shape[0]}, Testing size: {test_df.shape[0]}')
    
    # Ensure no overlap between train and test sets
    train_ids = set(train_df["image_id"])
    test_ids = set(test_df["image_id"])
    assert len(train_ids.intersection(test_ids)) == 0, "Found overlapping images between train and test sets!"
    
    train_df.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)
    
    # 3) SYMLINK IMAGES
    src_folder = os.path.join(path_root, "stage_2_train_images")
    for split, df_split in [("train", train_df), ("test", test_df)]:
        dst_folder = os.path.join(output_dir, "images", split)
        os.makedirs(dst_folder, exist_ok=True)
        for image_id in df_split["image_id"]:
            src = os.path.join(src_folder, f"{image_id}.dcm")
            dst = os.path.join(dst_folder, f"{image_id}.dcm")
            if not os.path.exists(dst):
                os.symlink(src, dst)
            else:
                os.remove(dst)
                os.symlink(src, dst)
    logger.info(f"Preprocessing RSNA-Pneumonia complete! The processed dataset is saved in {output_dir}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_rsna_pneumonia(args.path_root, args.output_dir, args.test_size)

if __name__ == "__main__":
    main()