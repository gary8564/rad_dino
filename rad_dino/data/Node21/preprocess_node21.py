import os
import argparse
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True,
                        help="Path to the root directory of the NODE21 dataset (containing images/ and metadata.csv)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the preprocessed output directory of the dataset")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction to reserve for test dataset")
    return parser.parse_args()

def prepare_node21(path_root: str, output_dir: str, test_size: float):
    """
    Preprocess the NODE21 dataset for binary classification (nodule vs no-nodule).
    
    The NODE21 metadata contains per-nodule bounding box annotations. An image with 
    label=1 contains at least one nodule; label=0 means no nodule. We aggregate per 
    image and perform a stratified train/test split.
    
    Args:
        path_root: Root directory containing images/ and metadata.csv
        output_dir: Directory to write train_labels.csv, test_labels.csv, and symlinked images
        test_size: Fraction of data to reserve for testing
    """
    if test_size <= 0 or test_size >= 1 or not isinstance(test_size, float):
        raise AttributeError("`test_size` attribute must be a float type between 0 and 1.")

    # 1) LOAD METADATA
    metadata_path = os.path.join(path_root, "metadata.csv")
    df_raw = pd.read_csv(metadata_path)
    logger.info(f"Loaded metadata with {len(df_raw)} rows and {df_raw['img_name'].nunique()} unique images.")

    # 2) AGGREGATE PER IMAGE: binary label (1 if any nodule, 0 otherwise)
    # Each image may have multiple rows (one per nodule). We take the max label per image.
    df = (
        df_raw
        .groupby("img_name")["label"]
        .max()
        .reset_index()
    )
    # Rename columns to match the convention used in other datasets
    # Strip the file extension from img_name to get image_id
    df["image_id"] = df["img_name"].apply(lambda x: os.path.splitext(x)[0])
    df = df[["image_id", "label"]]
    
    logger.info(f"Per-image label distribution:\n{df['label'].value_counts().to_string()}")

    # 3) VERIFY IMAGES EXIST
    img_dir = os.path.join(path_root, "images")
    existing_images = set(os.listdir(img_dir))
    # Keep only images that actually exist on disk
    df_raw_names = df["image_id"].apply(lambda x: x + ".mha")
    missing = df_raw_names[~df_raw_names.isin(existing_images)]
    if len(missing) > 0:
        logger.warning(f"Found {len(missing)} images in metadata but not on disk. They will be dropped.")
        df = df[df_raw_names.isin(existing_images)].reset_index(drop=True)

    # 4) STRATIFIED TRAIN/TEST SPLIT
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df["label"], random_state=42
    )
    logger.info(f"Training size: {len(train_df)}, Testing size: {len(test_df)}")
    logger.info(f"Train label distribution:\n{train_df['label'].value_counts().to_string()}")
    logger.info(f"Test label distribution:\n{test_df['label'].value_counts().to_string()}")

    # Ensure no overlap between train and test sets
    train_ids = set(train_df["image_id"])
    test_ids = set(test_df["image_id"])
    assert len(train_ids.intersection(test_ids)) == 0, "Found overlapping images between train and test sets!"

    # 5) SAVE LABEL FILES
    train_df.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)

    # 6) SYMLINK IMAGES
    src_folder = os.path.join(path_root, "images")
    for split, df_split in [("train", train_df), ("test", test_df)]:
        dst_folder = os.path.join(output_dir, "images", split)
        os.makedirs(dst_folder, exist_ok=True)
        symlink_pairs = [
            (os.path.join(src_folder, f"{image_id}.mha"),
             os.path.join(dst_folder, f"{image_id}.mha"))
            for image_id in df_split["image_id"]
        ]
        create_symlinks_parallel(symlink_pairs)
        logger.info(f"Symlinked {len(symlink_pairs)} images to {dst_folder}")
    
    logger.info(f"Preprocessing NODE21 complete! The processed dataset is saved in {output_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_node21(args.path_root, args.output_dir, args.test_size)


if __name__ == "__main__":
    main()
