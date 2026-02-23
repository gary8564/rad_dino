import os
import argparse
import pandas as pd
import logging
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True,
                        help="Path to the root directory of the COVID-CXR dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the preprocessed output directory of the dataset")
    return parser.parse_args()


def load_split_txt(txt_path: str) -> pd.DataFrame:
    """
    Load a COVID-CXR split file (.txt) into a DataFrame.
    
    The .txt files are space-delimited with 4 columns (no header):
        patient_id  image_filename  label  source
    
    Args:
        txt_path: Path to the .txt split file.
    Returns:
        DataFrame with columns: patient_id, filename, label, source
    """
    df = pd.read_csv(
        txt_path,
        sep=" ",
        header=None,
        names=["patient_id", "filename", "label", "source"],
        dtype={"patient_id": str},
    )
    return df


def prepare_covid_cxr(path_root: str, output_dir: str):
    """
    Preprocess the COVID-CXR dataset for binary classification (COVID-positive vs negative).
    
    The source dataset provides train/val/test splits with .txt label files.
    This script:
    1. Parses the .txt label files into CSVs with the project's expected format.
    2. Maps string labels ("positive"/"negative") to binary integers (1/0).
    3. Creates symlinks from the source image directories into the expected
       images/{split}/ directory structure.
    
    Args:
        path_root: Root directory of the COVID-CXR dataset containing train/, val/, test/
                   folders and train.txt, val.txt, test.txt label files.
        output_dir: Directory to write preprocessed label CSVs and image symlinks.
    """
    label_map = {"positive": 1, "negative": 0}
    
    for split in ["train", "val", "test"]:
        # 1) LOAD THE TXT LABEL FILE
        txt_path = os.path.join(path_root, f"{split}.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Expected label file not found: {txt_path}")
        df = load_split_txt(txt_path)
        logger.info(f"Loaded {len(df)} entries from {txt_path}")
        
        # 2) DERIVE image_id (filename without extension) AND BINARY LABEL
        df["image_id"] = df["filename"].apply(lambda f: os.path.splitext(f)[0])
        df["label"] = df["label"].map(label_map)
        
        # Verify no unmapped labels
        if df["label"].isna().any():
            unmapped = df.loc[df["label"].isna(), "label"].unique()
            raise ValueError(f"Encountered unmapped labels in {split}: {unmapped}")
        df["label"] = df["label"].astype(int)
        
        # 3) RESOLVE DUPLICATE image_ids via MAJORITY VOTE
        duplicates = df.duplicated(subset="image_id", keep=False)
        if duplicates.any():
            num_duplicates = duplicates.sum()
            duplicate_ids = df.loc[duplicates, "image_id"].nunique()
            # Check if duplicates have conflicting labels
            conflicting_labels = df.loc[duplicates].groupby("image_id")
            num_conflicting = sum(1 for _, g in conflicting_labels if g["label"].nunique() > 1)
            logger.warning(
                f"Found {num_duplicates} rows for {duplicate_ids} duplicate image_ids in {split} split "
                f"({num_conflicting} with conflicting labels). Resolving by majority vote."
            )
            # Aggregate: majority vote for label, keep first patient_id and filename
            df = (
                df.groupby("image_id", as_index=False)
                .agg({
                    "patient_id": "first",
                    "filename": "first",
                    "label": lambda x: x.mode().iloc[0],
                })
            )
        
        # 4) SAVE THE LABEL CSV
        df_labels = df[["image_id", "patient_id", "label"]].copy()
        df_labels.to_csv(os.path.join(output_dir, f"{split}_labels.csv"), index=False)
        logger.info(f"Saved {split}_labels.csv with {len(df_labels)} entries.")
        logger.info(f"  Label distribution: {df_labels['label'].value_counts().to_dict()}")
        
        # 5) SYMLINK IMAGES
        src_folder = os.path.join(path_root, split)
        dst_folder = os.path.join(output_dir, "images", split)
        os.makedirs(dst_folder, exist_ok=True)
        
        # Normalise extension to lowercase so the dataset loader can find it
        # (handles the rare .JPG → .jpg case)
        symlink_pairs = []
        for _, row in df.iterrows():
            src = os.path.join(src_folder, row["filename"])
            stem = row["image_id"]
            ext = os.path.splitext(row["filename"])[1].lower()
            dst = os.path.join(dst_folder, f"{stem}{ext}")
            symlink_pairs.append((src, dst))
        
        create_symlinks_parallel(symlink_pairs)
        logger.info(f"Symlinked {len(symlink_pairs)} images to {dst_folder}")
    
    logger.info(f"Preprocessing COVID-CXR complete! The processed dataset is saved in {output_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_covid_cxr(args.path_root, args.output_dir)


if __name__ == "__main__":
    main()
