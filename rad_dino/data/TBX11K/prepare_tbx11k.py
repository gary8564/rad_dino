import os
import argparse
import pandas as pd
import logging
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)

# Folder prefix → label mapping
BINARY_LABEL_MAP = {"tb": 1, "health": 0, "sick": 0}
MULTICLASS_LABEL_MAP = {"tb": 2, "health": 0, "sick": 1}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True,
                        help="Path to the TBX11K root directory (containing imgs/, lists/, annotations/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the preprocessed output directory")
    parser.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass"],
                        help="Classification task: 'binary' (TB vs Non-TB) or 'multiclass' (Healthy vs Sick vs TB)")
    return parser.parse_args()


def load_list_file(list_path: str) -> pd.DataFrame:
    """
    Parse a TBX11K list file into a DataFrame with columns: folder, filename, image_id.

    Each line has the format  ``<folder>/<filename>.png``  where <folder> is one of
    ``tb``, ``health``, or ``sick``.
    """
    with open(list_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    records = []
    for rel_path in lines:
        folder = rel_path.split("/")[0]
        filename = os.path.basename(rel_path)
        image_id = os.path.splitext(filename)[0]
        records.append({"rel_path": rel_path, "folder": folder, "filename": filename, "image_id": image_id})
    return pd.DataFrame(records)


def prepare_tbx11k(path_root: str, output_dir: str, task: str):
    """
    Preprocess the TBX11K dataset for image-level classification.

    Uses the official TBX11K_train / TBX11K_val splits (only core TBX11K images,
    excluding external datasets in ``imgs/extra/``).  Test labels are not publicly
    released, so ``TBX11K_val`` is used as the test set.

    Args:
        path_root: Root of the TBX11K dataset (the ``TBX11K/`` folder).
        output_dir: Base output directory. 
        task: "binary" (TB=1, Non-TB=0) or "multiclass" (Healthy=0, Sick=1, TB=2).
    """
    label_map = BINARY_LABEL_MAP if task == "binary" else MULTICLASS_LABEL_MAP
    output_dir = os.path.join(output_dir, task)
    os.makedirs(output_dir, exist_ok=True)
    imgs_dir = os.path.join(path_root, "imgs")

    split_mapping = {
        "train": os.path.join(path_root, "lists", "TBX11K_train.txt"),
        "test":  os.path.join(path_root, "lists", "TBX11K_val.txt"),
    }

    for split, list_path in split_mapping.items():
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Expected list file not found: {list_path}")

        # 1) PARSE LIST FILE
        df = load_list_file(list_path)
        logger.info(f"Loaded {len(df)} entries from {list_path}")

        # 2) ASSIGN LABELS
        df["label"] = df["folder"].map(label_map)
        unmapped = df["label"].isna()
        if unmapped.any():
            skipped_folders = df.loc[unmapped, "folder"].unique().tolist()
            logger.warning(f"Skipping {unmapped.sum()} entries with unmapped folders: {skipped_folders}")
            df = df[~unmapped].reset_index(drop=True)
        df["label"] = df["label"].astype(int)

        logger.info(f"[{split}] Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")

        # 3) VERIFY IMAGES EXIST ON DISK
        df["src_path"] = df["rel_path"].apply(lambda p: os.path.join(imgs_dir, p))
        missing = ~df["src_path"].apply(os.path.exists)
        if missing.any():
            logger.warning(f"Dropping {missing.sum()} entries whose images are missing on disk.")
            df = df[~missing].reset_index(drop=True)

        # 4) SAVE LABEL CSV
        df_labels = df[["image_id", "label"]].copy()
        label_csv = os.path.join(output_dir, f"{split}_labels.csv")
        df_labels.to_csv(label_csv, index=False)
        logger.info(f"Saved {label_csv} with {len(df_labels)} entries.")

        # 5) SYMLINK IMAGES
        dst_folder = os.path.join(output_dir, "images", split)
        os.makedirs(dst_folder, exist_ok=True)
        symlink_pairs = [
            (row["src_path"], os.path.join(dst_folder, f"{row['image_id']}.png"))
            for _, row in df.iterrows()
        ]
        create_symlinks_parallel(symlink_pairs)
        logger.info(f"Symlinked {len(symlink_pairs)} images to {dst_folder}")

    logger.info(f"Preprocessing TBX11K ({task}) complete! Output saved to {output_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_tbx11k(args.path_root, args.output_dir, args.task)


if __name__ == "__main__":
    main()
