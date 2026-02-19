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
    parser.add_argument("--path-root", type=str, required=True,
                        help="Path to the root directory of the VinDr-PCXR dataset")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the preprocessed output directory of the dataset")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Specify an integer k for top-k classes or a list of class names to keep. "
                             "If not provided, all 15 classes are used.")
    return parser.parse_args()


def filter_classes(df: pd.DataFrame,
                   label_cols: list[str],
                   classes: Union[int, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter the multi-hot label DataFrame to a subset of classes.

    Args:
        df: DataFrame with image_id as index and label columns.
        label_cols: All available label column names.
        classes: Either an integer k (top-k by prevalence) or an explicit list of class names.

    Returns:
        Filtered DataFrame and the selected class names.
    """
    if isinstance(classes, int):
        # Top-k classes by total positive count
        class_counts = df[label_cols].sum().sort_values(ascending=False)
        selected = class_counts.head(classes).index.tolist()
    else:
        selected = list(classes)
        unknown = set(selected) - set(label_cols)
        if unknown:
            raise ValueError(f"Unknown class names: {unknown}. Available: {label_cols}")
    return df[selected], selected


def prepare_vindrpcxr(path_root: str, output_dir: str,
                      class_labels: Union[int, str, list[str], None]):
    """
    Preprocess the VinDr-PCXR dataset for multilabel classification.

    The raw dataset provides predefined train/test splits with per-image multi-hot
    label CSVs and DICOM images. 

    This script:
    1. Loads the image_labels_{split}.csv files.
    2. Optionally filters to a subset of classes.
    3. Saves {split}_labels.csv with image_id as index and label columns only.
    4. Creates symlinks from source .dicom files into images/{split}/.

    Args:
        path_root: Root directory of the VinDr-PCXR dataset containing train/, test/,
                   image_labels_train.csv, and image_labels_test.csv.
        output_dir: Directory to write preprocessed label CSVs and image symlinks.
        class_labels: None (all 15 classes), an integer k (top-k), a single class
                      name string, or a list of class names.
    """
    train_labels_path = os.path.join(path_root, "image_labels_train.csv")
    test_labels_path = os.path.join(path_root, "image_labels_test.csv")

    for path in [train_labels_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected label file not found: {path}")

    # 1) LOAD LABEL FILES
    df_train = pd.read_csv(train_labels_path)
    df_test = pd.read_csv(test_labels_path)

    # Identify the label columns (everything except image_id and rad_ID)
    non_label_cols = {"image_id", "rad_ID"}
    label_cols = [c for c in df_train.columns if c not in non_label_cols]
    logger.info(f"Found {len(label_cols)} label classes: {label_cols}")

    # Set image_id as index
    df_train = df_train.set_index("image_id")
    df_test = df_test.set_index("image_id")

    # Drop rad_ID column (not needed for classification)
    df_train = df_train.drop(columns=["rad_ID"])
    df_test = df_test.drop(columns=["rad_ID"])

    # 2) OPTIONALLY FILTER CLASSES
    if class_labels is not None:
        if isinstance(class_labels, str):
            class_labels = [class_labels]
        elif isinstance(class_labels, list) and len(class_labels) == 1:
            try:
                class_labels = int(class_labels[0])
            except ValueError:
                pass
        df_train, selected = filter_classes(df_train, label_cols, class_labels)
        df_test, _ = filter_classes(df_test, label_cols, selected)
        label_cols = selected
        logger.info(f"Filtered to {len(label_cols)} classes: {label_cols}")

    # Convert to integer (0/1) in case of float values
    df_train = df_train.astype(int)
    df_test = df_test.astype(int)

    # 3) VERIFY IMAGES EXIST ON DISK AND DROP MISSING ENTRIES
    #    The raw download may be incomplete, so we keep only image_ids
    #    that have a corresponding .dicom file on disk.
    for split, df in [("train", df_train), ("test", df_test)]:
        src_folder = os.path.join(path_root, split)
        existing_files = set(os.listdir(src_folder))
        before = len(df)
        mask = df.index.to_series().apply(lambda x: f"{x}.dicom" in existing_files)
        missing = before - mask.sum()
        if missing > 0:
            logger.warning(
                f"{split}: {missing}/{before} image_ids in CSV have no .dicom file on disk. "
                f"Dropping them."
            )
        if split == "train":
            df_train = df_train[mask.values]
        else:
            df_test = df_test[mask.values]

    logger.info(f"Train set: {len(df_train)} images (after filtering missing files)")
    logger.info(f"Test set: {len(df_test)} images (after filtering missing files)")
    logger.info(f"Train label distribution:\n{df_train.sum().to_string()}")
    logger.info(f"Test label distribution:\n{df_test.sum().to_string()}")

    # 4) SAVE LABEL CSVs
    df_train.to_csv(os.path.join(output_dir, "train_labels.csv"))
    df_test.to_csv(os.path.join(output_dir, "test_labels.csv"))
    logger.info(f"Saved train_labels.csv ({len(df_train)} rows) and test_labels.csv ({len(df_test)} rows)")

    # 5) SYMLINK DICOM IMAGES
    for split, df in [("train", df_train), ("test", df_test)]:
        src_folder = os.path.join(path_root, split)
        dst_folder = os.path.join(output_dir, "images", split)
        os.makedirs(dst_folder, exist_ok=True)

        symlink_pairs = [
            (os.path.join(src_folder, f"{image_id}.dicom"),
             os.path.join(dst_folder, f"{image_id}.dcm"))
            for image_id in df.index
        ]
        create_symlinks_parallel(symlink_pairs)
        logger.info(f"Symlinked {len(symlink_pairs)} images to {dst_folder}")

    logger.info(f"Preprocessing VinDr-PCXR complete! Output saved to {output_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse --classes: could be a single int or a list of strings
    class_labels = args.classes
    if class_labels is not None and len(class_labels) == 1:
        try:
            class_labels = int(class_labels[0])
        except ValueError:
            class_labels = class_labels[0]

    prepare_vindrpcxr(args.path_root, args.output_dir, class_labels)


if __name__ == "__main__":
    main()
