import os
import argparse
import pandas as pd
import logging
from typing import Union
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)

CLASS_LABELS = [
    "No finding",
    "Osteophytes",
    "Disc space narrowing",
    "Foraminal stenosis",
    "Other lesions",
    "Surgical implant",
    "Spondylolysthesis",
    "Vertebral collapse",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess VinDr-SpineXR for multilabel image classification."
    )
    parser.add_argument("--path-root", type=str, required=True,
                        help="Root directory of the VinDr-SpineXR dataset "
                             "(containing annotations/, train_images/, test_images/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write preprocessed label CSVs and image symlinks")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Specify an integer k for top-k classes (by prevalence) "
                             "or an explicit list of class names. "
                             "If not provided, all 8 classes (including 'No finding') are used.")
    return parser.parse_args()


def annotations_to_multilabel(df_annot: pd.DataFrame,
                              class_labels: list[str]) -> pd.DataFrame:
    """
    Convert bounding-box annotation rows into a per-image multi-hot label DataFrame.

    Args:
        df_annot: Raw annotation DataFrame with columns [image_id, lesion_type, ...].
        class_labels: Ordered list of all label names including "No finding".

    Returns:
        DataFrame indexed by image_id with one column per label class (int 0/1).
    """
    all_image_ids = df_annot["image_id"].unique()
    disease_classes = [c for c in class_labels if c != "No finding"]

    # Keep only disease rows (exclude "No finding")
    df_disease = df_annot[df_annot["lesion_type"].isin(disease_classes)].copy()

    if len(df_disease) == 0:
        # Edge case: no disease annotations at all (all "No finding")
        df_labels = pd.DataFrame(0, index=all_image_ids, columns=disease_classes)
    else:
        # One-hot encode the lesion_type column
        dummies = pd.get_dummies(df_disease["lesion_type"]).astype(int)
        dummies["image_id"] = df_disease["image_id"].values

        # Aggregate per image: max across all bounding-box rows to get multi-hot encoding
        df_labels = dummies.groupby("image_id").max()

        # Ensure all disease_classes columns exist (some class labels may have zero instances in annotations)
        for cls in disease_classes:
            if cls not in df_labels.columns:
                df_labels[cls] = 0

        # Reorder disease columns
        df_labels = df_labels[disease_classes]

        # Re-index to include "No finding" images
        df_labels = df_labels.reindex(all_image_ids, fill_value=0)

    # Derive "No finding" class label when all disease class labels are 0
    df_labels.insert(0, "No finding", (df_labels[disease_classes].sum(axis=1) == 0).astype(int))

    # Reorder columns to match the canonical class_labels order
    df_labels = df_labels[class_labels]
    df_labels.index.name = "image_id"

    return df_labels


def filter_classes(df: pd.DataFrame,
                   label_cols: list[str],
                   classes: Union[int, list[str]]) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter the multi-hot label DataFrame to a subset of classes.

    Args:
        df: DataFrame with image_id as index and multi-hot label columns.
        label_cols: All available label column names.
        classes: Either an integer k (top-k by prevalence) or an explicit list of class names.

    Returns:
        Filtered DataFrame and the selected class names.
    """
    if isinstance(classes, int):
        class_counts = df[label_cols].sum().sort_values(ascending=False)
        selected = class_counts.head(classes).index.tolist()
    else:
        selected = list(classes)
        unknown = set(selected) - set(label_cols)
        if unknown:
            raise ValueError(f"Unknown class names: {unknown}. Available: {label_cols}")
    return df[selected], selected


def prepare_vindrspinexr(path_root: str, output_dir: str,
                         class_labels: Union[int, str, list[str], None]):
    """
    Preprocess the VinDr-SpineXR dataset for multilabel classification.

    The raw dataset provides per-bounding-box annotations with lesion_type labels.
    This script converts them into per-image multi-hot label CSVs suitable for
    RadImageClassificationDataset.

    Steps:
    1. Load annotations/train.csv and annotations/test.csv.
    2. Convert bounding-box rows into per-image multi-hot vectors across 8
       classes (7 diseases + "No finding"). "No finding" is derived as 1 when
       all disease columns are 0.
    3. Optionally filter to a subset of classes.
    4. Verify images exist on disk and drop missing entries.
    5. Save {split}_labels.csv with image_id as index.
    6. Create symlinks from train_images/ and test_images/ into images/{split}/.

    Args:
        path_root: Root directory containing annotations/, train_images/, test_images/.
        output_dir: Directory to write preprocessed outputs.
        class_labels: None (all 8 classes including "No finding"), an integer k
                      (top-k), a single class name string, or a list of class names.
    """
    train_annot_path = os.path.join(path_root, "annotations", "train.csv")
    test_annot_path = os.path.join(path_root, "annotations", "test.csv")

    for path in [train_annot_path, test_annot_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected annotation file not found: {path}")

    # 1) LOAD ANNOTATIONS
    df_train_annot = pd.read_csv(train_annot_path)
    df_test_annot = pd.read_csv(test_annot_path)

    logger.info(f"Loaded train annotations: {len(df_train_annot)} rows, "
                f"{df_train_annot['image_id'].nunique()} unique images")
    logger.info(f"Loaded test annotations: {len(df_test_annot)} rows, "
                f"{df_test_annot['image_id'].nunique()} unique images")
    logger.info(f"Lesion types in train: {sorted(df_train_annot['lesion_type'].unique())}")

    # 2) CONVERT BOUNDING-BOX ANNOTATIONS TO PER-IMAGE MULTI-HOT LABELS
    class_labels = list(CLASS_LABELS)
    df_train = annotations_to_multilabel(df_train_annot, class_labels)
    df_test = annotations_to_multilabel(df_test_annot, class_labels)

    logger.info(f"Built multi-hot labels: train={len(df_train)}, test={len(df_test)}")

    # 3) OPTIONALLY FILTER CLASSES
    label_cols = class_labels
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

    # 4) VERIFY IMAGES EXIST ON DISK AND DROP MISSING ENTRIES
    #    Image directories are named train_images/ and test_images/ (not train/ and test/).
    split_to_img_dir = {
        "train": os.path.join(path_root, "train_images"),
        "test": os.path.join(path_root, "test_images"),
    }
    for split, df in [("train", df_train), ("test", df_test)]:
        src_folder = split_to_img_dir[split]
        existing_files = set(os.listdir(src_folder))
        before = len(df)
        mask = df.index.to_series().apply(lambda x: f"{x}.dicom" in existing_files)
        missing = before - mask.sum()
        if missing > 0:
            logger.warning(
                f"{split}: {missing}/{before} image_ids in annotations have no .dicom file "
                f"on disk. Dropping them."
            )
        if split == "train":
            df_train = df_train[mask.values]
        else:
            df_test = df_test[mask.values]

    logger.info(f"Train set: {len(df_train)} images (after filtering missing files)")
    logger.info(f"Test set: {len(df_test)} images (after filtering missing files)")

    # Log label distribution
    logger.info(f"Train label distribution:\n{df_train.sum().to_string()}")
    logger.info(f"Test label distribution:\n{df_test.sum().to_string()}")

    # Log normal vs abnormal counts
    if "No finding" in df_train.columns:
        train_normal = df_train["No finding"].sum()
        test_normal = df_test["No finding"].sum()
        logger.info(f"Train: {len(df_train) - train_normal} abnormal, {train_normal} normal (No finding)")
        logger.info(f"Test: {len(df_test) - test_normal} abnormal, {test_normal} normal (No finding)")

    # 5) SAVE LABEL CSVs
    df_train.to_csv(os.path.join(output_dir, "train_labels.csv"))
    df_test.to_csv(os.path.join(output_dir, "test_labels.csv"))
    logger.info(f"Saved train_labels.csv ({len(df_train)} rows) and "
                f"test_labels.csv ({len(df_test)} rows)")

    # 6) SYMLINK DICOM IMAGES
    for split, df in [("train", df_train), ("test", df_test)]:
        src_folder = split_to_img_dir[split]
        dst_folder = os.path.join(output_dir, "images", split)
        os.makedirs(dst_folder, exist_ok=True)

        symlink_pairs = [
            (os.path.join(src_folder, f"{image_id}.dicom"),
             os.path.join(dst_folder, f"{image_id}.dcm"))
            for image_id in df.index
        ]
        create_symlinks_parallel(symlink_pairs)
        logger.info(f"Symlinked {len(symlink_pairs)} images to {dst_folder}")

    logger.info(f"Preprocessing VinDr-SpineXR complete! Output saved to {output_dir}")


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

    prepare_vindrspinexr(args.path_root, args.output_dir, class_labels)


if __name__ == "__main__":
    main()
