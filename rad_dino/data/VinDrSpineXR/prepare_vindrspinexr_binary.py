import os
import argparse
import pandas as pd
import logging
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import create_symlinks_parallel
init_logging()
logger = logging.getLogger(__name__)

DISEASE_CLASSES = [
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
        description="Preprocess VinDr-SpineXR for binary (normal/abnormal) classification."
    )
    parser.add_argument("--path-root", type=str, required=True,
                        help="Root directory of the VinDr-SpineXR dataset "
                             "(containing annotations/, train_images/, test_images/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write preprocessed label CSVs and image symlinks")
    return parser.parse_args()


def annotations_to_binary(df_annot: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bounding-box annotation rows into a per-image binary label DataFrame.

    label = 0 (normal)  when the image has only "No finding" annotations.
    label = 1 (abnormal) when the image has at least one disease bounding box.

    Args:
        df_annot: Raw annotation DataFrame with columns [image_id, lesion_type, ...].

    Returns:
        DataFrame indexed by image_id with a single 'label' column (int 0/1).
    """
    all_image_ids = df_annot["image_id"].unique()

    # An image is abnormal if it has any disease annotation row
    df_disease = df_annot[df_annot["lesion_type"].isin(DISEASE_CLASSES)]
    abnormal_ids = set(df_disease["image_id"].unique())

    df_labels = pd.DataFrame(
        {"label": [1 if iid in abnormal_ids else 0 for iid in all_image_ids]},
        index=all_image_ids,
    )
    df_labels.index.name = "image_id"
    return df_labels


def prepare_vindrspinexr_binary(path_root: str, output_dir: str):
    """
    Preprocess the VinDr-SpineXR dataset for binary (normal vs abnormal) classification.

    Normal  (label=0): images annotated only as "No finding".
    Abnormal (label=1): images with at least one disease bounding box.

    Steps:
    1. Load annotations/train.csv and annotations/test.csv.
    2. Derive a per-image binary label from the bounding-box annotations.
    3. Verify images exist on disk and drop missing entries.
    4. Save {split}_labels.csv with image_id as index and a single 'label' column.
    5. Create symlinks from train_images/ and test_images/ into images/{split}/.
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

    # 2) CONVERT TO BINARY LABELS
    df_train = annotations_to_binary(df_train_annot)
    df_test = annotations_to_binary(df_test_annot)

    logger.info(f"Built binary labels: train={len(df_train)}, test={len(df_test)}")

    # 3) VERIFY IMAGES EXIST ON DISK AND DROP MISSING ENTRIES
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

    # Log class distribution
    train_abnormal = df_train["label"].sum()
    test_abnormal = df_test["label"].sum()
    logger.info(f"Train: {train_abnormal} abnormal, {len(df_train) - train_abnormal} normal")
    logger.info(f"Test: {test_abnormal} abnormal, {len(df_test) - test_abnormal} normal")

    # 4) SAVE LABEL CSVs
    df_train.to_csv(os.path.join(output_dir, "train_labels.csv"))
    df_test.to_csv(os.path.join(output_dir, "test_labels.csv"))
    logger.info(f"Saved train_labels.csv ({len(df_train)} rows) and "
                f"test_labels.csv ({len(df_test)} rows)")

    # 5) SYMLINK DICOM IMAGES
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

    logger.info(f"Preprocessing VinDr-SpineXR (binary) complete! Output saved to {output_dir}")
    logger.info("Classes: {0: 'normal (No finding)', 1: 'abnormal (any disease)'}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_vindrspinexr_binary(args.path_root, args.output_dir)


if __name__ == "__main__":
    main()
