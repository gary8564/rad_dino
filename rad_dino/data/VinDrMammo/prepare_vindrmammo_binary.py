import os
import argparse
import pandas as pd
import logging
from typing import List

from rad_dino.loggings.setup import init_logging
from rad_dino.utils.preprocessing_utils import copy_files_parallel

init_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-root", type=str, required=True, help="Path to the root directory of the dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to the preprocessed output directory of the dataset")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying source images.")
    return parser.parse_args()


def birads_to_binary(birads_value: str) -> int:
    """Map BI-RADS category to binary label.

    Convention:
    - Negative (0): BI-RADS 1, BI-RADS 2, BI-RADS 3
    - Positive (1): BI-RADS 4, BI-RADS 5
    """
    positive_set = {"BI-RADS 4", "BI-RADS 5"}
    if pd.isna(birads_value):
        return 0
    return 1 if str(birads_value) in positive_set else 0


def select_highest_birad(values: pd.Series) -> str:
    """Select the highest BI-RADS among the provided values.

    Severity order: 1 < 2 < 3 < 4 < 5
    """
    severity_order = {
        'BI-RADS 1': 1,
        'BI-RADS 2': 2,
        'BI-RADS 3': 3,
        'BI-RADS 4': 4,
        'BI-RADS 5': 5,
    }
    max_birad = max(values, key=lambda x: severity_order.get(x, 0))
    logger.info(f"Selected max BI-RADS value from views {values.tolist()}: {max_birad}")
    return max_birad


EXPECTED_VIEWS = {("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")}


def filter_complete_studies(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Keep only studies that have all 4 expected mammography views.

    Expected views: L_CC, L_MLO, R_CC, R_MLO.
    Studies with missing or duplicate views are dropped.
    """
    complete_mask = df_agg.apply(
        lambda row: set(zip(row["laterality"], row["view_position"])) == EXPECTED_VIEWS,
        axis=1,
    )
    n_before = len(df_agg)
    df_filtered = df_agg[complete_mask]
    n_dropped = n_before - len(df_filtered)
    if n_dropped > 0:
        dropped_ids = df_agg.index[~complete_mask].tolist()
        logger.warning(
            f"Dropped {n_dropped}/{n_before} studies with incomplete views "
            f"(expected {EXPECTED_VIEWS}): {dropped_ids}"
        )
    else:
        logger.info(f"All {n_before} studies have complete 4-view exams.")
    return df_filtered


def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    values = df[column].unique()
    values = [v for v in values if v != column and pd.notna(v)]
    return sorted(values)


def create_multi_view_structure(
    df_agg: pd.DataFrame,
    output_dir: str,
    split: str,
    src_images_folder: str,
    use_symlink: bool = False,
) -> None:
    dst_folder = os.path.join(output_dir, "images", split)
    os.makedirs(dst_folder, exist_ok=True)

    file_pairs = []
    for study_id, row in df_agg.iterrows():
        image_ids = row['image_id']
        lateralities = row['laterality']
        view_positions = row['view_position']

        study_dir = os.path.join(dst_folder, study_id)
        os.makedirs(study_dir, exist_ok=True)

        for i, image_id in enumerate(image_ids):
            laterality = lateralities[i]
            view_pos = view_positions[i]
            src_file = os.path.join(src_images_folder, study_id, f"{image_id}.dicom")
            dst = os.path.join(study_dir, f"{laterality}_{view_pos}.dcm")
            file_pairs.append((src_file, dst))

    copy_files_parallel(file_pairs, raise_on_missing=True, use_symlink=use_symlink)
    action = "Symlinked" if use_symlink else "Copied"
    logger.info(f"{action} {len(file_pairs)} multi-view images to {dst_folder}")


def main():
    # 0) PARSE ARGUMENTS
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) READ THE ORIGINAL CSV
    df = pd.read_csv(os.path.join(args.path_root, "breast-level_annotations.csv"))

    # 2) INFO
    unique_values = get_unique_values(df, 'breast_birads')
    logger.info(f"Unique BI-RADS values: {unique_values}")

    # 3) SPLIT BASED ON THE 'split' COLUMN
    df_train_raw = df[df["split"] == "training"].copy()
    df_test_raw = df[df["split"] == "test"].copy()

    # 4) AGGREGATION
    df_train_agg = df_train_raw.groupby('study_id').agg({
        'image_id': list,
        'breast_birads': lambda x: select_highest_birad(x),
        'laterality': list,
        'view_position': list,
    })
    df_test_agg = df_test_raw.groupby('study_id').agg({
        'image_id': list,
        'breast_birads': lambda x: select_highest_birad(x),
        'laterality': list,
        'view_position': list,
    })
    logger.info(f"Multi-view: {len(df_train_agg)} studies in train, {len(df_test_agg)} studies in test")
    df_train_agg = filter_complete_studies(df_train_agg)
    df_test_agg = filter_complete_studies(df_test_agg)
    logger.info(f"After filtering: {len(df_train_agg)} train, {len(df_test_agg)} test")

    # 5) CONVERT TO BINARY LABELS
    for df_agg in [df_train_agg, df_test_agg]:
        df_agg['label'] = df_agg['breast_birads'].apply(birads_to_binary)
        
    # 6) SAVE LABEL FILES
    df_train_labels = df_train_agg[['label']].copy()
    df_test_labels = df_test_agg[['label']].copy()
    df_train_labels.index.name = 'study_id'
    df_test_labels.index.name = 'study_id'

    df_train_labels.to_csv(os.path.join(args.output_dir, "train_labels.csv"))
    df_test_labels.to_csv(os.path.join(args.output_dir, "test_labels.csv"))

    # 7) COPY IMAGE FILES
    src_images_folder = os.path.join(args.path_root, "images")
    create_multi_view_structure(df_train_agg, args.output_dir, "train", src_images_folder, use_symlink=args.symlink)
    create_multi_view_structure(df_test_agg, args.output_dir, "test", src_images_folder, use_symlink=args.symlink)

    logger.info("Preprocessing VinDr-Mammo Binary (from BI-RADS) complete!")
    logger.info("Classes: {0: 'negative', 1: 'positive'}; positive := BI-RADS 4 or 5")
    logger.info(f"The processed dataset is saved in {args.output_dir}")


if __name__ == "__main__":
    main()
