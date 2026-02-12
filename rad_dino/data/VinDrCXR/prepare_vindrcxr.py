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
    parser.add_argument("--classes", nargs="+", default=None, help="Specify an integer k for top-k classes or a list of subset of class names.")
    return parser.parse_args()

def filter_subset_annot_labels(label_path: str, annot_path: str, labels: Union[int, list[str]]):
    """
    A subset of class labels is selected, given the long-tailed distribution in the original dataset.
    
    Args:
        label_path: Path to label csv file
        annot_path: Path to annotation csv file
        labels: the class labels to be filtered
    Returns:
        A list of the class labels considered for image classification task
    
    """
    df_label = pd.read_csv(label_path, index_col="image_id")
    df_annot  = pd.read_csv(annot_path, index_col="image_id")
    annot_dir = os.path.dirname(label_path)
    if isinstance(labels, int):
        # find top-k classes in the training dataset
        class_labels = (
            df_annot['class_name']
            .value_counts()
            .head(labels)
            .index
            .tolist()
        )
    else:
        class_labels = list(labels)
        
    # filter classes
    df_label_filtered = df_label[class_labels]
    df_annot_filtered = df_annot[df_annot['class_name'].isin(class_labels)].copy()
    df_annot_filtered.reset_index(drop=True, inplace=True)
    
    # output the filtered csv 
    label_out = os.path.join(annot_dir, "filtered_image_labels_train.csv")
    annot_out = os.path.join(annot_dir, "filtered_annotations_train.csv")
    df_label_filtered.to_csv(label_out, index=True)
    df_annot_filtered.to_csv(annot_out, index=True)
    logger.debug(f"Filtered csv are saved!")
    return class_labels, df_label_filtered

def prepare_vindrcxr(path_root: str, output_dir: str, class_labels: Union[int, str, list[str], None]):
    # 1) SELECT CLASS SET 
    train_labels_path = os.path.join(path_root, "annotations/image_labels_train.csv")
    train_annot_path = os.path.join(path_root, "annotations/annotations_train.csv")
    test_labels_path = os.path.join(path_root, "annotations/image_labels_test.csv")
    test_annot_path = os.path.join(path_root, "annotations/annotations_test.csv")
    if class_labels is None:
        df_train = pd.read_csv(train_labels_path, index_col="image_id")
        df_test = pd.read_csv(test_labels_path, index_col="image_id")
        labels = df_train.columns.tolist()
        assert len(labels) == 15, f"The number of class labels must be 15 when `class_labels` is None."
    elif isinstance(class_labels, int):
        labels, df_train = filter_subset_annot_labels(train_labels_path, train_annot_path, class_labels)
        _, df_test = filter_subset_annot_labels(test_labels_path, test_annot_path, labels)
    elif isinstance(class_labels, str):
        labels, df_train = filter_subset_annot_labels(train_labels_path, train_annot_path, [class_labels])
        _, df_test = filter_subset_annot_labels(test_labels_path, test_annot_path, labels)
    else:
        labels, df_train = filter_subset_annot_labels(train_labels_path, train_annot_path, class_labels)
        _, df_test = filter_subset_annot_labels(test_labels_path, test_annot_path, labels)
    
    # 2) AGGREGATE LABELS WITH MAJORITY VOTING AND BUILD MULTI-HOT ENCODING DATAFRAME FOR CLASSIFICATION 
    df_train_agg = df_train.groupby(level=0).agg(lambda x: 1 if (len(x.mode()) == 2) else pd.Series.mode(x)[0])
    df_test_agg = df_test.groupby(level=0).agg(lambda x: 1 if (len(x.mode()) == 2) else pd.Series.mode(x)[0])
    df_train_agg.to_csv(os.path.join(output_dir, "train_labels.csv"))
    df_test_agg.to_csv(os.path.join(output_dir, "test_labels.csv"))
    
    # 3) SYMLINK .dicom FILES
    for split, df in [("train", df_train_agg), ("test", df_test_agg)]:
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
    logger.info(f"Preprocessing VinDr-CXR complete! The processed dataset is saved in {output_dir}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_vindrcxr(args.path_root, args.output_dir, args.classes)

if __name__ == "__main__":
    main()