import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import pydicom
import numpy as np
from typing import Optional, Union
from rad_dino.data.VinDrCXR.preprocessing import dicom2array, filter_subset_annot_labels

class VinDrCXR_Dataset(Dataset):
    PATH_ROOT = "/hpcwork/rwth1833/datasets/VinDr-CXR2/download/physionet.org/files/vindr-cxr/1.0.0"
    CLASSES = [
      "Aortic enlargement",
      "Atelectasis",
      "Calcification",
      "Cardiomegaly",
      "Consolidation",
      "ILD",
      "Infiltration",
      "Lung Opacity",
      "Nodule/Mass",
      "Other lesion",
      "Pleural effusion",
      "Pleural thickening",
      "Pneumothorax",
      "Pulmonary fibrosis",
      "Lung tumor",
      "Pneumonia",
      "Tuberculosis",
      "Other diseases",
      "COPD",
      "No finding"
    ]
    def __init__(self,
                split: str,
                path_root: Optional[str] = None,
                class_labels: Union[int, str, list[str]] = None, 
                transform: Optional[transforms.Compose] = None):
        """
        split: 
        path_root: root directory of VinDrCXR dataset
        class_labels: If None, all labels in original dataset are considered. 
                      If specified as an interger k, the class labels considered for the image classification task are chosen by selecting the top-k most frequent "local" findings among all training images to avoid “noise” from extreme class imbalance.
                      If specified as a string or list of string, the class labels are specified as the defined class_labels (a string will be a single label, a list of string will be a sequence of specified labels).
                      By default, None.
        transform:  torchvision transforms
        """
        if split not in ["train", "test"]:
            raise AttributeError(f"`split` attribute must be a str type and specified as either `train` or `test`.")
        self.path_root = self.PATH_ROOT if path_root is None else path_root
        # Preprocessing (1): select the subset of class labels
        train_labels_path = os.path.join(self.path_root, "annotations/image_labels_train.csv")
        test_labels_path = os.path.join(self.path_root, "annotations/image_labels_test.csv")
        if class_labels is None:
            self.labels = self.CLASSES
            assert len(self.labels) == 15, f"The number of class labels must be 15 when `class_labels` is None."
        elif isinstance(class_labels, int):
            self.labels = filter_subset_annot_labels(train_labels_path, test_labels_path, class_labels)
        elif isinstance(class_labels, str):
            self.labels = [class_labels]
            _ = filter_subset_annot_labels(train_labels_path, test_labels_path, self.labels)
        else:
            self.labels = class_labels
            _ = filter_subset_annot_labels(train_labels_path, test_labels_path, self.labels)
        self.cls2idx = {c: i for i, c in enumerate(self.labels)}
        
        # Preprocessing (2): load annotation and dicom files
        if split == "train": 
            self.annot_file = train_labels_path if class_labels is None else os.path.join(self.path_root, "annotations/filtered_annotations_train.csv")
        else:
            self.annot_file = test_labels_path if class_labels is None else os.path.join(self.path_root, "annotations/filtered_annotations_test.csv")            
        df = pd.read_csv(self.annot_file)
        self.dicom_root = os.path.join(self.path_root, "train")
        df_grouped_by_imageid = df.groupby("image_id")
        self.image_ids = list(df_grouped_by_imageid.groups.keys())
        self.df_grouped_by_imageid = df_grouped_by_imageid
        
        # Preprocessing (3): Data augmentation  
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        rows = self.df_grouped_by_imageid.get_group(image_id)
        # Multi-hot encoding
        targets = np.zeros(len(self.labels), dtype=np.float32)
        for cls in rows['class_name'].unique():
            targets[self.cls2idx[cls]] = 1.0
        
        # Convert DICOM to 2d image array
        dicom_file = os.path.join(self.dicom_root, f"{image_id}.dicom")
        img = dicom2array(dicom_file)

        if self.transform:
            img = self.transform(img)
        return img, torch.from_numpy(targets)


if __name__ == "__main__":
    from sklearn.preprocessing import MultiLabelBinarizer
    # from sklearn.model_selection import StratifiedKFold
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    from torch.utils.data import Subset, DataLoader

    
    # Instantiate train dataset
    class_labels = ["Lung Opacity", "Cardiomegaly", "Pleural thickening", "Aortic enlargement", "Pleural effusion", "Pulmonary fibrosis", "Tuberculosis", "No finding"]
    ds = VinDrCXR_Dataset("train", class_labels=class_labels)
    print(f"Number of training data: {len(ds.image_ids)}")

    # Build a multi‑label matrix Y (N_images × n_classes)
    mlb = MultiLabelBinarizer(classes=ds.labels)
    all_labels_per_img = []
    for img_id in ds.image_ids:
        cls_list = ds.df_grouped_by_imageid.get_group(img_id)['class_name'].unique().tolist()
        all_labels_per_img.append(cls_list)
    Y = mlb.fit_transform(all_labels_per_img)  

    # Create the KFold splitter
    mskf = MultilabelStratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Generate folds
    fold_subsets = []
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X=ds.image_ids, y=Y)):
        print(f"Fold {fold}: {len(train_idx)} train / {len(val_idx)} val")
        train_ds = Subset(ds, train_idx)
        val_ds   = Subset(ds, val_idx)
        fold_subsets.append((train_ds, val_ds))

    # Iterate folds
    # for fold, (train_ds, val_ds) in enumerate(fold_subsets):
    #     print(f"\n=== Fold {fold} Label Counts ===")
    #     def count_labels(dataset):
    #         agg = np.zeros(len(ds.labels), dtype=int)
    #         for _, targets in dataset:
    #             agg += targets.numpy().astype(int)
    #         return agg

    #     tr_counts = count_labels(train_ds)
    #     vl_counts = count_labels(val_ds)
    #     for i, cls in enumerate(ds.labels):
    #         print(f" {cls:20s} — train: {tr_counts[i]:4d}, val: {vl_counts[i]:4d}")

    #     train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4)
    #     val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4)

        # … your train/validation loop here …


