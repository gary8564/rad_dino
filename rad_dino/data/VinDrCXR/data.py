import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import pydicom
import numpy as np
from typing import Optional, Union
from data.preprocessing import dicom2array, filter_subset_annot_labels

class VinDrCXR_Dataset(Dataset):
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
                path_root: str,
                split: str,
                class_labels: Union[int, str, list[str], None] = None, 
                transform: Optional[transforms.Compose] = None):
        """
        path_root: root directory of VinDrCXR dataset
        split: specify the data loader to be in "train" dataset or "test" dataset.
        class_labels: If None, all labels in original dataset are considered. 
                      If specified as an interger k, the class labels considered for the image classification task are chosen by selecting the top-k most frequent "local" findings among all training images to avoid “noise” from extreme class imbalance.
                      If specified as a string or list of string, the class labels are specified as the defined class_labels (a string will be a single label, a list of string will be a sequence of specified labels).
                      By default, None.
        transform:  torchvision transforms
        """
        if split not in ["train", "test"]:
            raise AttributeError(f"`split` attribute must be a str type and specified as either `train` or `test`.")
        self.path_root = path_root
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
            self.annot_file = train_labels_path if class_labels is None else os.path.join(self.path_root, "annotations/filtered_image_labels_train.csv")
            self.dicom_root = os.path.join(self.path_root, "train")
        else:
            self.annot_file = test_labels_path if class_labels is None else os.path.join(self.path_root, "annotations/filtered_image_labels_test.csv")            
            self.dicom_root = os.path.join(self.path_root, "test")
        # annotations.csv:
        # self.df = pd.read_csv(self.annot_file)
        # df_grouped_by_imageid = self.df.groupby("image_id")
        # self.image_ids = list(df_grouped_by_imageid.groups.keys())
        # self.df_grouped_by_imageid = df_grouped_by_imageid
        # image_labels.csv:
        df = pd.read_csv(self.annot_file, index_col="image_id")
        df = df[self.labels]
        df = df.groupby(level=0).max() # group by image_id (the index) and take max across the 3 annotators
        self.df = df
        self.image_ids = self.df.index.to_list()
        
        # Preprocessing (3): Data augmentation  
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Multi-hot encoding
        # annotations.csv:
        # rows = self.df_grouped_by_imageid.get_group(image_id)
        # targets = np.zeros(len(self.labels), dtype=np.float32)
        # for cls in rows['class_name'].unique():
        #     targets[self.cls2idx[cls]] = 1.0
        # image_labels.csv:
        targets = self.df.loc[image_id].to_numpy(dtype=np.float32)
        
        # Convert DICOM to 2d image array
        dicom_file = os.path.join(self.dicom_root, f"{image_id}.dicom")
        img = dicom2array(dicom_file)

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(targets, dtype=torch.float32), image_id

if __name__ == "__main__":
    path_root = "/hpcwork/rwth1833/datasets/VinDr-CXR/download/physionet.org/files/vindr-cxr/1.0.0"
    class_labels = ["Lung Opacity", "Cardiomegaly", "Pleural thickening", "Aortic enlargement", "Pleural effusion", "Pulmonary fibrosis", "Tuberculosis", "No finding"]
    ds_train = VinDrCXR_Dataset(path_root, "train", class_labels=class_labels)
    ds_test = VinDrCXR_Dataset(path_root, "test", class_labels=class_labels)
    print(f"Number of training data: {len(ds_train.image_ids)}")
    print(f"Number of test data: {len(ds_test.image_ids)}")