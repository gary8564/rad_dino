import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import numpy as np
from typing import Optional
from rad_dino.data.preprocessing import dicom2array
import logging
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

class RadImageClassificationDataset(Dataset):
    def __init__(self,
                 path_root: str,
                 split: str,
                 task: str,
                 transform: Optional[transforms.Compose] = None):
        """
        path_root: root directory of the preprocessed dataset
        split: specify the data loader to be in "train" dataset or "test" dataset.
        task: specify the task of the dataset, either "binary", "multi-class", or "multi-label".
        transform:  torchvision transforms. By default, None.
        """
        if split not in ["train", "test"]:
            raise AttributeError(f"`split` attribute must be a str type and specified as either `train` or `test`.")
        if task not in ["binary", "multiclass", "multilabel"]:
            raise AttributeError(f"`task` attribute must be a str type and specified as either `binary`, `multiclass`, or `multilabel`.")
        self.path_root = path_root
        self.task = task
        self.df = pd.read_csv(os.path.join(self.path_root, f"{split}_labels.csv"), index_col="image_id")
        
        if task == "binary":
            self.labels = self.df["label"].tolist()
        else:
            self.labels = self.df.columns.tolist()
        
        self.dicom_root = os.path.join(self.path_root, "images", split)
        self.image_ids = self.df.index.to_list()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.task == "binary":
            targets = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)  # Convert to tensor and reshape to [1, 1]
        else:
            targets = self.df.loc[image_id].to_numpy(dtype=np.float32)
            targets = torch.tensor(targets, dtype=torch.float32)
        
        # Convert DICOM to 2d image array
        dicom_file = os.path.join(self.dicom_root, f"{image_id}.dcm")
        img = dicom2array(dicom_file)

        if self.transform:
            img = self.transform(img)
        return img, targets, image_id
    

if __name__ == "__main__":
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/RSNA-Pneumonia"
    ds_train = RadImageClassificationDataset(path_root, "train", "binary")
    ds_test = RadImageClassificationDataset(path_root, "test", "binary")
    print("RSNA-Pneumonia...")
    print(f"Number of training data: {len(ds_train.image_ids)}")
    print(f"Number of test data: {len(ds_test.image_ids)}")
    
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-CXR"
    class_labels = ["Lung Opacity", "Cardiomegaly", "Pleural thickening", "Aortic enlargement", "Pleural effusion", "Pulmonary fibrosis", "Tuberculosis", "No finding"]
    ds_train = RadImageClassificationDataset(path_root, "train", "multilabel")
    ds_test = RadImageClassificationDataset(path_root, "test", "multilabel")
    print("VinDr-CXR...")
    assert set(ds_train.labels) == set(class_labels), f"Class labels do not match: {set(ds_train.labels)} != {set(class_labels)}"
    print(f"Number of training data: {len(ds_train.image_ids)}")
    print(f"Number of test data: {len(ds_test.image_ids)}")
    
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo"
    class_labels = ['Architectural Distortion', 'Asymmetry', 'Mass', 'No Finding', 'Skin Thickening', 'Suspicious Calcification', 'Suspicious Lymph Node']
    ds_train = RadImageClassificationDataset(path_root, "train", "multilabel")
    ds_test = RadImageClassificationDataset(path_root, "test", "multilabel")
    print("VinDr-Mammo...")
    assert set(ds_train.labels) == set(class_labels), f"Class labels do not match: {set(ds_train.labels)} != {set(class_labels)}"
    print(f"Number of training data: {len(ds_train.image_ids)}")
    print(f"Number of test data: {len(ds_test.image_ids)}")