import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
import pydicom
import numpy as np
from typing import Optional, Union
from sklearn.model_selection import train_test_split
from data.preprocessing import dicom2array
import logging
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

class RSNAPneumonia_Dataset(Dataset):
    def __init__(self,
                path_root: str,
                split: str,
                transform: Optional[transforms.Compose] = None):
        """
        path_root: root directory of RSNAPneumonia dataset
        split: specify the data loader to be in "train" dataset or "test" dataset.
        transform:  torchvision transforms
        """
        if split not in ["train", "test"]:
            raise AttributeError(f"`split` attribute must be a str type and specified as either `train` or `test`.")
        self.path_root = path_root

        if not os.path.exists(os.path.join(self.path_root, "train/train_labels.csv")) or \
            not os.path.exists(os.path.join(self.path_root, "test/test_labels.csv")):
            os.makedirs(os.path.join(self.path_root, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.path_root, "test"), exist_ok=True)
            train_df, test_df = self._train_test_split()
            self.df = train_df if split == "train" else test_df
        else:
            self.df = pd.read_csv(os.path.join(self.path_root, f"{split}/train_labels.csv"))
        
        self.labels = self.df["Target"].tolist()
        
        self.dicom_root = os.path.join(self.path_root, "raw/stage_2_train_images")
        self.image_ids = self.df["patientId"].to_list()
        
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        targets = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)  # Convert to tensor and reshape to [1, 1]
        
        # Convert DICOM to 2d image array
        dicom_file = os.path.join(self.dicom_root, f"{image_id}.dcm")
        img = dicom2array(dicom_file)

        if self.transform:
            img = self.transform(img)
        return img, targets, image_id
    
    def _train_test_split(self, train_size: float = 0.8):
        """Split the dataset into training and testing sets.
        
        Args:
            train_size: float, the proportion of training data in the dataset
        """
        if train_size <= 0 or train_size >= 1 or not isinstance(train_size, float):
            raise AttributeError(f"`train_size` attribute must be a float type between 0 and 1.")
        labels_path = os.path.join(self.path_root, "raw/stage_2_train_labels.csv")
        df = pd.read_csv(labels_path)
        idx = df.groupby('patientId')['Target'].idxmax()
        df_selected = df.loc[idx].reset_index(drop=True)
        
        # train/test split on labels
        train_df, test_df = train_test_split(df_selected, test_size=0.2, random_state=42,
                                            stratify=df_selected['Target'])
        logger.info(f'Training size: {train_df.shape[0]}, Testing size: {test_df.shape[0]}')
        train_df.to_csv(os.path.join(self.path_root, "train/train_labels.csv"), index=False)
        test_df.to_csv(os.path.join(self.path_root, "test/test_labels.csv"), index=False)
        
        return train_df, test_df

if __name__ == "__main__":
    path_root = "/hpcwork/rwth1833/datasets/RSNA-Pneumonia"
    ds_train = RSNAPneumonia_Dataset(path_root, "train")
    ds_test = RSNAPneumonia_Dataset(path_root, "test")
    print(f"Number of training data: {len(ds_train.image_ids)}")
    print(f"Number of test data: {len(ds_test.image_ids)}")