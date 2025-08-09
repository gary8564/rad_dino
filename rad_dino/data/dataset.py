import os
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import numpy as np
import logging
from transformers import AutoImageProcessor
from typing import Optional, Callable
from PIL import Image
from torch.utils.data import Dataset
from rad_dino.utils.preprocessing_utils import dicom2array
from rad_dino.utils.config_utils import get_model_config
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def load_image_processor(model_name: str) -> AutoImageProcessor:
    """
    Load the image processor for the given model.
    """
    config = get_model_config(model_name)
    return AutoImageProcessor.from_pretrained(config["hf_repo"])

class RadImageClassificationDataset(Dataset):
    def __init__(self,
                 path_root: str,
                 split: str,
                 task: str,
                 transform: Optional[Callable] = None,
                 model_name: Optional[str] = None,
                 multi_view: bool = False):
        """
        path_root: root directory of the preprocessed dataset
        split: specify the data loader to be in "train" dataset or "test" dataset.
        task: specify the task of the dataset, either "binary", "multi-class", or "multi-label".
        transform: Any callable that maps an image to a tensor (e.g., torchvision `Compose`). By default, None.
        model_name: name of the model. If None, the image processor will be loaded from the model config.
        multi_view: whether to load multi-view mammography data (4 images per study).
        """
        if split not in ["train", "test"]:
            raise AttributeError(f"`split` attribute must be a str type and specified as either `train` or `test`.")
        if task not in ["binary", "multiclass", "multilabel"]:
            raise AttributeError(f"`task` attribute must be a str type and specified as either `binary`, `multiclass`, or `multilabel`.")
        if model_name is None and transform is None:
            raise AttributeError(f"`model_name` attribute must be specified when no transforms are applied.")
        if model_name is not None and transform is not None:
           logging.warning(f"""`model_name`(AutoImageProcessor) and `transform`(torchvision.transforms) attributes are specified at the same time. 
                           `transform` has the priority over the `model_name`. AutoImageProcessor will be ignored.""")
        
        self.model_name = model_name
        self.image_processor = None
        # Only load image processor if transform is None
        if transform is None:
            self.image_processor = load_image_processor(model_name)
        self.path_root = path_root
        self.task = task
        self.multi_view = multi_view
        label_file = f"{split}_labels.csv"
        if not os.path.exists(os.path.join(self.path_root, label_file)):
            raise FileNotFoundError(f"No label file found in {self.path_root}.")
        # Load labels based on task and multi-view setting
        if multi_view:
            # For multi-view, we expect study_id as index
            self.df = pd.read_csv(os.path.join(self.path_root, label_file), index_col="study_id")
        else:
            # For single-view, we expect image_id as index
            self.df = pd.read_csv(os.path.join(self.path_root, label_file), index_col="image_id")
        
        if task == "multilabel":
            self.labels = self.df.columns.tolist()
        else:
            self.labels = self.df["label"].tolist()
            
        
        self.dicom_root = os.path.join(self.path_root, "images", split)
        self.sample_ids = self.df.index.to_list()  # study_id for multi-view and image_id for single-view
        self.transform = transform

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Extract targets - same logic for both single-view and multi-view
        # For multi-view: one set of labels per study (applies to all 4 views)
        # For single-view: one set of labels per image
        if self.task == "multilabel":
            targets = self.df.loc[sample_id].to_numpy(dtype=np.float32)
            targets = torch.tensor(targets, dtype=torch.float32)
        else:
            target_value = self.df.loc[sample_id, "label"]
            targets = torch.tensor(target_value, dtype=torch.long)  
        
        if self.multi_view:
            # Load 4 images for multi-view mammography
            images = self._load_multi_view_images(sample_id)  # List of 4 images
            
            if self.transform:
                # Apply transforms to each view separately 
                transformed_images = []
                for view_img in images: 
                    transformed_image = self.transform(view_img)
                    transformed_images.append(transformed_image)
                
                # Stack transformed views: [4, C, H, W]
                imgs = torch.stack(transformed_images, dim=0)
            else:
                # If no data augmentation is applied
                # Use AutoImageProcessor to process the images
                pil_images = [Image.fromarray(img) for img in images]
                imgs = self.image_processor(pil_images, return_tensors="pt")["pixel_values"]
                # Remove batch dimension from AutoImageProcessor output
                imgs = imgs.squeeze(0)  # [1, 4, C, H, W] -> [4, C, H, W]
        else:
            # Load single image
            dicom_file = os.path.join(self.dicom_root, f"{sample_id}.dcm")
            img = dicom2array(dicom_file)

            if self.transform:
                imgs = self.transform(img)
            else:
                # If no data augmentation is applied
                # Use AutoImageProcessor to process the images
                pil_image = Image.fromarray(img)
                imgs = self.image_processor(pil_image, return_tensors="pt")["pixel_values"]
                # Remove batch dimension from AutoImageProcessor output
                imgs = imgs.squeeze(0)  # [1, C, H, W] -> [C, H, W]
        return imgs, targets, sample_id
    
    def _load_multi_view_images(self, study_id: str):
        """
        Load 4 mammography views for a study as separate images.
        
        Args:
            study_id: The study identifier
            
        Returns:
            list: List of 4 individual images as numpy arrays
        """
        study_dir = os.path.join(self.dicom_root, study_id)
        
        # Define the expected view files (Left CC, Left MLO, Right CC, Right MLO)
        view_files = ['L_CC.dcm', 'L_MLO.dcm', 'R_CC.dcm', 'R_MLO.dcm']
        
        # Initialize list to store individual images
        images = []
        
        for view_file in view_files:
            view_path = os.path.join(study_dir, view_file)
            
            if not os.path.exists(view_path):
                raise FileNotFoundError(f"Multi-view image not found: {view_path}")
            
            # Load the view
            view_img = dicom2array(view_path)
            images.append(view_img)
        
        return images
    

if __name__ == "__main__":
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/RSNA-Pneumonia"
    ds_train = RadImageClassificationDataset(path_root, "train", "binary", model_name="rad-dino")
    ds_test = RadImageClassificationDataset(path_root, "test", "binary", model_name="rad-dino")
    print("RSNA-Pneumonia...")
    print(f"Number of training data: {len(ds_train.sample_ids)}")
    print(f"Number of test data: {len(ds_test.sample_ids)}")
    
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-CXR"
    class_labels = ["Lung Opacity", "Cardiomegaly", "Pleural thickening", "Aortic enlargement", "Pleural effusion", "Pulmonary fibrosis", "Tuberculosis", "No finding"]
    ds_train = RadImageClassificationDataset(path_root, "train", "multilabel", model_name="rad-dino")
    ds_test = RadImageClassificationDataset(path_root, "test", "multilabel", model_name="rad-dino")
    print("VinDr-CXR...")
    assert set(ds_train.labels) == set(class_labels), f"Class labels do not match: {set(ds_train.labels)} != {set(class_labels)}"
    print(f"Number of training data: {len(ds_train.sample_ids)}")
    print(f"Number of test data: {len(ds_test.sample_ids)}")
    
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/findings/multi_view"
    class_labels = ['Architectural Distortion', 'Asymmetry', 'Mass', 'No Finding', 'Skin Thickening', 'Suspicious Calcification', 'Suspicious Lymph Node']
    ds_train = RadImageClassificationDataset(path_root, "train", "multilabel", multi_view=True, model_name="rad-dino")
    ds_test = RadImageClassificationDataset(path_root, "test", "multilabel", multi_view=True, model_name="rad-dino")
    print("VinDr-Mammo...")
    assert set(ds_train.labels) == set(class_labels), f"Class labels do not match: {set(ds_train.labels)} != {set(class_labels)}"
    print(f"Number of training data: {len(ds_train.sample_ids)}")
    print(f"Number of test data: {len(ds_test.sample_ids)}")
    
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/birads/multi_view"
    ds_train = RadImageClassificationDataset(path_root, "train", "multiclass", multi_view=True, model_name="rad-dino")
    ds_test = RadImageClassificationDataset(path_root, "test", "multiclass", multi_view=True, model_name="rad-dino")
    print("VinDr-Mammo BIRADS (Multi-view)...")
    print(f"Number of training studies: {len(ds_train.sample_ids)}")
    print(f"Number of test studies: {len(ds_test.sample_ids)}")
    print(f"Labels: {set(ds_train.labels)}")