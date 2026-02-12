import os
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import numpy as np
import logging
import SimpleITK as sitk
from transformers import AutoImageProcessor
from typing import Optional, Callable
from PIL import Image
from torch.utils.data import Dataset
from rad_dino.utils.config_utils import get_model_config
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

# Supported image file extensions for SimpleITK loading
SUPPORTED_EXTENSIONS = ('.dcm', '.dicom', '.png', '.jpg', '.jpeg', '.mha', '.mhd')


def load_image_from_path(img_path: str) -> Image.Image:
    """
    Unified 2D image loader using SimpleITK. Handles all formats supported by SimpleITK
    including DICOM (.dcm), MetaImage (.mha, .mhd), PNG, JPEG, etc.
    
    Args:
        img_path: Path to the image file.
        
    Returns:
        PIL Image in grayscale ("L") mode with pixel values in [0, 255].
        
    Raises:
        ValueError: If the image is a 3D volume (ndim > 2).
    """
    sitk_image = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(sitk_image).astype(np.float64)
    
    # SimpleITK often wraps single-slice DICOMs as (1, H, W).
    # Squeeze singleton dimensions if any, then reject 3D volumes.
    img_array = np.squeeze(img_array)
    if img_array.ndim > 2:
        raise ValueError(
            f"Expected a 2D image but got {img_array.ndim}D volume with shape {img_array.shape} "
            f"from '{img_path}'. This repository only supports 2D chest X-ray images."
        )
    
    # Normalize to [0, 255]
    img_min, img_max = img_array.min(), img_array.max()
    if img_max - img_min > 0:
        img_array = (img_array - img_min) / (img_max - img_min) * 255.0
    else:
        raise ValueError(
            f"Image has constant pixel value {img_min} and cannot be normalized: '{img_path}'. "
            "The file is likely corrupted."
        )
    img_array = img_array.astype(np.uint8)
    return Image.fromarray(img_array)

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
        split: specify the data loader split: "train", "val", or "test".
        task: specify the task of the dataset, either "binary", "multi-class", or "multi-label".
        transform: Any callable that maps an image to a tensor (e.g., torchvision `Compose`). By default, None.
        model_name: name of the model. If None, the image processor will be loaded from the model config.
        multi_view: whether to load multi-view mammography data (4 images per study).
        """
        if split not in ["train", "val", "test"]:
            raise AttributeError("`split` attribute must be a str type and specified as either `train`, `val`, or `test`.")
        if task not in ["binary", "multiclass", "multilabel"]:
            raise AttributeError("`task` attribute must be a str type and specified as either `binary`, `multiclass`, or `multilabel`.")
        if model_name is None and transform is None:
            raise AttributeError("`model_name` attribute must be specified when no transforms are applied.")
        if model_name is not None and transform is not None:
           logging.warning("""`model_name`(AutoImageProcessor) and `transform`(torchvision.transforms) attributes are specified at the same time. 
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
            
        
        self.img_dir = os.path.join(self.path_root, "images", split)
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
        elif self.task == "binary":
            # For binary classification, BCEWithLogitsLoss expects target shape [B, 1] and float dtype
            target_value = self.df.loc[sample_id, "label"]
            targets = torch.tensor(target_value, dtype=torch.float32).unsqueeze(0)
        else:  # multiclass
            target_value = self.df.loc[sample_id, "label"]
            targets = torch.tensor(target_value, dtype=torch.long)  
        
        if self.multi_view:
            # Load 4 images for multi-view mammography
            pil_images = self._load_multi_view_images(sample_id)  # List of 4 images
            
            if self.transform:
                # Apply transforms to each view separately 
                transformed_images = []
                for view_img in pil_images: 
                    transformed_image = self.transform(view_img.convert("RGB"))
                    transformed_images.append(transformed_image)
                
                # Stack transformed views: [4, C, H, W]
                imgs = torch.stack(transformed_images, dim=0)
            else:
                # If no data augmentation is applied, use AutoImageProcessor to process the images               
                # Ensure 3-channel RGB for processors that expect 3 channels
                pil_images = [img.convert("RGB") for img in pil_images]
                imgs = self.image_processor(pil_images, return_tensors="pt")["pixel_values"]
                # Remove batch dimension from AutoImageProcessor output
                imgs = imgs.squeeze(0)  # [1, 4, C, H, W] -> [4, C, H, W]
        else:
            # Load single image
            img_path = None
            for ext in SUPPORTED_EXTENSIONS:
                path = os.path.join(self.img_dir, f"{sample_id}{ext}")
                if os.path.exists(path):
                    img_path = path
                    break
            
            if img_path is None:
                raise FileNotFoundError(
                    f"No image found for id {sample_id} in {self.img_dir} with extensions {SUPPORTED_EXTENSIONS}"
                )
            
            pil_image = load_image_from_path(img_path)

            if self.transform:
                imgs = self.transform(pil_image.convert("RGB"))
            else:
                # If no data augmentation is applied, use AutoImageProcessor to process the images
                # Ensure 3-channel RGB for processors that expect 3 channels    
                imgs = self.image_processor(pil_image.convert("RGB"), return_tensors="pt")["pixel_values"]
                # Remove batch dimension from AutoImageProcessor output
                imgs = imgs.squeeze(0)  # [1, C, H, W] -> [C, H, W]
        return imgs, targets, sample_id
    
    def _load_multi_view_images(self, study_id: str):
        """
        Load 4 mammography views for a study as separate images.
        
        Args:
            study_id: The study identifier
            
        Returns:
            pil_images: list of PIL images
        """
        study_dir = os.path.join(self.img_dir, study_id)
        
        # Define the expected view files (Left CC, Left MLO, Right CC, Right MLO)
        view_names = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
        
        pil_images = []
        for view_name in view_names:
            view_path = None
            for ext in SUPPORTED_EXTENSIONS:
                path = os.path.join(study_dir, f"{view_name}{ext}")
                if os.path.exists(path):
                    view_path = path
                    break
            
            if view_path is None:
                raise FileNotFoundError(
                    f"Multi-view image not found for {view_name} in {study_dir} with extensions {SUPPORTED_EXTENSIONS}"
                )
            
            # Unified loading via SimpleITK for all formats
            view_pil = load_image_from_path(view_path)
            
            pil_images.append(view_pil)
        
        return pil_images
    

if __name__ == "__main__":
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/RSNA-Pneumonia"
    ds_train = RadImageClassificationDataset(path_root, "train", "binary", model_name="rad-dino")
    ds_test = RadImageClassificationDataset(path_root, "test", "binary", model_name="rad-dino")
    print("RSNA-Pneumonia...")
    print(f"Number of training data: {len(ds_train.sample_ids)}")
    print(f"Number of test data: {len(ds_test.sample_ids)}")
    
    # path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-CXR"
    # class_labels = ["Lung Opacity", "Cardiomegaly", "Pleural thickening", "Aortic enlargement", "Pleural effusion", "Pulmonary fibrosis", "Tuberculosis", "No finding"]
    # ds_train = RadImageClassificationDataset(path_root, "train", "multilabel", model_name="rad-dino")
    # ds_test = RadImageClassificationDataset(path_root, "test", "multilabel", model_name="rad-dino")
    # print("VinDr-CXR...")
    # assert set(ds_train.labels) == set(class_labels), f"Class labels do not match: {set(ds_train.labels)} != {set(class_labels)}"
    # print(f"Number of training data: {len(ds_train.sample_ids)}")
    # print(f"Number of test data: {len(ds_test.sample_ids)}")
    
    # path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/findings/multi_view"
    # class_labels = ['Architectural Distortion', 'Asymmetry', 'Mass', 'No Finding', 'Skin Thickening', 'Suspicious Calcification', 'Suspicious Lymph Node']
    # ds_train = RadImageClassificationDataset(path_root, "train", "multilabel", multi_view=True, model_name="rad-dino")
    # ds_test = RadImageClassificationDataset(path_root, "test", "multilabel", multi_view=True, model_name="rad-dino")
    # print("VinDr-Mammo...")
    # assert set(ds_train.labels) == set(class_labels), f"Class labels do not match: {set(ds_train.labels)} != {set(class_labels)}"
    # print(f"Number of training data: {len(ds_train.sample_ids)}")
    # print(f"Number of test data: {len(ds_test.sample_ids)}")
    
    # path_root = "/hpcwork/rwth1833/datasets/preprocessed/VinDr-Mammo/birads/multi_view"
    # ds_train = RadImageClassificationDataset(path_root, "train", "multiclass", multi_view=True, model_name="rad-dino")
    # ds_test = RadImageClassificationDataset(path_root, "test", "multiclass", multi_view=True, model_name="rad-dino")
    # print("VinDr-Mammo BIRADS (Multi-view)...")
    # print(f"Number of training studies: {len(ds_train.sample_ids)}")
    # print(f"Number of test studies: {len(ds_test.sample_ids)}")
    # print(f"Labels: {set(ds_train.labels)}")
    
    # path_root = "/hpcwork/rwth1833/datasets/preprocessed/TAIX-Ray"
    # class_labels = ["Cardiomegaly", "Pulmonary congestion", "Pleural effusion", "Pulmonary opacities", "Atelectasis"]
    # ds_train = RadImageClassificationDataset(path_root, "train", "multilabel", model_name="rad-dino")
    # ds_test = RadImageClassificationDataset(path_root, "test", "multilabel", model_name="rad-dino")
    # print("TAIX-Ray...")
    # assert set(ds_train.labels) == set(class_labels), f"Class labels do not match: {set(ds_train.labels)} != {set(class_labels)}"
    # print(f"Number of training data: {len(ds_train.sample_ids)}")
    # print(f"Number of test data: {len(ds_test.sample_ids)}")
    
    path_root = "/hpcwork/rwth1833/datasets/preprocessed/NODE21"
    ds_train = RadImageClassificationDataset(path_root, "train", "binary", model_name="rad-dino")
    ds_test = RadImageClassificationDataset(path_root, "test", "binary", model_name="rad-dino")
    print("NODE21...")
    print(f"Number of training data: {len(ds_train.sample_ids)}")
    print(f"Number of test data: {len(ds_test.sample_ids)}")