import torch
import torchvision.transforms.v2 as transforms
from transformers import AutoImageProcessor
from typing import Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, Subset
from rad_dino.data.dataset import RadImageClassificationDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import logging
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

# K-Fold Cross-Validation
@dataclass
class KFold:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    checkpoint_dir: str
    best_checkpoint: str
    best_metric: float
    start_epoch: int
    train_loader: DataLoader
    val_loader: DataLoader

# Cope with variable-length custom dataset
def collate_fn(batch):
    imgs, targets, sample_ids = zip(*batch)
    pixel_values = torch.stack(imgs, dim=0)    # [B, C, H, W] or [B, 4, C, H, W]
    target = torch.stack(targets, dim=0)       # [B, num_classes] or [B, 4, num_classes]
    return pixel_values, target, sample_ids

# Data augmentation: image transformation
def get_transforms(pretrained_model_path):
    """
    Define the transforms for data augmentation. 
    Get the transformers from pretrained model to ensure custom data is 
    transformed/formatted in the same way the data the original model was 
    trained on.
    
    Args:
        pretrained_model_path: Path to the pretrained model
    """
    image_processor = AutoImageProcessor.from_pretrained(pretrained_model_path)
    if "swin" in pretrained_model_path.lower():
        # Special handling for Ark models (not from HuggingFace)
        # Ark models use 768x768 image size and ImageNet normalization
        mean = image_processor.image_mean
        std = image_processor.image_std
        interpolation = image_processor.resample
        crop_size = (768, 768)
        size = 768
        logger.info(f"Ark model are not from HuggingFace, image size {size}x{size} need to be specified manually!")
        normalize = transforms.Normalize(mean=mean, std=std)
    else:
        # Standard HuggingFace model handling
        mean = image_processor.image_mean
        std = image_processor.image_std
        interpolation = image_processor.resample
        crop_size = (image_processor.crop_size["height"], image_processor.crop_size["width"])
        size = image_processor.size["shortest_edge"]
        normalize = transforms.Normalize(mean=mean, std=std)
    
    # First resize all images to a consistent size before any other transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size, interpolation=interpolation),  
        transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomAffine(
            degrees=30,                            # ±30° rotation
            scale=(0.8,1.2),                       # ±20% zoom
            shear=15                               # ±15° shear
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0))],
            p=0.5,
        ),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size, interpolation=interpolation),  
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform

def create_loaders(data_root_folder: str,
                  task: str,
                  train_idx: list,
                  val_idx: list,
                  train_transforms: transforms.Compose,
                  val_transforms: transforms.Compose,
                  mini_batch_size: int,
                  batch_size: int,
                  num_workers: int,
                  multi_view: bool = False,
                  ):
    train_ds = Subset(
        RadImageClassificationDataset(data_root_folder, "train", task, train_transforms, multi_view=multi_view),
        train_idx
    )
    val_ds = Subset(
        RadImageClassificationDataset(data_root_folder, "train", task, val_transforms, multi_view=multi_view),
        val_idx
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=mini_batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader

def load_data(data_root_folder: str,
              task: str,
              batch_size: int, 
              train_transforms: transforms.Compose, 
              val_transforms: transforms.Compose, 
              num_workers: int, 
              gradient_accumulation_steps: int,
              kfold: int | None = None,
              train_size: float = 0.75,
              seed: int = 42,
              multi_view: bool = False):
    base_ds = RadImageClassificationDataset(data_root_folder, "train", task, transform=None, target_size=(518, 518), multi_view=multi_view)
    
    # Calculate mini-batch size
    mini_batch_size = batch_size // gradient_accumulation_steps
    if batch_size % gradient_accumulation_steps != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by gradient_accumulation_steps ({gradient_accumulation_steps})")
    
    # Only do k-fold splitting if kfold is explicitly set to a value greater than 1
    if kfold is not None and kfold > 1:
        logger.info(f"K-fold splitting with {kfold} folds")
        # Handle different dataset structures
        if task == "binary":
            # binary: use StratifiedKFold
            Y = base_ds.df["label"].to_numpy(dtype=np.int64)
            skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            return [create_loaders(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
                                 mini_batch_size, batch_size, num_workers, multi_view) 
                   for train_idx, val_idx in skf.split(base_ds.sample_ids, Y)]
        elif task == "multilabel":
            # multilabel: use MultilabelStratifiedKFold
            Y = base_ds.df.to_numpy(dtype=np.int64)
            mskf = MultilabelStratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            return [create_loaders(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
                                 mini_batch_size, batch_size, num_workers, multi_view) 
                   for train_idx, val_idx in mskf.split(X=base_ds.sample_ids, y=Y)]
        else:
            # multiclass: use StratifiedKFold
            Y = base_ds.df.to_numpy(dtype=np.int64)
            skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            return [create_loaders(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
                                 mini_batch_size, batch_size, num_workers, multi_view) 
                   for train_idx, val_idx in skf.split(base_ds.sample_ids, Y)]
    else:
        logger.info("Single split case")
        n_total_samples = len(base_ds)
        n_train = int(train_size * n_total_samples)
        if n_train == 0 or n_train == n_total_samples:
            raise ValueError("Train or validation set is empty due to small dataset size")
        
        torch.manual_seed(seed)
        perm = torch.randperm(n_total_samples).tolist()
        train_idx, val_idx = perm[:n_train], perm[n_train:]
        return create_loaders(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
                            mini_batch_size, batch_size, num_workers, multi_view)
        
def get_class_weights(task: str, dataset: Dataset):
    """
    Calculate pos_weight or class weights for weighted loss functions.
    
    For binary classification:
        Returns pos_weight for BCEWithLogitsLoss using the formula:
        pos_weight = num_negative_samples / num_positive_samples
        
    For multilabel classification:
        Returns pos_weight tensor for each class using the formula:
        pos_weight[i] = num_negative_samples[i] / num_positive_samples[i]
        
    For multiclass classification:
        Returns class weights using the formula:
        weight_for_class_i = total_samples / (num_samples_in_class_i)
    
    Args:
        task: Type of classification task
        dataset: Dataset to calculate weights from
        
    Returns:
        torch.Tensor or None: Weights for the loss function
    """
    if task == "binary":
        # For binary classification, calculate pos_weight
        class_counts = dataset.df["label"].value_counts().to_dict()
        # Get counts for negative (0) and positive (1) classes
        num_negative = class_counts.get(0, 0)
        num_positive = class_counts.get(1, 0)
        if num_positive == 0:
            logger.warning("No positive samples found in dataset for binary classification")
            return None
        # Calculate pos_weight = num_negative / num_positive
        pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32)
        return pos_weight
    elif task == "multilabel":
        # For multilabel classification, calculate pos_weight for each class
        # Assuming dataset.df contains binary columns for each class
        pos_weights = []
        for col in dataset.df.columns:
            if col != "image_id":  # Skip non-label columns
                class_counts = dataset.df[col].value_counts().to_dict()
                num_negative = class_counts.get(0, 0)
                num_positive = class_counts.get(1, 0)
                
                if num_positive == 0:
                    logger.warning(f"No positive samples found for class {col}")
                    pos_weight = 1.0  # Default weight
                else:
                    pos_weight = num_negative / num_positive
                    
                pos_weights.append(pos_weight)        
        if pos_weights:
            return torch.tensor(pos_weights, dtype=torch.float32)
        else:
            logger.warning("No pos_weights calculated for multilabel classification")
            return None
    elif task == "multiclass":
        # For multiclass classification, calculate class weights
        # Using formula: weight_for_class_i = total_samples / (num_samples_in_class_i)
        class_weights = []
        total_samples = len(dataset.df)
        # Get all columns except image_id
        class_columns = [col for col in dataset.df.columns if col != "image_id"]
        # Calculate weight for each class column
        for class_idx, col in enumerate(class_columns):
            # Count positive samples for this class (assuming binary encoding: 0/1)
            class_counts = dataset.df[col].value_counts().to_dict()
            num_positive = class_counts.get(1, 0)
            if num_positive == 0:
                logger.warning(f"No positive samples found for class {col}, using default weight 1.0")
                weight = 1.0
            else:
                weight = total_samples / num_positive
            class_weights.append(weight)
        return torch.tensor(class_weights, dtype=torch.float32)
    else:
        raise ValueError(f"Task {task} is not supported for weight calculation")