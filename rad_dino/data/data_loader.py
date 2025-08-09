import logging
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from typing import Optional

from rad_dino.data.dataset import RadImageClassificationDataset
from rad_dino.utils.data_utils import collate_fn
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def create_train_loader(data_root_folder: str,
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
    """
    Create train and validation data loaders.
    
    Args:
        data_root_folder: Root folder containing the dataset
        task: Type of classification task
        train_idx: Indices for training samples
        val_idx: Indices for validation samples
        train_transforms: Transforms for training data
        val_transforms: Transforms for validation data
        mini_batch_size: Batch size for training
        batch_size: Batch size for validation
        num_workers: Number of workers for data loading
        multi_view: Whether to use multi-view data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
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

def create_test_loader(data_root_folder: str,
                      task: str, 
                      batch_size: int,
                      test_transforms: Optional[transforms.Compose] = None,
                      multi_view: bool = False,
                      model_name: Optional[str] = None) -> DataLoader:
    """
    Create test data loader.
    
    Args:
        data_root_folder: Root folder containing the dataset
        task: Type of classification task
        batch_size: Batch size for testing
        test_transforms: Transforms for test data
        multi_view: Whether to use multi-view data
        
    Returns:
        test_loader
    """
    # Create test dataset
    test_ds = RadImageClassificationDataset(
        data_root_folder, 
        "test", 
        task, 
        transform=test_transforms, 
        model_name=model_name,
        multi_view=multi_view
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=False
    )
    return test_loader

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
    """
    Load and split data for training and validation.
    
    Args:
        data_root_folder: Root folder containing the dataset
        task: Type of classification task
        batch_size: Batch size for training
        train_transforms: Transforms for training data
        val_transforms: Transforms for validation data
        num_workers: Number of workers for data loading
        gradient_accumulation_steps: Number of gradient accumulation steps
        kfold: Number of folds for k-fold cross-validation
        train_size: Fraction of data to use for training
        seed: Random seed for reproducibility
        multi_view: Whether to use multi-view data
        
    Returns:
        List of (train_loader, val_loader) tuples for k-fold, or single tuple for single split
    """
    # Use a no-op transform (callable) to avoid requiring model_name and loading AutoImageProcessor
    base_ds = RadImageClassificationDataset(
        data_root_folder,
        "train",
        task,
        transform=(lambda x: x),
        multi_view=multi_view,
    )
    
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
            return [create_train_loader(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
                                     mini_batch_size, batch_size, num_workers, multi_view) 
                   for train_idx, val_idx in skf.split(base_ds.sample_ids, Y)]
        elif task == "multilabel":
            # multilabel: use MultilabelStratifiedKFold
            Y = base_ds.df.to_numpy(dtype=np.int64)
            mskf = MultilabelStratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            return [create_train_loader(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
                                     mini_batch_size, batch_size, num_workers, multi_view) 
                   for train_idx, val_idx in mskf.split(X=base_ds.sample_ids, y=Y)]
        else:
            # multiclass: use StratifiedKFold
            Y = base_ds.df.to_numpy(dtype=np.int64)
            skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
            return [create_train_loader(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
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
        return create_train_loader(data_root_folder, task, train_idx, val_idx, train_transforms, val_transforms, 
                                mini_batch_size, batch_size, num_workers, multi_view) 