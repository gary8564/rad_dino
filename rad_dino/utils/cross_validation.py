import torch
from torch.utils.data import DataLoader
from typing import Optional
from dataclasses import dataclass

@dataclass
class KFold:
    """
    Data class for storing k-fold cross-validation state.
    
    Attributes:
        model: The neural network model
        optimizer: The optimizer for training
        lr_scheduler: Learning rate scheduler (optional)
        checkpoint_dir: Directory to save checkpoints
        best_checkpoint: Path to the best checkpoint
        best_metric: Best validation metric achieved
        start_epoch: Starting epoch number
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    checkpoint_dir: str
    best_checkpoint: str
    best_metric: float
    start_epoch: int
    train_loader: DataLoader
    val_loader: DataLoader