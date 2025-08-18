import torch
import logging
from rad_dino.loggings.setup import init_logging
import os
from typing import Optional, Union
from accelerate import Accelerator
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score
init_logging()
logger = logging.getLogger(__name__)
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

# Get loss function
def get_criterion(task: str, weights: Union[torch.Tensor, None] = None, device: Union[str, torch.device] = "cpu"):
    """
    Get appropriate loss function based on task type with optional weighting.
    
    Args:
        task: Task type
        weights: Weights for multiclass CrossEntropyLoss or pos_weights for binary/multilabel BCEWithLogitsLoss
        device: Device to move weights to (e.g., "cuda:0", "cpu")
        
    Returns:
        torch.nn loss function
    """
    if weights is not None:
        weights = weights.to(device)
    criterion_map = {
        "multiclass": torch.nn.CrossEntropyLoss(weight=weights),
        "multilabel": torch.nn.BCEWithLogitsLoss(pos_weight=weights),
        "binary": torch.nn.BCEWithLogitsLoss(pos_weight=weights),
        "regression": torch.nn.MSELoss()
    }
    if task not in criterion_map:
        raise NotImplementedError(f"Task {task} is not supported")
    return criterion_map[task]

# Get evaluation metrics
def get_eval_metrics(task: str, num_classes: int, device: str):
    """Create appropriate metrics based on task type."""
    metrics = {}
    
    if task == "multiclass":
        metrics.update({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "top5_acc": Accuracy(task="multiclass", num_classes=num_classes, top_k=5),
            "auroc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
            "ap": AveragePrecision(task="multiclass", num_classes=num_classes, average="macro"),
            "f1_score": F1Score(task="multiclass", num_classes=num_classes)
        })
    elif task == "multilabel":
        metrics.update({
            "acc": Accuracy(task="multilabel", num_labels=num_classes),
            "auroc": AUROC(task="multilabel", num_labels=num_classes, average="macro"),
            "ap": AveragePrecision(task="multilabel", num_labels=num_classes, average="macro"),
            "f1_score": F1Score(task="multilabel", num_labels=num_classes)
        })
    elif task == "binary":
        metrics.update({
            "acc": Accuracy(task="binary"),
            "top5_acc": Accuracy(task="binary", top_k=5),
            "auroc": AUROC(task="binary"),
            "ap": AveragePrecision(task="binary"),
            "f1_score": F1Score(task="binary")
        })
    
    return {k: v.to(device) for k, v in metrics.items() if v is not None}

# Early Stopping
class EarlyStopping:
    """
    Early stops the training if monitored metric doesn't improve after a given patience.
    """
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "max",
        ckpt_path: str = "best.pt",
        accelerator: Optional[Accelerator] = None
    ):
        """
        Args:
            patience: how many epochs to wait after last time metric improved.
            min_delta: minimum change in the monitored metric to qualify as improvement.
            mode: one of {"min", "max"}. In 'min' mode, lower metric is better.
            ckpt_path: where to save the best model checkpoint.
        """
        assert mode in ("min", "max"), f"`mode` attribute must be specified either `min` or `max`."
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.ckpt_path = ckpt_path
        self.accelerator = accelerator

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def step(
        self,
        val_score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        epoch: int
    ):
        score = val_score if self.mode == "max" else -val_score

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, optimizer, scheduler, epoch, score)
            return self.early_stop, self.best_score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop, None
        else:
            self.best_score = score
            self._save_checkpoint(model, optimizer, scheduler, epoch, score)
            self.counter = 0
            return self.early_stop, self.best_score

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        epoch: int,
        best_metric: float
    ):
        """Saves model when metric improves."""
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return
        try:
            # Get the unwrapped model state
            if self.accelerator is not None:
                model_state = self.accelerator.get_state_dict(model)
            else:
                model_state = model.state_dict()
            # Save checkpoint directly
            checkpoint_data = {
                "epoch": epoch,
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "best_metric": best_metric,
            }
            
            # Add multi-view configuration if applicable
            if hasattr(model, 'multi_view') and model.multi_view:
                checkpoint_data.update({
                    "num_views": model.num_views,
                    "view_fusion_type": model.view_fusion_type,
                    "adapter_dim": getattr(model, 'adapter_dim', None),
                    "view_fusion_hidden_dim": getattr(model, 'view_fusion_hidden_dim', None),
                })
            
            # Add Ark-specific configuration if applicable
            if hasattr(model, 'use_backbone_projector'):
                checkpoint_data.update({
                    "use_backbone_projector": model.use_backbone_projector,
                })
            
            torch.save(checkpoint_data, self.ckpt_path)
            logger.info(f"New best validation metric = {best_metric:.4f} at epoch {epoch+1}, saved best.pt")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    def load_best_model(self, model: torch.nn.Module):
        """Loads the best saved model weights into `model`."""
        try:
            # Load checkpoint with proper device placement
            ckpt = torch.load(self.ckpt_path, map_location='cpu')
            # Load state dict to model
            model.load_state_dict(ckpt["model_state"])
            return model
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise