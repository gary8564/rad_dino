import torchvision.transforms.v2 as transforms
import torch
from transformers import AutoImageProcessor
import logging
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

# Cope with variable-length custom dataset
def collate_fn(batch):
    imgs, targets, image_ids = zip(*batch)
    pixel_values = torch.stack(imgs, dim=0)    # [B, C, H, W]
    target = torch.stack(targets, dim=0)       # [B, num_classes]
    return pixel_values, target, image_ids

# Data augmentation: image transformation
def get_transforms(pretrained_model_path):
    """
    Define the transforms for data augmentation. 
    Get the transformers from pretrained model to ensure custom data is 
    transformed/formatted in the same way the data the original model was 
    trained on.
    """
    image_processor  = AutoImageProcessor.from_pretrained(pretrained_model_path)
    mean = image_processor.image_mean
    std = image_processor.image_std
    interpolation = image_processor.resample
    crop_size = (image_processor.crop_size["height"], image_processor.crop_size["width"])
    size = image_processor.size["shortest_edge"]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    # First resize all images to a consistent size before any other transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),  
        transforms.RandomResizedCrop(crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=10,                            # small rotation
            translate=(0.1,0.1),                   # 10% translations
            scale=(0.9,1.1),                       # ±10% zoom
            shear=10                               # ±10° shear
        ),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4))
        ], p=0.3),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1,2.0)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),  
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform

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
        ckpt_path: str = "best.pt"
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
            self._save_checkpoint(model, optimizer, scheduler, epoch, val_score)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(model, optimizer, scheduler, epoch, val_score)
            self.counter = 0

        return self.early_stop

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        epoch: int,
        best_metric: float
    ):
        """Saves model when metric improves."""
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "best_metric": best_metric,
        }, self.ckpt_path)
        logger.info(f"New best validation metric = {best_metric:.4f} at epoch {epoch+1}, saved best.pt")

    def load_best_model(self, model: torch.nn.Module):
        """Loads the best saved model weights into `model`."""
        ckpt = torch.load(self.ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        return model
