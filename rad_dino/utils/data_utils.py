import torch
import logging
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """
    Custom collate function to handle variable-length custom dataset.
    
    Args:
        batch: Batch of data from the dataset
        
    Returns:
        Tuple of (pixel_values, target, sample_ids)
    """
    imgs, targets, sample_ids = zip(*batch)
    pixel_values = torch.stack(imgs, dim=0)    # [B, C, H, W] or [B, 4, C, H, W]
    target = torch.stack(targets, dim=0)       # [B, num_classes] or [B, 4, num_classes]
    return {"pixel_values": pixel_values, "labels": target, "sample_ids": sample_ids}
        
