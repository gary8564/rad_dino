import torchvision.transforms.v2 as transforms
from typing import Tuple
from rad_dino.utils.config_utils import get_model_config

def get_transforms(model_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Define the transforms for data augmentation. 
    Uses model-specific configurations to ensure consistent image processing with the pretrained model.
    
    Args:
        model_name: Name of the pretrained model
        
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Get model-specific configuration
    config = get_model_config(model_name)
    
    # Extract configuration parameters
    crop_size = tuple(config["crop_size"])
    size = config["size"]
    mean = config["image_mean"]
    std = config["image_std"]
    interpolation = config["interpolation"]
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
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
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(size, interpolation=interpolation),  
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, val_transform 