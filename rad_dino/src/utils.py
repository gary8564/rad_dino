import torchvision.transforms.v2 as transforms
import torch
from transformers import AutoImageProcessor

def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
}

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
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
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
        transforms.Resize(size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform