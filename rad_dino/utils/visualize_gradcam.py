import torch
import numpy as np
import math
import os
import logging
from PIL import Image
from torchvision.transforms import ToPILImage
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import AutoImageProcessor
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def visualize_gradcam(model, input_tensor, target_layer, image_id, path_out, accelerator, model_repo, class_labels=None, threshold=0.5):
    """
    Generate and save Grad-CAM heatmaps for positive labels.
    
    Args:
        model: Trained DinoClassifier model
        input_tensor: Input image tensor [1, C, H, W]
        target_layer: Layer for Grad-CAM (e.g., backbone.blocks[-1])
        image_id: Image identifier
        path_out: Directory to save heatmaps
        accelerator: Accelerator instance
        model_repo: Model repository for normalization stats
        threshold: Probability threshold for positive labels
        class_labels: List of class names, only used for multilabel/multiclass classification
    """
    if not accelerator.is_main_process:
        return
    
    model.eval()
    input_tensor = input_tensor.to(accelerator.device)

    # Initialize Grad-CAM
    def reshape_transform(tensor):
        """
        transformer tensor: [B, T, C] with T = 1 + H*W
        returns      : [B, C, H, W]
        """
        # 1) remove cls token
        tensor = tensor[:, 1:, :]
        B, N, C = tensor.size()
        H = W = int(math.sqrt(N))
        # 2) reshape   : [B, N, C] -> [B, C, H, W]
        tensor = tensor.permute(0, 2, 1).reshape(B, C, H, W)
        return tensor
    
    cam = GradCAM(
        model=model, 
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)  

    if isinstance(outputs, tuple):
        logits, _ = outputs  # Unpack tuple to get logits, ignore attentions
    else:
        logits = outputs  # In case model returns only logits [1, num_classes]
    
    # Determine if binary or multilabel classification
    num_classes = logits.shape[1]
    
    if num_classes == 1:
        # Binary classification with single output
        pred_probs = torch.sigmoid(logits).cpu().numpy()[0]
        positive_prob = pred_probs[0]
        # For binary classification, always visualize what contributes to the positive class prediction
        positive_indices = [0]
        logger.info(f"Binary classification: Probability = {positive_prob:.3f}")
            
    else:
        # Multilabel or multiclass classification
        pred_probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Select positive labels or top-3
        positive_indices = np.where(pred_probs > threshold)[0]
        if len(positive_indices) == 0:
            logger.warning(f"No positive labels for {image_id}. Using top-3.")
            positive_indices = np.argsort(pred_probs)[-3:]

    # Convert input tensor to RGB for visualization
    input_img = input_tensor.cpu().squeeze(0)
    
    # Get normalization stats from the model's image processor
    image_processor = AutoImageProcessor.from_pretrained(model_repo)
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    
    # Denormalize the tensor (reverse model normalization)
    input_img = input_img * std + mean  # Denormalize
    input_img = torch.clamp(input_img, 0, 1)  # Ensure values are in [0, 1]
    
    input_img = ToPILImage()(input_img)
    input_img_np = np.array(input_img) / 255.0

    # Save original image
    input_img.save(os.path.join(path_out, f'input_{image_id}.png'))

    # Generate overlay heatmaps
    for class_idx in positive_indices:
        if num_classes == 1:
            # Binary classification - add probability info to filename
            prob_val = pred_probs[0]
            class_label = f"_binary_prob_{prob_val:.3f}"
        elif class_labels is not None:
            class_label = f"_{class_labels[class_idx]}"
        else:
            class_label = f"_class_{class_idx}"
            
        targets = [ClassifierOutputTarget(class_idx)]
        # Pass eigen_smooth=True to apply smoothing; One image in a batch
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)[0, :]
        visualization = show_cam_on_image(input_img_np, grayscale_cam, use_rgb=True)
        visualization = Image.fromarray(visualization)
        visualization.save(os.path.join(path_out, f"gradcam_{image_id}{class_label}.png"))
