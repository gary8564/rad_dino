import torch
import numpy as np
import math
import os
import logging
from PIL import Image
from torchvision.transforms import ToPILImage
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def visualize_gradcam(model, input_tensor, target_layer, image_id, path_out, accelerator, image_mean, image_std, class_labels=None, threshold=0.5):
    """
    Generate and save Grad-CAM heatmaps for positive labels.
    
    Args:
        model: Trained DinoClassifier model
        input_tensor: Input image tensor [1, C, H, W] for single-view or [1, 4, C, H, W] for multi-view
        target_layer: Layer for Grad-CAM (e.g., backbone.blocks[-1])
        image_id: Image identifier
        path_out: Directory to save heatmaps
        accelerator: Accelerator instance
        image_mean: Mean values for image denormalization
        image_std: Standard deviation values for image denormalization
        threshold: Probability threshold for positive labels
        class_labels: List of class names, only used for multilabel/multiclass classification
    """
    if not accelerator.is_main_process:
        return
    
    model.eval()
    input_tensor = input_tensor.to(accelerator.device)
    
    # Check if this is multi-view input
    is_multi_view = len(input_tensor.shape) == 5 and input_tensor.shape[1] == 4
    
    if is_multi_view:
        logger.info(f"Generating multi-view GradCAM for {image_id}")
        _visualize_gradcam(
            model, input_tensor, target_layer, image_id, path_out, 
            image_mean, image_std, class_labels, threshold, is_multi_view=True
        )
    else:
        logger.info(f"Generating single-view GradCAM for {image_id}")
        _visualize_gradcam(
            model, input_tensor, target_layer, image_id, path_out, 
            image_mean, image_std, class_labels, threshold, is_multi_view=False
        )

def _get_predictions_and_positive_label_indices(model, input_tensor, image_id, threshold=0.5):
    """Get predictions and positive label indices."""
    with torch.no_grad():
        outputs = model(input_tensor)  

    if isinstance(outputs, tuple):
        logits, _ = outputs  # Unpack tuple to get logits, ignore attentions
    else:
        logits = outputs  # In case model returns only logits [1, num_classes]
    
    num_classes = logits.shape[1]
    pred_probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    if num_classes == 1:
        # Binary classification
        positive_indices = [1]
        logger.info(f"Binary classification: Probability = {pred_probs[0]:.3f}")
    else:
        # Multilabel or multiclass classification
        # Select positive labels or top-3
        positive_indices = np.where(pred_probs > threshold)[0]
        if len(positive_indices) == 0:
            logger.warning(f"No positive labels for {image_id}. Using top-3.")
            positive_indices = np.argsort(pred_probs)[-3:]
    
    return pred_probs, positive_indices, num_classes

def _get_class_label(class_idx, num_classes, pred_probs, class_labels):
    """Generate class label for filename."""
    if num_classes == 1:
        prob_val = pred_probs[0]
        return f"_binary_prob_{prob_val:.3f}"
    elif class_labels is not None:
        return f"_{class_labels[class_idx]}"
    else:
        return f"_class_{class_idx}"

def _denormalize_and_convert_to_PILImage(tensor, image_mean, image_std):
    """Denormalize tensor and convert to PIL Image."""
    # Denormalize
    tensor = tensor * image_std + image_mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    pil_img = ToPILImage()(tensor)
    return pil_img, np.array(pil_img) / 255.0

def _get_single_view_reshape_transform(tensor):
    """Single-view transformer tensor: [B, T, C] with T = 1 + H*W"""
    # remove cls token
    tensor = tensor[:, 1:, :]
    B, N, C = tensor.size()
    H = W = int(math.sqrt(N))
    # reshape: [B, N, C] -> [B, C, H, W]
    tensor = tensor.permute(0, 2, 1).reshape(B, C, H, W)
    return tensor

def _get_multiview_reshape_transform(tensor):
    """Multi-view transformer tensor: [B*V, T, C] where V=4 views, T=1+H*W"""
    BV, T, C = tensor.shape
    V = 4
    B = BV // V
    # remove CLS, reshape patch tokens
    x = tensor[:, 1:, :].reshape(B, V, T-1, C)   # [B, 4, H*W, C]
    H = W = int(math.sqrt(T-1))
    x = x.permute(0, 1, 3, 2).reshape(B*V, C, H, W) # back to [B*V, C, H, W]
    return x

def _get_reshape_transform(is_multi_view):
    """Get appropriate reshape transform for single or multi-view."""
    if is_multi_view:
        return _get_multiview_reshape_transform
    else:
        return _get_single_view_reshape_transform

def _get_target_function(num_classes, class_idx):
    """Get appropriate target function based on classification type."""
    if num_classes == 1:
        # Binary classification: use BinaryClassifierOutputTarget
        # For binary classification, we want to explain the positive class (disease presence)
        logger.info(f"Using BinaryClassifierOutputTarget(1) for binary classification")
        return BinaryClassifierOutputTarget(1)  # 1 for positive class
    else:
        # Multi-class/multi-label: use ClassifierOutputTarget
        logger.info(f"Using ClassifierOutputTarget({class_idx}) for multi-class/multi-label classification")
        return ClassifierOutputTarget(class_idx)

def _save_predictions(pred_probs, image_id, path_out, class_labels):
    """Save prediction probabilities to file."""
    with open(os.path.join(path_out, f'predictions_{image_id}.txt'), 'w') as f:
        f.write(f"Predictions for {image_id}:\n")
        if class_labels is not None:
            for label, prob in zip(class_labels, pred_probs):
                f.write(f"{label}: {prob:.4f}\n")
        elif len(pred_probs) == 1:
            f.write(f"Binary classification: {pred_probs[0]:.4f}\n")
        else:
            for i, prob in enumerate(pred_probs):
                f.write(f"Class {i}: {prob:.4f}\n")

def _visualize_gradcam(model, input_tensor, target_layer, image_id, path_out, image_mean, image_std, class_labels=None, threshold=0.5, is_multi_view=False):
    """GradCAM visualization for both single and multi-view inputs."""
    
    # Get predictions and target indices
    pred_probs, positive_indices, num_classes = _get_predictions_and_positive_label_indices(
        model, input_tensor, image_id, threshold
    )
    
    # Initialize GradCAM
    reshape_transform = _get_reshape_transform(is_multi_view)
    cam = GradCAM(
        model=model, 
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )
    
    if is_multi_view:
        _process_multiview_gradcam(
            cam, input_tensor, positive_indices, num_classes, pred_probs,
            image_id, path_out, image_mean, image_std, class_labels
        )
    else:
        _process_single_view_gradcam(
            cam, input_tensor, positive_indices, num_classes, pred_probs,
            image_id, path_out, image_mean, image_std, class_labels
        )
    
    # Save predictions
    _save_predictions(pred_probs, image_id, path_out, class_labels)

def _process_single_view_gradcam(cam, input_tensor, positive_indices, num_classes, pred_probs, image_id, path_out, image_mean, image_std, class_labels):
    """Process single-view GradCAM."""
    # Convert input tensor to RGB for visualization
    input_img_tensor = input_tensor.cpu().squeeze(0)
    input_img, input_img_np = _denormalize_and_convert_to_PILImage(input_img_tensor, image_mean, image_std)
    
    # Save original image
    input_img.save(os.path.join(path_out, f'input_{image_id}.png'))

    # Generate overlay heatmaps
    for class_idx in positive_indices:
        class_label = _get_class_label(class_idx, num_classes, pred_probs, class_labels)
        
        # Get appropriate target function
        target = _get_target_function(num_classes, class_idx)
        targets = [target]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        visualization = show_cam_on_image(input_img_np, grayscale_cam, use_rgb=True)
        visualization = Image.fromarray(visualization)
        visualization.save(os.path.join(path_out, f"gradcam_{image_id}{class_label}.png"))

def _process_multiview_gradcam(cam, input_tensor, positive_indices, num_classes, pred_probs, image_id, path_out, image_mean, image_std, class_labels):
    """Process multi-view GradCAM."""
    view_names = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
    
    # Prepare batched views: [1, 4, C, H, W] -> [4, C, H, W]
    batched_views = input_tensor.squeeze(0)  # [4, C, H, W]
    
    # Convert all views to RGB for visualization
    input_imgs_np = []
    for view_idx in range(4):
        # Extract single view for visualization
        single_view_tensor = batched_views[view_idx:view_idx+1].squeeze(0)  # [C, H, W]
        
        # Convert to RGB
        input_img, input_img_np = _denormalize_and_convert_to_PILImage(single_view_tensor, image_mean, image_std)
        input_imgs_np.append(input_img_np)
        
        # Save original view image
        input_img.save(os.path.join(path_out, f'input_{image_id}_{view_names[view_idx]}.png'))

    # Generate overlay heatmaps for each positive class
    for class_idx in positive_indices:
        class_label = _get_class_label(class_idx, num_classes, pred_probs, class_labels)
        
        # Get appropriate target function
        target = _get_target_function(num_classes, class_idx)
        targets = [target] * 4  # Same target for all 4 views
        
        # Compute GradCAM for all views at once
        grayscale_cams = cam(
            input_tensor=batched_views, 
            targets=targets, 
        )  # shape [4, H, W]
        
        # Map CAMs back to each view
        for view_idx, view_name in enumerate(view_names):
            cam_map = grayscale_cams[view_idx]
            visualization = show_cam_on_image(input_imgs_np[view_idx], cam_map, use_rgb=True)
            visualization = Image.fromarray(visualization)
            visualization.save(os.path.join(path_out, f"gradcam_{image_id}_{view_name}{class_label}.png"))
