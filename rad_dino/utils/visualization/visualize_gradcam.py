import torch
import torch.nn as nn
import numpy as np
import math
import os
import logging
from PIL import Image
from torchvision.transforms import ToPILImage
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, BinaryClassifierOutputTarget
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.visualization.visualize_vit_attention import _smooth_attention_overlay
init_logging()
logger = logging.getLogger(__name__)

STANDARD_MAMMO_VIEW_NAMES = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']


def _infer_spatial_tokens(num_tokens: int, min_prefix_tokens: int = 0, max_prefix_tokens: int = 16):
    """Infer prefix-token count and square spatial size."""
    for num_prefix in range(min_prefix_tokens, max_prefix_tokens + 1):
        num_spatial = num_tokens - num_prefix
        if num_spatial <= 0:
            continue
        S = math.isqrt(num_spatial)
        if S * S == num_spatial:
            return num_prefix, S
    raise ValueError(
        f"Could not infer square spatial layout from token length {num_tokens}. "
        f"Tried prefix-token counts in [{min_prefix_tokens}, {max_prefix_tokens}]."
    )


class _PerViewGradCAMWrapper(nn.Module):
    """
    Wrapper class that runs a single view through backbone and classifier.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self._model.extract_features(x)
        return self._model.classifier(features)


def visualize_gradcam(model, input_tensor, target_layer, image_id, path_out, accelerator, image_mean, image_std, class_labels=None, threshold=0.5, has_cls_token=True):
    """
    Generate and save Grad-CAM heatmaps for positive labels.

    For multi-view inputs each view is explained independently through the
    backbone and classifier (bypassing view fusion).

    Args:
        model: Trained classifier model
        input_tensor: [1, C, H, W] (single-view) or [1, V, C, H, W] (multi-view)
        target_layer: Layer for Grad-CAM
        image_id: Image identifier
        path_out: Directory to save heatmaps
        accelerator: Accelerator instance
        image_mean: Mean values for image denormalization
        image_std: Standard deviation values for image denormalization
        threshold: Probability threshold for positive labels
        class_labels: List of class names (multilabel/multiclass only)
        has_cls_token: Whether the model uses a CLS token (affects reshape transform)
    """
    if not accelerator.is_main_process:
        return

    model.eval()
    input_tensor = input_tensor.to(accelerator.device)

    is_multi_view = input_tensor.dim() == 5 and input_tensor.shape[1] > 1

    if is_multi_view:
        logger.info(f"Generating per-view GradCAM for multi-view study {image_id}")
        _visualize_multiview_gradcam(
            model, input_tensor, target_layer, image_id, path_out,
            image_mean, image_std, class_labels, threshold, has_cls_token,
        )
    else:
        logger.info(f"Generating single-view GradCAM for {image_id}")
        _visualize_singleview_gradcam(
            model, input_tensor, target_layer, image_id, path_out,
            image_mean, image_std, class_labels, threshold, has_cls_token,
        )

def _get_predictions(model, input_tensor, image_id, threshold=0.5):
    """Get predictions from model."""
    with torch.no_grad():
        outputs = model(input_tensor)

    logits = outputs[0] if isinstance(outputs, tuple) else outputs
    num_classes = logits.shape[1]
    pred_probs = torch.sigmoid(logits).cpu().numpy()[0]

    if num_classes == 1:
        positive_indices = [1]
        logger.info(f"Binary classification: Probability = {pred_probs[0]:.3f}")
    else:
        positive_indices = np.where(pred_probs > threshold)[0]
        if len(positive_indices) == 0:
            logger.warning(f"No positive labels for {image_id}. Using top-3.")
            positive_indices = np.argsort(pred_probs)[-3:]

    return pred_probs, positive_indices, num_classes


def _get_class_label(class_idx, num_classes, pred_probs, class_labels):
    """Generate class label for filename."""
    if num_classes == 1:
        return f"_binary_prob_{pred_probs[0]:.3f}"
    if class_labels is not None:
        return f"_{class_labels[class_idx]}"
    return f"_class_{class_idx}"


def _get_target_function(num_classes, class_idx):
    """Get appropriate target function based on classification type."""
    if num_classes == 1:
        logger.info("Using BinaryClassifierOutputTarget(1) for binary classification")
        return BinaryClassifierOutputTarget(1)
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


def _denormalize_and_convert(tensor, image_mean, image_std):
    """Denormalize tensor and convert to PIL Image + float numpy array."""
    tensor = tensor * image_std + image_mean
    tensor = torch.clamp(tensor, 0, 1)
    pil_img = ToPILImage()(tensor)
    return pil_img, np.array(pil_img) / 255.0

def _reshape_with_cls(tensor):
    """[B, T, C] → [B, C, H, W], skipping CLS and register prefix tokens."""
    num_prefix, side = _infer_spatial_tokens(tensor.size(1), min_prefix_tokens=1)
    tensor = tensor[:, num_prefix:, :]
    B, _, C = tensor.size()
    return tensor.permute(0, 2, 1).reshape(B, C, side, side)


def _reshape_no_cls(tensor):
    """[B, N, C] or [B, H, W, C] → [B, C, H, W] (no prefix tokens)."""
    if tensor.dim() == 4:
        return tensor.permute(0, 3, 1, 2)
    B, N, C = tensor.size()
    side = int(math.sqrt(N))
    return tensor.permute(0, 2, 1).reshape(B, C, side, side)


def _get_reshape_transform(has_cls_token: bool):
    return _reshape_with_cls if has_cls_token else _reshape_no_cls


def _visualize_singleview_gradcam(model, input_tensor, target_layer, image_id,
                                  path_out, image_mean, image_std,
                                  class_labels, threshold, has_cls_token):
    pred_probs, positive_indices, num_classes = _get_predictions(
        model, input_tensor, image_id, threshold
    )

    wrapper = _PerViewGradCAMWrapper(model)
    reshape_transform = _get_reshape_transform(has_cls_token)
    cam = GradCAM(model=wrapper, target_layers=[target_layer],
                  reshape_transform=reshape_transform)

    single_view = input_tensor.squeeze(0)  # [C, H, W]
    pil_img, img_np = _denormalize_and_convert(single_view.cpu(), image_mean, image_std)
    pil_img.save(os.path.join(path_out, f'input_{image_id}.png'))

    for class_idx in positive_indices:
        class_label = _get_class_label(class_idx, num_classes, pred_probs, class_labels)
        target = _get_target_function(num_classes, class_idx)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])[0, :]
        overlay = _smooth_attention_overlay(img_np, grayscale_cam)
        Image.fromarray(overlay).save(
            os.path.join(path_out, f"gradcam_{image_id}{class_label}.png")
        )

    _save_predictions(pred_probs, image_id, path_out, class_labels)

def _visualize_multiview_gradcam(model, input_tensor, target_layer, image_id,
                                 path_out, image_mean, image_std,
                                 class_labels, threshold, has_cls_token):
    """Generate per-view GradCAM maps for a multi-view study.

    Each view is explained independently through backbone and classifier,
    bypassing the multi-view fusion layer.
    """
    # Study-level predictions (through full model including fusion)
    pred_probs, positive_indices, num_classes = _get_predictions(
        model, input_tensor, image_id, threshold
    )
    _save_predictions(pred_probs, image_id, path_out, class_labels)

    num_views = input_tensor.shape[1]
    view_names = (
        STANDARD_MAMMO_VIEW_NAMES
        if num_views == len(STANDARD_MAMMO_VIEW_NAMES)
        else [f"view_{i}" for i in range(num_views)]
    )

    wrapper = _PerViewGradCAMWrapper(model)
    reshape_transform = _get_reshape_transform(has_cls_token)
    cam = GradCAM(model=wrapper, target_layers=[target_layer],
                  reshape_transform=reshape_transform)

    batched_views = input_tensor.squeeze(0)  # [V, C, H, W]

    for view_idx, view_name in enumerate(view_names):
        single_view = batched_views[view_idx:view_idx + 1]  # [1, C, H, W]

        pil_img, img_np = _denormalize_and_convert(
            single_view.squeeze(0).cpu(), image_mean, image_std
        )
        pil_img.save(os.path.join(path_out, f'input_{image_id}_{view_name}.png'))

        for class_idx in positive_indices:
            class_label = _get_class_label(class_idx, num_classes, pred_probs, class_labels)
            target = _get_target_function(num_classes, class_idx)
            grayscale_cam = cam(input_tensor=single_view, targets=[target])[0, :]
            overlay = _smooth_attention_overlay(img_np, grayscale_cam)
            Image.fromarray(overlay).save(
                os.path.join(path_out, f"gradcam_{image_id}_{view_name}{class_label}.png")
            )
