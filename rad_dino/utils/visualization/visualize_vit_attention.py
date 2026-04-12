# Note the original is taken from dino repo:https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
import random
import colorsys
import numpy as np
import cv2
import torch
import torch.nn as nn
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon  
from skimage.measure import find_contours
from skimage.io import imread
from torchvision.transforms import ToPILImage
from typing import Union
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)
DEFAULT_OVERLAY_CMAP = "jet"
DEFAULT_OVERLAY_ALPHA = 0.6
DEFAULT_OVERLAY_MIN_ALPHA = 0.075
DEFAULT_PERCENTILE_CLIP = 99.0

def _normalize_heatmap(attention_map: np.ndarray) -> np.ndarray:
    """Normalize a heatmap to [0, 1] for stable visualization."""
    attention_map = np.asarray(attention_map, dtype=np.float32)
    attention_map = attention_map - attention_map.min()
    max_val = attention_map.max()
    if max_val <= 1e-8:
        return np.zeros_like(attention_map, dtype=np.float32)
    return attention_map / max_val


def _compute_mass_threshold_mask(attention_map: np.ndarray, threshold: float) -> np.ndarray:
    """Keep only the top attention mass defined by ``threshold``."""
    flat_attention = torch.as_tensor(attention_map, dtype=torch.float32).flatten()
    flat_attention = flat_attention / (flat_attention.sum() + 1e-8)
    values, sort_idx = torch.sort(flat_attention)
    cumulative = torch.cumsum(values, dim=0)
    masked_flat = cumulative > (1 - threshold)
    masked_flat = masked_flat[torch.argsort(sort_idx)]
    return masked_flat.reshape(attention_map.shape).cpu().numpy()

def _apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def _smooth_attention_overlay(img_arr: np.ndarray,
                             attention_map: np.ndarray,
                             cmap: str = DEFAULT_OVERLAY_CMAP,
                             alpha: float = DEFAULT_OVERLAY_ALPHA,
                             min_alpha: float = DEFAULT_OVERLAY_MIN_ALPHA,
                             percentile_clip: float = DEFAULT_PERCENTILE_CLIP) -> np.ndarray:
    """
    Create an attention-weighted colormap overlay with nonlinear scaling.

    Applies percentile clipping (to suppress border/corner artifacts common
    in DINO-family models) followed by power-law gamma correction (to expand
    the mid-range attention into stronger colours).  Per-pixel alpha is then
    proportional to the transformed attention, keeping low-attention
    background nearly transparent.

    Args:
        img_arr: RGB image [H, W, 3] in [0, 1].
        attention_map: 2-D attention map [H, W].
        cmap: Matplotlib colormap name.
        alpha: Maximum blending factor at the highest attention value.
        min_alpha: Minimum blending factor at the lowest attention value.
        percentile_clip: Upper percentile at which to clip before re-normalising
            (e.g. 98 caps border artifact spikes).

    Returns:
        Blended image as uint8 array [H, W, 3].
    """
    attention_map = _normalize_heatmap(attention_map)

    # 1. Percentile clip: cap extreme border artifacts, then re-normalise
    if 0 < percentile_clip < 100:
        cap = np.percentile(attention_map, percentile_clip)
        if cap > 1e-8:
            attention_map = np.clip(attention_map, 0, cap) / cap

    # 2. Gaussian blur for spatial smoothness
    if min(attention_map.shape[:2]) > 2:
        ksize = max(3, (min(attention_map.shape[:2]) // 40) | 1)
        attention_map = cv2.GaussianBlur(attention_map, (ksize, ksize), sigmaX=2.0)
        attention_map = _normalize_heatmap(attention_map)

    colormap = plt.get_cmap(cmap)
    heatmap = colormap(attention_map)[:, :, :3]
    per_pixel_alpha = (min_alpha + (alpha - min_alpha) * attention_map)[:, :, np.newaxis]
    blended = (1 - per_pixel_alpha) * img_arr + per_pixel_alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)

def _compute_attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean", num_prefix_tokens=1):
    """
    Compute attention rollout across all layers.
    
    Args:
        attentions: Attention tensor from all layers (num_layers, num_heads, seq_len, seq_len)
        discard_ratio: Ratio of lowest attention values to discard (0-1)
        head_fusion: How to fuse attention heads ("mean", "max", "min")
    
    Returns:
        rollout: Attention rollout tensor (seq_len-1,) representing attention from CLS to patches
    """
    # Validate input tensor shape
    if len(attentions.shape) != 4:
        raise ValueError(f"Expected attention tensor with 4 dimensions (num_layers, num_heads, seq_len, seq_len), got {attentions.shape}")
    
    # attentions is already a torch tensor with shape (num_layers, num_heads, seq_len, seq_len)
    device = attentions.device
    num_layers, num_heads, seq_len, seq_len_2 = attentions.shape
    
    if seq_len != seq_len_2:
        raise ValueError(f"Attention tensor should be square, got shape {attentions.shape}")
    
    # Initialize rollout with identity matrix
    result = torch.eye(seq_len, device=device)
    
    with torch.no_grad():
        # Process all layers
        for layer_idx in range(num_layers):
            attention = attentions[layer_idx]  # (num_heads, seq_len, seq_len)
            
            # Fuse attention heads
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=0)  # (seq_len, seq_len)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(dim=0)[0]  # (seq_len, seq_len)
            elif head_fusion == "min":
                attention_heads_fused = attention.min(dim=0)[0]  # (seq_len, seq_len)
            else:
                raise ValueError(f"Attention head fusion type '{head_fusion}' not supported")

            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(-1)
            _, indices = flat.topk(int(flat.numel() * discard_ratio), largest=False) 
            indices = indices[indices != 0]  # Don't drop class token
            flat[indices] = 0

            # Add identity matrix and normalize
            I = torch.eye(attention_heads_fused.size(-1), device=device)
            a = (attention_heads_fused + 1.0 * I) / 2
            denom = a.sum(dim=-1)
            if torch.any(denom == 0):
                logger.error(f"Zero row sum detected in attention normalization at layer {layer_idx}.")
                raise RuntimeError(f"Zero row sum detected in attention normalization at layer {layer_idx}.")
            a = a / denom
            if torch.isnan(a).any():
                logger.error(f"NaN detected in normalized attention matrix 'a' at layer {layer_idx}.")
                raise RuntimeError(f"NaN detected in normalized attention matrix 'a' at layer {layer_idx}.")
            
            # Update rollout
            result = torch.matmul(a, result)
    
    # Look at the total attention between the class token and the image patches
    mask = result[0, num_prefix_tokens:]  # Skip CLS + any register tokens

    # Debug: Check if mask is all zeros or nearly all zeros
    if torch.all(mask == 0) or torch.isclose(mask, torch.zeros_like(mask)).all():
        logger.error(f"Attention rollout result is all zeros for head_fusion={head_fusion}.")
        raise RuntimeError("Attention rollout result is all zeros! This indicates a problem with the attention maps or fusion/discard settings.")
    if torch.isnan(mask).any():
        logger.error(f"NaN detected in rollout mask for head_fusion={head_fusion}.")
        raise RuntimeError("NaN detected in rollout mask! This indicates a numerical issue during attention rollout computation.")

    logger.debug(f"Rollout computation completed. Output shape: {mask.shape}.")
    return mask    


def _gradient_rollout(attentions, gradients, discard_ratio=0.9,
                      num_prefix_tokens=1, return_full_matrix=False):
    """
    Compute gradient-weighted attention rollout for class-specific explainability.
    
    Each layer's attention is element-wise multiplied by its gradient w.r.t. the
    target class, averaged across heads, and negative contributions are zeroed
    before applying standard rollout.
    
    Adapt from https://github.com/jacobgil/vit-explain

    Args:
        attentions: List of per-layer attention tensors, each [num_heads, seq_len, seq_len].
        gradients: List of per-layer attention gradient tensors (same shapes).
        discard_ratio: Fraction of lowest values to zero out per layer.
        num_prefix_tokens: Number of prefix tokens to skip (CLS + registers).
        return_full_matrix: If True, return the full [N, N] rollout matrix instead
            of the CLS→patches row.  Required for SigLIP where a pooler combines
            with the encoder rollout.

    Returns:
        If ``return_full_matrix``: Tensor [N, N].
        Otherwise: Tensor [num_patches] — gradient-weighted rollout from CLS to patches.
    """
    device = attentions[0].device
    seq_len = attentions[0].size(-1)
    result = torch.eye(seq_len, device=device)

    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            attention_heads_fused = (attention * grad).mean(dim=0)
            attention_heads_fused = attention_heads_fused.clamp(min=0)

            flat = attention_heads_fused.view(-1)
            k = int(flat.numel() * discard_ratio)
            if 0 < k < flat.numel():
                _, indices = flat.topk(k, largest=False)
                flat[indices] = 0

            I = torch.eye(seq_len, device=device)
            a = (attention_heads_fused + I) / 2
            denom = a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            a = a / denom

            result = torch.matmul(a, result)

    if return_full_matrix:
        return result

    mask = result[0, num_prefix_tokens:]
    if mask.max() > 0:
        mask = mask / mask.max()
    return mask

def _grad_rollout_class_suffix(logits, category_index, num_classes):
    """Build a descriptive filename suffix for gradient rollout results.

    For binary classification (single output neuron) the suffix contains the
    sigmoid probability, matching GradCAM's convention.  For multi-class it
    contains the class index.
    """
    if num_classes == 1:
        prob = torch.sigmoid(logits[0, 0]).item()
        return f"binary_prob_{prob:.3f}"
    return f"class{category_index}"


def _save_gradient_rollout_visualizations(
    rollout_map, image_tensor, image_mean, image_std,
    output_dir, image_id, class_suffix,
):
    """Save gradient rollout heatmap, overlay, and masked visualization."""
    os.makedirs(output_dir, exist_ok=True)

    img_denorm = image_tensor.detach().cpu() * image_std.cpu() + image_mean.cpu()
    img_denorm = torch.clamp(img_denorm, 0, 1)
    original_image = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    original_image_float = original_image.astype(np.float32) / 255.0

    plt.imsave(
        os.path.join(output_dir, f"grad_rollout_{image_id}_{class_suffix}.png"),
        rollout_map, format='png', cmap='viridis', dpi=300,
    )
    overlay = _smooth_attention_overlay(original_image_float, rollout_map)
    plt.imsave(
        os.path.join(output_dir, f"grad_rollout_overlay_{image_id}_{class_suffix}.png"),
        overlay, format='png', dpi=300,
    )
    masked = _compute_mass_threshold_mask(rollout_map, 0.6)
    _display_instances(
        original_image, masked,
        fname=os.path.join(output_dir, f"grad_rollout_masked_{image_id}_{class_suffix}.png"),
        blur=False, figsize=(8, 8),
    )
    logger.info(f"Gradient rollout saved for image={image_id}, {class_suffix}")


# Gradient rollout for DINO/ViT
def compute_vit_gradient_rollout(
    model,
    input_tensor,
    image_mean,
    image_std,
    category_index=None,
    discard_ratio=0.9,
    num_prefix_tokens=1,
    patch_size=14,
    output_dir=None,
    image_id=None,
):
    """
    End-to-end gradient attention rollout for a HuggingFace ViT/DINO model.

    Performs a forward pass (with gradient tracking), backward pass from the
    target class, then computes and optionally saves the gradient-weighted
    rollout visualisation.

    Args:
        model: A DinoClassifier (or compatible) with a HuggingFace ViT backbone.
        input_tensor: Input image tensor [1, C, H, W].
        image_mean: Channel means for denormalisation [3, 1, 1].
        image_std: Channel stds for denormalisation [3, 1, 1].
        category_index: Target class. ``None`` → uses the predicted class.
        discard_ratio: Fraction of lowest values to discard.
        num_prefix_tokens: Prefix tokens to skip (CLS + registers).
        patch_size: ViT patch size for reshaping the rollout to a spatial map.
        output_dir: Directory to save visualisations.  ``None`` → skip saving.
        image_id: Sample identifier for filenames.

    Returns:
        rollout_map_np: the [H, W] rollout heatmap normalised to [0, 1]
        category_index: the class index used.
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()

    backbone_outputs = model.backbone(
        input_tensor,
        output_attentions=True,
        return_dict=True,
    )

    layer_attentions = backbone_outputs.attentions
    for attn in layer_attentions:
        attn.retain_grad()

    cls_tokens = backbone_outputs.last_hidden_state[:, 0]
    logits = model.classifier(cls_tokens)
    num_classes = logits.shape[-1]

    if category_index is None:
        category_index = logits.argmax(dim=-1).item()

    model.zero_grad()
    target = torch.zeros_like(logits)
    target[:, category_index] = 1.0
    (logits * target).sum().backward()

    attentions_list = [a[0].detach().cpu() for a in layer_attentions]
    gradients_list = [a.grad[0].detach().cpu() for a in layer_attentions]

    mask = _gradient_rollout(
        attentions_list, gradients_list,
        discard_ratio=discard_ratio,
        num_prefix_tokens=num_prefix_tokens,
    )

    w_feat = input_tensor.shape[-2] // patch_size
    h_feat = input_tensor.shape[-1] // patch_size
    rollout_spatial = mask.reshape(w_feat, h_feat)
    target_size = (input_tensor.shape[-2], input_tensor.shape[-1])

    rollout_upsampled = nn.functional.interpolate(
        rollout_spatial.unsqueeze(0).unsqueeze(0).float(),
        size=target_size, mode="bilinear", align_corners=False,
    )[0, 0].numpy()

    rollout_upsampled = _normalize_heatmap(rollout_upsampled)

    if output_dir is not None and image_id is not None:
        class_suffix = _grad_rollout_class_suffix(logits, category_index, num_classes)
        _save_gradient_rollout_visualizations(
            rollout_upsampled, input_tensor[0], image_mean, image_std,
            output_dir, image_id, class_suffix,
        )

    return rollout_upsampled, category_index

# Gradient rollout for BiomedCLIP
def _hook_attn_with_grad(attn_module, storage: list):
    """
    Monkey-patch an open_clip/timm Attention.forward to capture attention
    weights with gradient tracking.
    """
    def forward(x, **kwargs):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, attn_module.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * attn_module.scale
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)
        attn.retain_grad()
        storage.append(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x
    return forward

def compute_biomedclip_gradient_rollout(
    model,
    input_tensor,
    image_mean,
    image_std,
    category_index=None,
    discard_ratio=0.9,
    num_prefix_tokens=1,
    patch_size=16,
    output_dir=None,
    image_id=None,
):
    """
    End-to-end gradient attention rollout for a BiomedCLIP (open_clip ViT) model.

    Monkey-patches timm Attention blocks to capture attention weights with
    gradient tracking, then performs forward and backward passes, and computes
    gradient-weighted rollout.

    Args:
        model: A BiomedCLIPClassifier.
        input_tensor: [1, C, H, W].
        image_mean, image_std: For denormalisation.
        category_index: Target class (``None`` → predicted).
        discard_ratio: Fraction to discard.
        num_prefix_tokens: CLS + register tokens to skip.
        patch_size: ViT patch size.
        output_dir: Save directory (``None`` → skip).
        image_id: Sample id for filenames.

    Returns:
        ``(rollout_map_np, category_index)``.
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    model.eval()

    trunk = model.backbone.visual.trunk
    if trunk is None or not hasattr(trunk, "blocks"):
        raise RuntimeError("Cannot find visual trunk blocks for BiomedCLIP gradient rollout")

    attn_storage: list = []
    original_forwards: list = []
    for block in trunk.blocks:
        attn_module = block.attn
        original_forwards.append((attn_module, attn_module.forward))
        attn_module.forward = _hook_attn_with_grad(attn_module, attn_storage)

    try:
        features = model.backbone.encode_image(input_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        logits = model.classifier(features)
        num_classes = logits.shape[-1]

        if category_index is None:
            category_index = logits.argmax(dim=-1).item()

        model.zero_grad()
        target = torch.zeros_like(logits)
        target[:, category_index] = 1.0
        (logits * target).sum().backward()

        attentions_list = [a[0].detach().cpu() for a in attn_storage]
        gradients_list = [a.grad[0].detach().cpu() for a in attn_storage]
    finally:
        for attn_module, orig_fwd in original_forwards:
            attn_module.forward = orig_fwd

    mask = _gradient_rollout(
        attentions_list, gradients_list,
        discard_ratio=discard_ratio,
        num_prefix_tokens=num_prefix_tokens,
    )

    w_feat = input_tensor.shape[-2] // patch_size
    h_feat = input_tensor.shape[-1] // patch_size
    rollout_spatial = mask.reshape(w_feat, h_feat)
    target_size = (input_tensor.shape[-2], input_tensor.shape[-1])

    rollout_upsampled = nn.functional.interpolate(
        rollout_spatial.unsqueeze(0).unsqueeze(0).float(),
        size=target_size, mode="bilinear", align_corners=False,
    )[0, 0].numpy()
    rollout_upsampled = _normalize_heatmap(rollout_upsampled)

    if output_dir is not None and image_id is not None:
        class_suffix = _grad_rollout_class_suffix(logits, category_index, num_classes)
        _save_gradient_rollout_visualizations(
            rollout_upsampled, input_tensor[0], image_mean, image_std,
            output_dir, image_id, class_suffix,
        )

    return rollout_upsampled, category_index


def _random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def _display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = _random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = _apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"{fname} saved.")
    return

def _process_attentions_per_image(
    attention, 
    image, 
    image_output_dir, 
    image_mean, 
    image_std, 
    w_featmap, 
    h_featmap, 
    patch_size, 
    threshold, 
    head_fusion, 
    compute_rollout, 
    all_layer_attentions=None,
    rollout_discard_ratio=0.9,
    num_prefix_tokens=1,
):
    """
    Process attention maps for a single image/view.
    
    Args:
        attention: Attention tensor (num_heads, seq_len, seq_len)
        image: Input image tensor (C, H, W)
        image_output_dir: Directory to save visualizations
        image_mean: Mean values for image denormalization
        image_std: Standard deviation values for image denormalization
        w_featmap, h_featmap: Feature map dimensions
        patch_size: Patch size of the vision transformer
        threshold: Threshold for attention masking
        head_fusion: How to handle attention heads - "mean", "max", "min" for fusion across heads 
                     or int k to save k random per-head overlays
        compute_rollout: Whether to compute attention rollout in addition to raw attention maps
        all_layer_attentions: Attentions from all layers for rollout computation
        rollout_discard_ratio: Ratio of lowest attention values to discard in rollout computation (0-1, default: 0.9)
    """
    if compute_rollout and isinstance(head_fusion, int):
        raise ValueError("Attention rollout computation is only supported when attention head fusion is specified as 'mean', 'max', or 'min'.")
    # Process attention maps
    num_heads = attention.shape[0]  # number of heads
    
    # Keep only CLS→patch attention (skip CLS and any register tokens)
    attention_maps = attention[:, 0, num_prefix_tokens:].reshape(num_heads, -1)  # (num_heads, num_patches)
    
    target_size = (w_featmap * patch_size, h_featmap * patch_size)
    attention_maps = attention_maps.reshape(num_heads, w_featmap, h_featmap)
    attention_maps = nn.functional.interpolate(
        attention_maps.unsqueeze(0).float(), size=target_size, mode="bilinear", align_corners=False
    )[0].cpu()
    attention_maps = attention_maps.detach().numpy()

    # Save original image with proper denormalization
    image_denorm = image.clone()
    
    # Get normalization stats
    mean = image_mean.to(image.device)
    std = image_std.to(image.device)
    
    # Denormalize the tensor
    image_denorm = image_denorm * std + mean  # Denormalize
    image_denorm = torch.clamp(image_denorm, 0, 1)  # Ensure values are in [0, 1]
    
    # Save as PIL image
    input_image = ToPILImage()(image_denorm)
    input_image.save(os.path.join(image_output_dir, "original.png"))
    
    # Determine which heads to save
    if head_fusion == "max":
        # Fuse attention heads using max across heads (like vit-explain)
        attention_heads_fused = np.max(attention_maps, axis=0)
        heads_to_save = ["max_fused"]
        selected_attention_maps = np.expand_dims(attention_heads_fused, axis=0)
    elif head_fusion == "min":
        # Fuse attention heads using min across heads (like vit-explain)
        attention_heads_fused = np.min(attention_maps, axis=0)
        heads_to_save = ["min_fused"]
        selected_attention_maps = np.expand_dims(attention_heads_fused, axis=0)
    elif head_fusion == "mean":
        # Create mean attention map
        mean_attention = np.mean(attention_maps, axis=0)
        heads_to_save = ["mean_fused"]
        selected_attention_maps = np.expand_dims(mean_attention, axis=0)
    elif isinstance(head_fusion, int):
        if head_fusion > num_heads:
            logger.warning(f"Number of heads to save ({head_fusion}) is greater than the number of heads ({num_heads}). Using all heads.")
        heads_to_save = random.sample(range(num_heads), min(head_fusion, num_heads))
        selected_attention_maps = attention_maps[heads_to_save]
    else:
        raise ValueError(f"Head fusion type '{head_fusion}' not supported. Use 'mean', 'max', 'min' or an integer for random selection.")
    
    # Read the saved original image for visualization
    original_image = imread(os.path.join(image_output_dir, "original.png"))
    if original_image.shape[2] == 4:  # Remove alpha channel if present
        original_image = original_image[:, :, :3]
    
    # Prepare normalised original image for smooth overlay
    original_image_float = original_image.astype(np.float32) / 255.0

    # Save attention heatmaps for selected heads 
    for map_idx, head_idx in enumerate(heads_to_save):
        attention_map = _normalize_heatmap(selected_attention_maps[map_idx])
        masked_map = _compute_mass_threshold_mask(attention_map, threshold)
        
        # Save raw attention heatmap
        head_name = f"head_{head_idx}"
        fname = os.path.join(image_output_dir, f"attn_{head_name}.png")
        plt.imsave(fname=fname, arr=attention_map, format='png', cmap='viridis', dpi=300)
            
        # Create masked visualization
        mask_fname = os.path.join(image_output_dir, f"masked_head_{threshold * 100:.0f}%_{head_name}.png")
        _display_instances(
            original_image, 
            masked_map, 
            fname=mask_fname, 
            blur=False,
            figsize=(8, 8)
        )

        # Smooth colormap overlay
        overlay = _smooth_attention_overlay(
            original_image_float, attention_map, cmap=DEFAULT_OVERLAY_CMAP, alpha=DEFAULT_OVERLAY_ALPHA
        )
        overlay_fname = os.path.join(image_output_dir, f"overlay_{head_name}.png")
        plt.imsave(fname=overlay_fname, arr=overlay, format='png', dpi=300)
    
    # Compute attention rollout if requested
    if compute_rollout and all_layer_attentions is not None:
        logger.info(f"Computing attention rollout with {all_layer_attentions.shape[0]} layers and head_fusion={head_fusion}")
        # Compute rollout
        rollout_mask = _compute_attention_rollout(all_layer_attentions, discard_ratio=rollout_discard_ratio, head_fusion=head_fusion, num_prefix_tokens=num_prefix_tokens)
        width = int(rollout_mask.size(-1)**0.5)
        if width != w_featmap or width != h_featmap:
            raise ValueError(f"NotEqualError: width of rollout_mask: {width}, width_featmap: {w_featmap}, height_featmap: {h_featmap}")
        rollout_spatial = rollout_mask.reshape(width, width).cpu().numpy()
        
        # Interpolate to image size with bilinear for smoother output
        rollout_interpolated = nn.functional.interpolate(
            torch.from_numpy(rollout_spatial).unsqueeze(0).unsqueeze(0).float(), 
            size=target_size, 
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        
        # Normalize to [0, 1] range
        rollout_interpolated = _normalize_heatmap(rollout_interpolated)
        
        # Save rollout visualization
        rollout_fname = os.path.join(image_output_dir, f"rollout_{head_fusion}.png")
        plt.imsave(fname=rollout_fname, arr=rollout_interpolated, format='png', cmap='viridis', dpi=300)
        
        # Create masked rollout visualization
        rollout_mask_fname = os.path.join(image_output_dir, f"rollout_masked_{head_fusion}.png")
        rollout_thresholded = _compute_mass_threshold_mask(rollout_interpolated, threshold)
        
        _display_instances(
            original_image, 
            rollout_thresholded, 
            fname=rollout_mask_fname, 
            blur=False,
            figsize=(8, 8)
        )

        # Smooth rollout overlay
        rollout_overlay = _smooth_attention_overlay(
            original_image_float, rollout_interpolated, cmap=DEFAULT_OVERLAY_CMAP, alpha=DEFAULT_OVERLAY_ALPHA
        )
        rollout_overlay_fname = os.path.join(image_output_dir, f"rollout_overlay_{head_fusion}.png")
        plt.imsave(fname=rollout_overlay_fname, arr=rollout_overlay, format='png', dpi=300)

def visualize_attention_maps(
    attentions, 
    images, 
    image_ids, 
    output_dir, 
    accelerator, 
    image_mean,
    image_std,
    patch_size=14, 
    threshold=0.6, 
    head_fusion: Union[str, int] = "mean",
    compute_rollout: bool = False,
    rollout_discard_ratio: float = 0.9,
    ):
    """
    Visualize attention maps from attention tensors based on DINO repo.
    
    Args:
        attentions: Attention tensor (num_layers, B, num_heads, seq_len, seq_len) for single-view
                   or (num_layers, B, 4, num_heads, seq_len, seq_len) for multi-view
        images: Input image tensor [B, C, H, W] for single-view or [B, 4, C, H, W] for multi-view
        image_ids: List of image identifiers
        output_dir: Directory to save attention visualizations
        accelerator: Accelerator instance
        image_mean: Mean values for image denormalization
        image_std: Standard deviation values for image denormalization
        patch_size: Patch size of the vision transformer (default: 14 for dinov2-base)
        threshold: Threshold for attention masking (default: 0.6)
        head_fusion: How to handle attention heads - "mean", "max", "min" for fusion across heads
                     or int k to save k random per-head overlays
        compute_rollout: Whether to compute attention rollout in addition to raw attention maps
        rollout_discard_ratio: Ratio of lowest attention values to discard in rollout computation (0-1, default: 0.9)
    """
    if not accelerator.is_main_process:
        return
    
    batch_size = images.shape[0]
    is_multi_view = len(images.shape) == 5 and images.shape[1] > 1
    
    # Process all images in the batch
    logger.info(f"Processing attention maps for {batch_size} images (multi_view={is_multi_view})...")
    
    # Get last layer attention
    last_attention = attentions[-1]  # Last layer
    
    if is_multi_view:
        # Multi-view: (B, V, num_heads, seq_len, seq_len)
        logger.info(f"Multi-view attention shape: {last_attention.shape}")
        num_views = last_attention.shape[1]
        view_names = (
            ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
            if num_views == 4
            else [f"view_{i}" for i in range(num_views)]
        )
    else:
        # Single-view: (B, num_heads, seq_len, seq_len)
        logger.info(f"Single-view attention shape: {last_attention.shape}")
        num_views = 1
        view_names = ['single']
    
    seq_len = last_attention.shape[-1]
    w_featmap = images.shape[-2] // patch_size
    h_featmap = images.shape[-1] // patch_size
    expected_patches = w_featmap * h_featmap
    num_prefix_tokens = seq_len - expected_patches  # CLS + any register tokens
    logger.info(
        f"seq_len: {seq_len}, expected_patches: {expected_patches}, "
        f"prefix_tokens (CLS + registers if any to remove): {num_prefix_tokens}"
    )
    
    for idx in range(batch_size):
        image_id = image_ids[idx]
        
        if is_multi_view:
            # Multi-view: process each view separately
            for view_idx in range(num_views):
                attention = last_attention[idx, view_idx]  # (num_heads, seq_len, seq_len)
                image = images[idx, view_idx]  # (C, H, W)
                view_name = view_names[view_idx]
                
                # Create view-specific output directory
                image_output_dir = os.path.join(output_dir, f"attention_{image_id}_{view_name}")
                os.makedirs(image_output_dir, exist_ok=True)
                
                # Collect all layer attentions for this view if rollout is requested
                all_layer_attentions = None
                if compute_rollout:
                    all_layer_attentions = attentions[:, idx, view_idx, :, :, :] # (num_layers, num_heads, seq_len, seq_len)
                
                # Process attention maps for this view
                _process_attentions_per_image(
                    attention, image, image_output_dir, image_mean, image_std, 
                    w_featmap, h_featmap, patch_size, threshold, head_fusion, 
                    compute_rollout, all_layer_attentions, rollout_discard_ratio,
                    num_prefix_tokens=num_prefix_tokens,
                )
        else:
            # Single-view: standard processing
            attention = last_attention[idx]  # (num_heads, seq_len, seq_len)
            image = images[idx]  # (C, H, W)
            
            # Create output directory for this image
            image_output_dir = os.path.join(output_dir, f"attention_{image_id}")
            os.makedirs(image_output_dir, exist_ok=True)
            
            # Collect all layer attentions if rollout is requested
            all_layer_attentions = None
            if compute_rollout:
                all_layer_attentions = attentions[:, idx, :, :, :] # (num_layers, num_heads, seq_len, seq_len)
            
            # Process attention maps
            _process_attentions_per_image(
                attention, image, image_output_dir, image_mean, image_std, 
                w_featmap, h_featmap, patch_size, threshold, head_fusion, 
                compute_rollout, all_layer_attentions, rollout_discard_ratio,
                num_prefix_tokens=num_prefix_tokens,
                )
        
        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info(f"Processed attention maps: {idx + 1}/{batch_size} images")
    
    logger.info(f"Attention visualization completed for all {batch_size} images. Results saved in {output_dir}")
