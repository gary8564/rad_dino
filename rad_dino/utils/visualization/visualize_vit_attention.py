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

def _apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def _compute_attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
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
    mask = result[0, 1:]  # Exclude class token attention to itself

    # Debug: Check if mask is all zeros or nearly all zeros
    if torch.all(mask == 0) or torch.isclose(mask, torch.zeros_like(mask)).all():
        logger.error(f"Attention rollout result is all zeros for head_fusion={head_fusion}.")
        raise RuntimeError("Attention rollout result is all zeros! This indicates a problem with the attention maps or fusion/discard settings.")
    if torch.isnan(mask).any():
        logger.error(f"NaN detected in rollout mask for head_fusion={head_fusion}.")
        raise RuntimeError("NaN detected in rollout mask! This indicates a numerical issue during attention rollout computation.")

    logger.debug(f"Rollout computation completed. Output shape: {mask.shape}.")
    return mask    

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
    print(f"{fname} saved.")
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
    
    # We keep only the output patch attention (exclude CLS token)
    # attention shape: (num_heads, seq_len, seq_len)
    # Remove the [CLS]→[CLS] entry and keep only [CLS]→patch 1..seq_len-1
    attention_maps = attention[:, 0, 1:].reshape(num_heads, -1)  # (num_heads, num_patches)
    
    # We keep only a certain percentage specified by threshold of the mass
    # Sort per head
    val, sort_idx = torch.sort(attention_maps)
    val /= torch.sum(val, dim=1, keepdim=True) # normalize each head's weights to sum=1
    cumval = torch.cumsum(val, dim=1) 
    # Mark the top (1-threshold)% of the mass
    masked_attns = cumval > (1 - threshold)
    # Aligned with original patch order
    sort_idx_inv = torch.argsort(sort_idx)
    for head in range(num_heads):
        masked_attns[head] = masked_attns[head][sort_idx_inv[head]]
    masked_attns = masked_attns.reshape(num_heads, w_featmap, h_featmap).float()
    
    # Interpolate
    masked_attention_maps = nn.functional.interpolate(masked_attns.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    attention_maps = attention_maps.reshape(num_heads, w_featmap, h_featmap)
    attention_maps = nn.functional.interpolate(attention_maps.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
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
        selected_masked_maps = np.expand_dims(attention_heads_fused, axis=0)  # For max, use same map
    elif head_fusion == "min":
        # Fuse attention heads using min across heads (like vit-explain)
        attention_heads_fused = np.min(attention_maps, axis=0)
        heads_to_save = ["min_fused"]
        selected_attention_maps = np.expand_dims(attention_heads_fused, axis=0)
        selected_masked_maps = np.expand_dims(attention_heads_fused, axis=0)  # For min, use same map
    elif head_fusion == "mean":
        # Create mean attention map
        mean_attention = np.mean(attention_maps, axis=0)
        mean_masked = np.mean(masked_attention_maps, axis=0)
        heads_to_save = ["mean_fused"]
        selected_attention_maps = np.expand_dims(mean_attention, axis=0)
        selected_masked_maps = np.expand_dims(mean_masked, axis=0)
    elif isinstance(head_fusion, int):
        if head_fusion > num_heads:
            logger.warning(f"Number of heads to save ({head_fusion}) is greater than the number of heads ({num_heads}). Using all heads.")
        heads_to_save = random.sample(range(num_heads), min(head_fusion, num_heads))
        selected_attention_maps = attention_maps[heads_to_save]
        selected_masked_maps = masked_attention_maps[heads_to_save]
    else:
        raise ValueError(f"Head fusion type '{head_fusion}' not supported. Use 'mean', 'max', 'min' or an integer for random selection.")
    
    # Read the saved original image for visualization
    original_image = imread(os.path.join(image_output_dir, "original.png"))
    if original_image.shape[2] == 4:  # Remove alpha channel if present
        original_image = original_image[:, :, :3]
    
    # Save attention heatmaps for selected heads 
    for map_idx, head_idx in enumerate(heads_to_save):
        attention_map = selected_attention_maps[map_idx]
        masked_map = selected_masked_maps[map_idx]
        
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
    
    # Compute attention rollout if requested
    if compute_rollout and all_layer_attentions is not None:
        logger.info(f"Computing attention rollout with {all_layer_attentions.shape[0]} layers and head_fusion={head_fusion}")
        # Compute rollout
        rollout_mask = _compute_attention_rollout(all_layer_attentions, discard_ratio=rollout_discard_ratio, head_fusion=head_fusion)
        width = int(rollout_mask.size(-1)**0.5)
        if width != w_featmap or width != h_featmap:
            raise ValueError(f"NotEqualError: width of rollout_mask: {width}, width_featmap: {w_featmap}, height_featmap: {h_featmap}")
        rollout_spatial = rollout_mask.reshape(width, width).cpu().numpy()
        
        # Interpolate to image size
        rollout_interpolated = nn.functional.interpolate(
            torch.from_numpy(rollout_spatial).unsqueeze(0).unsqueeze(0), 
            scale_factor=patch_size, 
            mode="nearest"
        )[0, 0].numpy()
        
        # Normalize to [0, 1] range
        rollout_interpolated = rollout_interpolated / np.max(rollout_interpolated)
        
        # Save rollout visualization
        rollout_fname = os.path.join(image_output_dir, f"rollout_{head_fusion}.png")
        plt.imsave(fname=rollout_fname, arr=rollout_interpolated, format='png', cmap='viridis', dpi=300)
        
        # Create masked rollout visualization
        rollout_mask_fname = os.path.join(image_output_dir, f"rollout_masked_{head_fusion}.png")
        # Apply threshold to rollout
        rollout_thresholded = np.where(rollout_interpolated > np.percentile(rollout_interpolated, (1-threshold)*100), rollout_interpolated, 0)
        
        _display_instances(
            original_image, 
            rollout_thresholded, 
            fname=rollout_mask_fname, 
            blur=False,
            figsize=(8, 8)
        )

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
    is_multi_view = len(images.shape) == 5 and images.shape[1] == 4
    
    # Process all images in the batch
    logger.info(f"Processing attention maps for {batch_size} images (multi_view={is_multi_view})...")
    
    # Get last layer attention
    last_attention = attentions[-1]  # Last layer
    
    if is_multi_view:
        # Multi-view: (B, 4, num_heads, seq_len, seq_len)
        logger.info(f"Multi-view attention shape: {last_attention.shape}")
        num_views = last_attention.shape[1]
        view_names = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']  # Standard mammography view order
    else:
        # Single-view: (B, num_heads, seq_len, seq_len)
        logger.info(f"Single-view attention shape: {last_attention.shape}")
        num_views = 1
        view_names = ['single']
    
    seq_len = last_attention.shape[-1]
    num_patches = seq_len - 1
    patch_dim = int(np.sqrt(num_patches)) 
    w_featmap = images.shape[-2] // patch_size
    h_featmap = images.shape[-1] // patch_size
    logger.info(f"patch_dim: {patch_dim}, num_patches: {num_patches}")
    assert patch_dim == w_featmap == h_featmap, f"NotEqualError: patch_dim: {patch_dim}, width_featmap: {w_featmap}, height_featmap: {h_featmap}"
    
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
                    compute_rollout, all_layer_attentions, rollout_discard_ratio
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
                compute_rollout, all_layer_attentions, rollout_discard_ratio
            )
        
        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info(f"Processed attention maps: {idx + 1}/{batch_size} images")
    
    logger.info(f"Attention visualization completed for all {batch_size} images. Results saved in {output_dir}")
