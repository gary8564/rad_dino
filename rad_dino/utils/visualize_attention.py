# Note the original is taken from dino repo:https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
import skimage.io
import random
import colorsys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import ToPILImage
import os
import logging
from typing import Union
from transformers import AutoImageProcessor
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

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
        masked_image = apply_mask(masked_image, _mask, color, alpha)
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

def visualize_attention_maps(
    attentions, 
    images, 
    image_ids, 
    output_dir, 
    accelerator, 
    model_repo, 
    patch_size=14, 
    threshold=0.6, save_heads: Union[int, str, list[str]] = 5
    ):
    """
    Visualize attention maps from attention tensors based on DINO repo.
    
    Args:
        attentions: Attention tensor (num_layers, B, num_heads, seq_len, seq_len)
        images: Input image tensor [B, C, H, W]
        image_ids: List of image identifiers
        output_dir: Directory to save attention visualizations
        accelerator: Accelerator instance
        model_repo: Model repository for denormalization
        patch_size: Patch size of the vision transformer (default: 14 for dinov2-base)
        threshold: Threshold for attention masking (default: 0.6)
        save_heads: Which heads to save - integer for random N heads, "all", "mean", or list of head indices (e.g., [0, 5, 11])
    """
    if not accelerator.is_main_process:
        return
    
    batch_size = images.shape[0]
    
    # Process all images in the batch
    logger.info(f"Processing attention maps for {batch_size} images...")
    
    # Get last layer attention: (B, num_heads, seq_len, seq_len)
    last_attention = attentions[-1]  # Last layer
    
    seq_len = last_attention.shape[2]
    num_patches = seq_len - 1
    patch_dim = int(np.sqrt(num_patches)) 
    w_featmap = images.shape[-2] // patch_size
    h_featmap = images.shape[-1] // patch_size
    logger.info(f"patch_dim: {patch_dim}, num_patches: {num_patches}")
    assert patch_dim == w_featmap == h_featmap, f"NotEqualError: patch_dim: {patch_dim}, width_featmap: {w_featmap}, height_featmap: {h_featmap}"
    
    for idx in range(batch_size):
        image_id = image_ids[idx]
        attention = last_attention[idx]  # (num_heads, seq_len, seq_len)
        image = images[idx]  # (C, H, W)
        
        # Process attention maps
        num_heads = attention.shape[0]  # number of heads
        if isinstance(save_heads, list) and len(save_heads) > num_heads:
            raise ValueError(f"Number of heads to save ({len(save_heads)}) is greater than the number of heads in the model ({num_heads})")
        if isinstance(save_heads, int) and (save_heads > num_heads or save_heads < 1):
            raise ValueError(f"Number of heads to save ({save_heads}) is greater than the number of heads in the model ({num_heads}) or less than 1.")
        
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

        # Show and save attentions heatmaps
        # Create output directory for this image
        image_output_dir = os.path.join(output_dir, f"attention_{image_id}")
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Save original image with proper denormalization
        image_denorm = image.clone()
        
        # Get normalization stats from the model's image processor
        image_processor = AutoImageProcessor.from_pretrained(model_repo)
        mean = torch.tensor(image_processor.image_mean).view(3, 1, 1).to(image.device)
        std = torch.tensor(image_processor.image_std).view(3, 1, 1).to(image.device)
        
        # Denormalize the tensor (reverse model normalization)
        image_denorm = image_denorm * std + mean  # Denormalize
        image_denorm = torch.clamp(image_denorm, 0, 1)  # Ensure values are in [0, 1]
        
        # Save as PIL image
        input_image = ToPILImage()(image_denorm)
        input_image.save(os.path.join(image_output_dir, "original.png"))
        
        # Determine which heads to save
        if save_heads == "all":
            heads_to_save = list(range(num_heads))
            selected_attention_maps = attention_maps
            selected_masked_maps = masked_attention_maps
        elif save_heads == "mean":
            # Create mean attention map
            mean_attention = np.mean(attention_maps, axis=0)
            mean_masked = np.mean(masked_attention_maps, axis=0)
            heads_to_save = ["mean"]
            selected_attention_maps = np.expand_dims(mean_attention, axis=0)
            selected_masked_maps = np.expand_dims(mean_masked, axis=0)
        elif isinstance(save_heads, list):
            heads_to_save = [h for h in save_heads if h < num_heads]
            selected_attention_maps = attention_maps[heads_to_save]
            selected_masked_maps = masked_attention_maps[heads_to_save]
        else:
            heads_to_save = random.sample(range(num_heads), min(save_heads, num_heads))
            selected_attention_maps = attention_maps[heads_to_save]
            selected_masked_maps = masked_attention_maps[heads_to_save]
        
        # Save attention heatmaps for selected heads
        for map_idx, head_idx in enumerate(heads_to_save):
            attention_map = selected_attention_maps[map_idx]
            masked_map = selected_masked_maps[map_idx]
            
            # Save raw attention heatmap
            head_name = "mean" if head_idx == "mean" else f"head_{head_idx}"
            fname = os.path.join(image_output_dir, f"attn_{head_name}.png")
            plt.imsave(fname=fname, arr=attention_map, format='png', cmap='viridis', dpi=300)
                
            # Create masked visualization
            original_image = skimage.io.imread(os.path.join(image_output_dir, "original.png"))
            if original_image.shape[2] == 4:  # Remove alpha channel if present
                original_image = original_image[:, :, :3]    
            mask_fname = os.path.join(image_output_dir, f"masked_head_{threshold * 100:.0f}%_{head_name}.png")
            display_instances(
                original_image, 
                masked_map, 
                fname=mask_fname, 
                blur=False,
                figsize=(8, 8)
            )
        
        # Log progress every 10 images to avoid spam
        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info(f"Processed attention maps: {idx + 1}/{batch_size} images")
    
    logger.info(f"Attention visualization completed for all {batch_size} images. Results saved in {output_dir}")