"""
Swin Transformer Attention Visualization: 
hierarchical structure, window-based attention, and patch merging between stages.
"""

import os
import math
import torch
import random
import logging
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union
from skimage.io import imread
from rad_dino.utils.visualization.visualize_vit_attention import _display_instances, smooth_attention_overlay

logger = logging.getLogger(__name__)

def _standardize_shift_size_format(shift: Optional[Union[int, tuple, list]]) -> Tuple[int, int]:
    """
    Standardize shift size format to a tuple[int, int]. 
    """
    if shift is None:
        return (0, 0)
    if isinstance(shift, int):
        return (shift, shift)
    if isinstance(shift, (tuple, list)):
        if len(shift) == 2:
            return (int(shift[0]), int(shift[1]))
    return (0, 0)

def _compute_pad_grid(H: int, W: int, ws: int) -> Tuple[int, int, int, int, int]:
    """
    Compute padded grid and window tiling sizes given input resolution and window size.
    Returns (H_padded, W_padded, pad_h, pad_w, nW)
    """
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    H_padded = H + pad_h
    W_padded = W + pad_w
    nWh = H_padded // ws
    nWw = W_padded // ws
    nW = nWh * nWw
    return H_padded, W_padded, pad_h, pad_w, nW

def _fuse_heads(attn: torch.Tensor, head_fusion: str) -> torch.Tensor:
    """
    Fuse attention across heads.
    attn: [..., num_heads, N, N]
    returns: [..., N, N]
    """
    if head_fusion == "mean":
        return attn.mean(dim=-3)
    if head_fusion == "max":
        return attn.max(dim=-3)[0]
    if head_fusion == "min":
        return attn.min(dim=-3)[0]
    raise ValueError(f"Invalid head fusion type: {head_fusion}")

def _discard_low_values_per_row(a: torch.Tensor, discard_ratio: float) -> torch.Tensor:
    """
    Zero out the lowest discard_ratio fraction per row in the last dimension of a.
    a: [..., N, N]
    """
    if discard_ratio <= 0:
        return a
    N = a.shape[-1]
    k = int(N * discard_ratio)
    if k <= 0 or k >= N:
        return a
    # Get indices of smallest k per row
    vals, idx = torch.topk(a, k, dim=-1, largest=False)
    a = a.clone()
    a.scatter_(-1, idx, 0.0)
    return a

def _row_normalize_with_identity(a: torch.Tensor, identity_weight: float = 1.0) -> torch.Tensor:
    """
    Add identity, then row-normalize the last dimension.
    a: [..., N, N]
    """
    N = a.shape[-1]
    I = torch.eye(N, device=a.device, dtype=a.dtype)
    while I.dim() < a.dim():
        I = I.unsqueeze(0)
    a = a + identity_weight * I
    denom = a.sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=1e-8)
    return a / denom

def _apply_threshold_mask(attention_map: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Apply threshold masking to attention map like in ViT visualization.
    Keep only the top (1-threshold)% of attention mass.
    
    Args:
        attention_map: [H, W] spatial attention map
        threshold: Threshold for masking (0.6 means keep top 40% of mass)
    
    Returns:
        Binary mask [H, W] where 1 indicates high attention
    """
    # Flatten the attention map
    flat_attention = attention_map.flatten()
    
    # Normalize to sum=1
    flat_attention = flat_attention / (flat_attention.sum() + 1e-8)
    
    # Sort and compute cumulative sum
    val, sort_idx = torch.sort(flat_attention)
    cumval = torch.cumsum(val, dim=0)
    
    # Mark the top (1-threshold)% of the mass
    masked_flat = cumval > (1 - threshold)
    
    # Restore original order
    sort_idx_inv = torch.argsort(sort_idx)
    masked_flat = masked_flat[sort_idx_inv]
    
    # Reshape back to spatial dimensions
    masked_attention = masked_flat.reshape(attention_map.shape).float()
    
    return masked_attention

def _stitch_block_attention(
    attn_map: torch.Tensor,
    H: int,
    W: int,
    shift_size: Tuple[int, int],
    head_fusion: str,
    batch_size: int
) -> torch.Tensor:
    """
    Stitch a single block's per-window attention to a global spatial heatmap per image.
    Reverse TIMM's procedure: tile windows onto padded canvas, reverse shift (+shift_size), crop

    attn_map: [batch_size*num_windows, num_heads, N, N] where N = window_size*window_size
    Returns: [batch_size, H, W] fused spatial map (head-fused and query-fused by mean over queries)
    """
    BnW, num_heads, N, _ = attn_map.shape
    window_size = int(math.isqrt(N))

    # Compute padding to make H,W divisible by ws
    H_padded, W_padded, pad_h, pad_w, num_windows = _compute_pad_grid(H, W, window_size)
    B = BnW // num_windows
    assert B == batch_size, f"Extracted batch size {B} != expected {batch_size}"

    # Fuse heads, then fuse queries by mean to get per-key weights within each window
    attn_fused = _fuse_heads(attn_map, head_fusion=head_fusion)  # [batch_size*num_windows, N, N]
    per_key = attn_fused.mean(dim=-2)  # [batch_size*num_windows, N] - average over query positions
    
    # Reshape to [B, nW, ws, ws] for tiling
    per_key = per_key.view(B, num_windows, window_size, window_size)

    # Tile into canvas following TIMM's window_partition order
    # TIMM: view(B, H//ws, ws, W//ws, ws, C).permute(0,1,3,2,4,5).view(-1, ws, ws, C)
    num_window_rows = H_padded // window_size
    num_window_cols = W_padded // window_size
    assert num_windows == num_window_rows * num_window_cols
    
    canvas = attn_map.new_zeros((B, H_padded, W_padded))
    for b in range(B):
        for wh in range(num_window_rows):
            for ww in range(num_window_cols):
                idx = wh * num_window_cols + ww  # TIMM's window ordering
                h0 = wh * window_size
                w0 = ww * window_size
                canvas[b, h0:h0+window_size, w0:w0+window_size] = per_key[b, idx]

    # Reverse cyclic shift to original coordinates 
    norm_shift = _standardize_shift_size_format(shift_size)
    if any(norm_shift):
        canvas = torch.roll(canvas, shifts=norm_shift, dims=(1, 2))

    # Crop padding 
    canvas = canvas[:, :H, :W]
    return canvas  # [B, H, W]

def _stitch_block_attention_per_head(
    attn_map: torch.Tensor,
    H: int,
    W: int,
    shift_size: Tuple[int, int],
    batch_size: int
) -> torch.Tensor:
    """
    Stitch per-head window attentions into global spatial maps.
    attn_map: [B*nW, num_heads, N, N]; returns [B, num_heads, H, W].
    """
    BnW, num_heads, N, _ = attn_map.shape
    window_size = int(math.isqrt(N))

    H_padded, W_padded, pad_h, pad_w, num_windows = _compute_pad_grid(H, W, window_size)
    B = BnW // num_windows
    assert B == batch_size

    # Average over queries to get per-key weights per head
    per_key = attn_map.mean(dim=-2)  # [batch_size*num_windows, num_heads, N]
    per_key = per_key.view(B, num_windows, num_heads, window_size, window_size)

    num_window_rows = H_padded // window_size
    num_window_cols = W_padded // window_size
    canvas = attn_map.new_zeros((B, num_heads, H_padded, W_padded))
    for b in range(B):
        for wh in range(num_window_rows):
            for ww in range(num_window_cols):
                idx = wh * num_window_cols + ww
                h0 = wh * window_size
                w0 = ww * window_size
                canvas[b, :, h0:h0+window_size, w0:w0+window_size] = per_key[b, idx]

    norm_shift = _standardize_shift_size_format(shift_size)
    if any(norm_shift):
        canvas = torch.roll(canvas, shifts=norm_shift, dims=(2, 3))

    return canvas[:, :, :H, :W]

def _apply_block_rollout(
    S: torch.Tensor,
    attn_map: torch.Tensor,
    H: int,
    W: int,
    shift_size: Tuple[int, int],
    head_fusion: str,
    discard_ratio: float
) -> torch.Tensor:
    """
    Apply one block's attention as a local mixing operator to saliency S without building a global NxN.
    Mirror TIMM's processing: shift S by (-shift_size), pad, apply windowed attention, reverse.
    
    Args:
        S: [B, H, W] current saliency,
        attn_map: [B*nW, num_heads, N, N],
        H: int, 
        W: int, 
        shift_size: Tuple[int, int], 
        head_fusion: str, 
        discard_ratio: float
    Returns:
        Updated S of shape [B, H, W]
    """
    BnW, num_heads, N, _ = attn_map.shape
    window_size = int(math.isqrt(N))
    assert window_size * window_size == N
    H_padded, W_padded, pad_h, pad_w, num_windows = _compute_pad_grid(H, W, window_size)
    B = S.shape[0]
    assert BnW == B * num_windows, f"Batch mismatch: attn batch*num_windows={BnW}, expected {B}*{num_windows}"

    # Prepare S on shifted, padded grid (to match how attention windows were formed)
    S_shifted = S
    norm_shift = _standardize_shift_size_format(shift_size)
    if any(norm_shift):
        S_shifted = torch.roll(S_shifted, shifts=(-norm_shift[0], -norm_shift[1]), dims=(1, 2))
    if pad_h or pad_w:
        S_shifted = nn.functional.pad(S_shifted, (0, pad_w, 0, pad_h), mode='constant', value=0)

    # Fuse heads and prepare per-window adjacency matrices
    A = _fuse_heads(attn_map, head_fusion=head_fusion)  # [B*nW, N, N]
    A = _discard_low_values_per_row(A, discard_ratio=discard_ratio)
    A = _row_normalize_with_identity(A, identity_weight=1.0)  # [B*nW, N, N]

    # Apply per-window operator following TIMM's window ordering
    num_window_rows = H_padded // window_size
    num_window_cols = W_padded // window_size
    S_new = S_shifted.new_zeros((B, H_padded, W_padded))

    # Process each batch and window
    for b in range(B):
        for wh in range(num_window_rows):
            for ww in range(num_window_cols):
                idx = wh * num_window_cols + ww  # TIMM's window ordering
                window_idx = b * num_windows + idx
                
                h0 = wh * window_size
                w0 = ww * window_size
                s_win = S_shifted[b, h0:h0+window_size, w0:w0+window_size].reshape(-1)  # [N]
                a_win = A[window_idx]  # [N, N]
                s_out = a_win @ s_win  # [N]
                S_new[b, h0:h0+window_size, w0:w0+window_size] = s_out.view(window_size, window_size)

    # Reverse shift and crop
    norm_shift = _standardize_shift_size_format(shift_size)
    if any(norm_shift):
        S_new = torch.roll(S_new, shifts=norm_shift, dims=(1, 2))
    S_new = S_new[:, :H, :W]
    return S_new

def _downsample_by2(S: torch.Tensor) -> torch.Tensor:
    """Downsample spatial saliency map by 2x using average pooling (patch merging proxy)."""
    return nn.functional.avg_pool2d(S.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)

def _group_block_indices(stage_metadata: List[Dict]) -> List[int]:
    """Return sorted unique block indices present in stage metadata."""
    blocks = sorted({int(meta.get('block', 0)) for meta in stage_metadata})
    return blocks

def _filter_maps_by_block(attn_maps: List[torch.Tensor], metadata: List[Dict], block_idx: int) -> Tuple[torch.Tensor, Dict]:
    """
    Return the single attention map and metadata for a given block index.
    Enforces exactly one capture per block.
    """
    idxs = [i for i, m in enumerate(metadata) if int(m.get('block', -1)) == block_idx]
    if len(idxs) == 0:
        raise RuntimeError(f"No attention map found for block index {block_idx}.")
    if len(idxs) > 1:
        raise RuntimeError(f"Expected exactly one attention map for block index {block_idx}, found {len(idxs)}.")
    i = idxs[0]
    return attn_maps[i], metadata[i]

def get_last_block_global_attention_map(
    hierarchical_attentions: Dict,
    image_index: int,
    head_fusion: str,
    batch_size: int
) -> Optional[torch.Tensor]:
    """
    Build a single global spatial attention map from the LAST BLOCK of the LAST STAGE,
    aligned to the original image grid (stitched across windows, reverse-shifted, cropped).

    This corresponds to the attention right before the linear/classifier head in practice
    and mirrors the ViT convention of showing the last block.

    Args:
        hierarchical_attentions: Stage-wise attention maps and metadata
        image_index: Index of image in batch
        head_fusion: "mean" | "max" | "min"
        batch_size: Batch size for validation

    Returns:
        Tensor [H, W] or None if unavailable
    """
    if not hierarchical_attentions:
        return None
    last_stage = max(hierarchical_attentions.keys())
    stage_data = hierarchical_attentions[last_stage]
    if not stage_data['attention_maps']:
        return None
    # Find last block index present in metadata
    blocks = _group_block_indices(stage_data['metadata'])
    if not blocks:
        return None
    last_block_idx = max(blocks)
    attn_map, meta = _filter_maps_by_block(stage_data['attention_maps'], stage_data['metadata'], last_block_idx)
    shift = meta.get('shift_size', (0, 0))
    H, W = stage_data['stage_info']['input_resolution']
    stitched = _stitch_block_attention(
        attn_map=attn_map,
        H=H,
        W=W,
        shift_size=shift,
        head_fusion=head_fusion,
        batch_size=batch_size,
    )  # [B, H, W]
    return stitched[image_index]

def _compute_swin_attention_rollout(
    hierarchical_attentions: Dict, 
    batch_size: int,
    discard_ratio: float = 0.9, 
    head_fusion: str = "mean") -> torch.Tensor:
    """
    Compute attention rollout for Swin Transformer with hierarchical structure using a windowed operator.

    Returns per-batch rollout map at the last stage resolution: [B, H_last, W_last].
    """
    # Get first stage to determine initial resolution and batch size
    first_stage = min(hierarchical_attentions.keys())
    fs_data = hierarchical_attentions[first_stage]
    fs_maps = fs_data['attention_maps']
    assert fs_maps, "No attention maps available"
    
    # Determine batch size from first attention map and stage resolution
    BnW = fs_maps[0].shape[0]
    H0, W0 = fs_data['stage_info']['input_resolution']
    N = fs_maps[0].shape[-1]
    window_size = int(math.isqrt(N))
    _, _, _, _, num_windows_stage0 = _compute_pad_grid(H0, W0, window_size)
    
    # Verify batch size is consistent
    assert BnW % num_windows_stage0 == 0, f"Attention batch dim {BnW} not divisible by num_windows {num_windows_stage0}"
    B = BnW // num_windows_stage0
    if B != batch_size:
        raise ValueError(f"Computed batch size {B} != expected {batch_size}.")

    # Initialize S at stage 0 resolution with ones
    S = fs_maps[0].new_ones((B, H0, W0))

    # Process stages in order
    stage_ids = sorted(hierarchical_attentions.keys())
    for stage_idx in stage_ids:
        stage_data = hierarchical_attentions[stage_idx]
        Hs, Ws = stage_data['stage_info']['input_resolution']
        
        # If current S is at a different resolution, downsample via pooling
        if S.shape[-2:] != (Hs, Ws):
            # Expect downsample by factor 2 per stage transition
            while S.shape[-2] > Hs or S.shape[-1] > Ws:
                S = _downsample_by2(S)
                
        # Apply each block in order
        blocks = _group_block_indices(stage_data['metadata'])
        for bidx in blocks:
            attn_map, meta = _filter_maps_by_block(stage_data['attention_maps'], stage_data['metadata'], bidx)
            shift = meta.get('shift_size', (0, 0))
            S = _apply_block_rollout(S, attn_map, Hs, Ws, shift_size=shift, head_fusion=head_fusion, discard_ratio=discard_ratio)

    return S  # [B, H_last, W_last]

def compute_swin_rollout(
    hierarchical_attentions: Dict,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Public helper to compute Swin attention rollout across blocks and stages.

    Args:
        hierarchical_attentions: Stage-wise attention maps and metadata
        head_fusion: Head fusion strategy ("mean", "max", "min")
        discard_ratio: Fraction of lowest attention per-row to discard before normalization
        batch_size: Optional expected batch size for validation; if None it's inferred

    Returns:
        Tensor of shape [B, H_last, W_last] at the last stage resolution
    """
    inferred_batch = None
    # Try to infer batch from first stage entry for better error messages
    if hierarchical_attentions:
        first_stage = min(hierarchical_attentions.keys())
        fs_data = hierarchical_attentions[first_stage]
        if fs_data['attention_maps']:
            BnW = fs_data['attention_maps'][0].shape[0]
            H0, W0 = fs_data['stage_info']['input_resolution']
            N = fs_data['attention_maps'][0].shape[-1]
            window_size = int(math.isqrt(N))
            _, _, _, _, num_windows_stage0 = _compute_pad_grid(H0, W0, window_size)
            if num_windows_stage0 > 0 and BnW % num_windows_stage0 == 0:
                inferred_batch = BnW // num_windows_stage0
    bs = batch_size if batch_size is not None else inferred_batch
    return _compute_swin_attention_rollout(
        hierarchical_attentions=hierarchical_attentions,
        discard_ratio=discard_ratio,
        head_fusion=head_fusion,
        batch_size=bs,
    )

def _process_swin_attentions_per_image(
    hierarchical_attentions: Dict,
    image: torch.Tensor,
    image_index: int,
    image_output_dir: str,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    head_fusion: Union[str, int],
    threshold: float,
    compute_rollout: bool,
    batch_size: int,
    rollout_discard_ratio: float = 0.9,
):
    """
    Process Swin attention maps for a single image.
    """
    if compute_rollout and isinstance(head_fusion, int):
        raise ValueError("Attention rollout computation is only supported when attention head fusion is specified as 'mean', 'max', or 'min'.")
    
    # Get last stage data
    last_stage = max(hierarchical_attentions.keys())
    stage_data = hierarchical_attentions[last_stage]
    
    # Get raw last-block attention map and metadata
    blocks = _group_block_indices(stage_data['metadata'])
    if not blocks:
        logger.warning("No blocks found in last stage metadata")
        return
    last_block_idx = max(blocks)
    attn_map_raw, meta = _filter_maps_by_block(stage_data['attention_maps'], stage_data['metadata'], last_block_idx)
    shift = meta.get('shift_size', (0, 0))
    Hs, Ws = stage_data['stage_info']['input_resolution']
    
    # Save original image
    image_denorm = image.clone()
    mean = image_mean.to(image.device)
    std = image_std.to(image.device)
    image_denorm = image_denorm * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)
    
    input_image = Image.fromarray((image_denorm.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    input_image.save(os.path.join(image_output_dir, "original.png"))
    
    # Read original image for overlay visualizations
    original_image = imread(os.path.join(image_output_dir, "original.png"))
    if original_image.shape[2] == 4:  # Remove alpha channel if present
        original_image = original_image[:, :, :3]
    original_image_float = original_image.astype(np.float32) / 255.0
    
    # Determine which heads/maps to save based on head_fusion
    if head_fusion == "max":
        # Fuse using max
        fused_map = _stitch_block_attention(attn_map_raw, Hs, Ws, shift, "max", batch_size)[image_index]
        heads_to_save = ["max_fused"]
        selected_maps = [fused_map]
    elif head_fusion == "min":
        # Fuse using min
        fused_map = _stitch_block_attention(attn_map_raw, Hs, Ws, shift, "min", batch_size)[image_index]
        heads_to_save = ["min_fused"]
        selected_maps = [fused_map]
    elif head_fusion == "mean":
        # Fuse using mean
        fused_map = _stitch_block_attention(attn_map_raw, Hs, Ws, shift, "mean", batch_size)[image_index]
        heads_to_save = ["mean_fused"]
        selected_maps = [fused_map]
    elif isinstance(head_fusion, int):
        # Per-head visualization
        per_head_maps = _stitch_block_attention_per_head(attn_map_raw, Hs, Ws, shift, batch_size)[image_index]
        num_heads = per_head_maps.shape[0]
        k = min(int(head_fusion), num_heads)
        if k > num_heads:
            logger.warning(f"Number of heads to save ({k}) is greater than the number of heads ({num_heads}). Using all heads.")
        head_indices = random.sample(range(num_heads), min(k, num_heads))
        heads_to_save = head_indices
        selected_maps = [per_head_maps[h] for h in head_indices]
    else:
        raise ValueError(f"Head fusion type '{head_fusion}' not supported. Use 'mean', 'max', 'min' or an integer for random selection.")
    
    # Save attention heatmaps for selected heads/maps
    for map_idx, head_idx in enumerate(heads_to_save):
        attention_map = selected_maps[map_idx]
        
        # Resize to image size
        attention_resized = nn.functional.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        # Normalize for display
        attention_normalized = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
        
        # Save raw attention heatmap
        head_name = f"head_{head_idx}" if isinstance(head_idx, int) else head_idx
        fname = os.path.join(image_output_dir, f"attn_{head_name}.png")
        plt.imsave(fname=fname, arr=attention_normalized.cpu().numpy(), format='png', cmap='viridis', dpi=300)
        
        # Create masked visualization
        masked_attention = _apply_threshold_mask(attention_normalized, threshold)
        mask_fname = os.path.join(image_output_dir, f"masked_{threshold * 100:.0f}%_{head_name}.png")
        _display_instances(
            original_image,
            masked_attention.cpu().numpy(),
            fname=mask_fname,
            blur=False,
            figsize=(8, 8)
        )

        # Smooth colormap overlay
        overlay = smooth_attention_overlay(
            original_image_float, attention_normalized.cpu().numpy(), cmap="jet", alpha=0.5)
        overlay_fname = os.path.join(image_output_dir, f"overlay_{head_name}.png")
        plt.imsave(fname=overlay_fname, arr=overlay, format='png', dpi=300)
    
    # Compute attention rollout if requested
    if compute_rollout:
        logger.info(f"Computing Swin attention rollout with head_fusion={head_fusion}")
        fused_for_rollout = head_fusion if isinstance(head_fusion, str) else "mean"
        rollout_maps = compute_swin_rollout(
            hierarchical_attentions=hierarchical_attentions,
            discard_ratio=rollout_discard_ratio,
            head_fusion=fused_for_rollout,
            batch_size=batch_size,
        )  # [B, H_last, W_last]
        rollout_spatial = rollout_maps[image_index]
        
        # Interpolate to image size
        rollout_resized = nn.functional.interpolate(
            rollout_spatial.unsqueeze(0).unsqueeze(0),
            size=image.shape[-2:],
            mode='bilinear',
            align_corners=False
        )[0, 0]
        
        # Normalize to [0, 1] range
        rollout_resized = (rollout_resized - rollout_resized.min()) / (rollout_resized.max() - rollout_resized.min() + 1e-8)
        
        # Save rollout visualization
        rollout_fname = os.path.join(image_output_dir, f"rollout_{fused_for_rollout}.png")
        plt.imsave(fname=rollout_fname, arr=rollout_resized.cpu().numpy(), format='png', cmap='viridis', dpi=300)
        
        # Create masked rollout visualization
        rollout_masked = _apply_threshold_mask(rollout_resized, threshold)
        rollout_mask_fname = os.path.join(image_output_dir, f"rollout_masked_{threshold * 100:.0f}%_{fused_for_rollout}.png")
        
        _display_instances(
            original_image,
            rollout_masked.cpu().numpy(),
            fname=rollout_mask_fname,
            blur=False,
            figsize=(8, 8)
        )

        # Smooth rollout overlay
        rollout_overlay = smooth_attention_overlay(
            original_image_float, rollout_resized.cpu().numpy(), cmap="jet", alpha=0.5)
        rollout_overlay_fname = os.path.join(image_output_dir, f"rollout_overlay_{fused_for_rollout}.png")
        plt.imsave(fname=rollout_overlay_fname, arr=rollout_overlay, format='png', dpi=300)

def visualize_swin_attention_maps(
    hierarchical_attentions: Dict,
    images: torch.Tensor,
    image_ids: List[str],
    output_dir: str,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    head_fusion: Union[str, int] = "mean",
    compute_rollout: bool = False,
    rollout_discard_ratio: float = 0.9,
    threshold: float = 0.6,
):
    """
    Visualize Swin Transformer attention maps with hierarchical structure support.
    
    Args:
        hierarchical_attentions: Dictionary with stage-wise attention maps
        images: Input image tensor [B, C, H, W] or [B, 4, C, H, W]
        image_ids: List of image identifiers
        output_dir: Directory to save visualizations
        image_mean: Mean values for image denormalization
        image_std: Standard deviation values for image denormalization
        head_fusion: "mean" | "max" | "min" for fused multi-head attention; 
                      or int k to save k random per-head overlays
        compute_rollout: Whether to compute attention rollout
        rollout_discard_ratio: Ratio of lowest attention values to discard in rollout
        threshold: Threshold for attention masking (default: 0.6)
    """
    batch_size = images.shape[0]
    is_multi_view = len(images.shape) == 5 and images.shape[1] == 4
    
    logger.info(f"Processing Swin Transformer attention maps for {batch_size} images (multi_view={is_multi_view})...")
    
    # Get the last stage for final visualization
    last_stage = max(hierarchical_attentions.keys())
    last_stage_data = hierarchical_attentions[last_stage]
    
    if not last_stage_data['attention_maps']:
        logger.warning("No attention maps found in the last stage")
        return
    
    # Last stage input resolution
    if last_stage_data['stage_info'] and last_stage_data['stage_info']['input_resolution']:
        target_resolution = last_stage_data['stage_info']['input_resolution']
    else:
        raise AttributeError("last_stage_data is missing stage_info or input_resolution.")
    logger.info(f"Target resolution: {target_resolution}")
    
    # Process each image
    for idx in range(batch_size):
        image_output_dir = os.path.join(output_dir, f"attention_{image_ids[idx]}")
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Get image for this sample
        if is_multi_view:
            image = images[idx, 0]
        else:
            image = images[idx]
        
        # Process attention maps for this image
        try:
            _process_swin_attentions_per_image(
                hierarchical_attentions, image, idx, image_output_dir,
                image_mean, image_std, head_fusion, threshold,
                compute_rollout, batch_size, rollout_discard_ratio,
            )
            logger.info(f"Attention visualization completed for image {image_ids[idx]}")
        except Exception as e:
            logger.error(f"Failed to process attention for image {image_ids[idx]}: {e}")
        
        # Log progress every 10 images to avoid spam
        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info(f"Processed attention maps: {idx + 1}/{batch_size} images")
    
    logger.info(f"All Swin Transformer attention visualizations saved to {output_dir}")