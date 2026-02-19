import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.io import imread
from typing import Union, Optional
from torchvision.transforms import ToPILImage

from rad_dino.loggings.setup import init_logging
from rad_dino.utils.visualization.visualize_vit_attention import _display_instances, smooth_attention_overlay

init_logging()
logger = logging.getLogger(__name__)


def _fuse_heads(attn: torch.Tensor, head_fusion: str) -> torch.Tensor:
    """
    Fuse attention across heads.
    attn: [num_heads, N, N] -> returns [N, N]
    """
    if head_fusion == "mean":
        return attn.mean(dim=0)
    if head_fusion == "max":
        return attn.max(dim=0)[0]
    if head_fusion == "min":
        return attn.min(dim=0)[0]
    raise ValueError(f"Invalid head fusion type: {head_fusion}")


def _discard_low_values_per_row(a: torch.Tensor, discard_ratio: float) -> torch.Tensor:
    """
    Zero out the lowest discard_ratio fraction per row in the last dimension of a.
    a: [N, N]
    """
    if discard_ratio <= 0:
        return a
    N = a.shape[-1]
    k = int(N * discard_ratio)
    if k <= 0 or k >= N:
        return a
    vals, idx = torch.topk(a, k, dim=-1, largest=False)
    a = a.clone()
    a.scatter_(-1, idx, 0.0)
    return a


def _row_normalize_with_identity(a: torch.Tensor, identity_weight: float = 1.0) -> torch.Tensor:
    N = a.shape[-1]
    I = torch.eye(N, device=a.device, dtype=a.dtype)
    a = a + identity_weight * I
    denom = a.sum(dim=-1, keepdim=True)
    denom = torch.clamp(denom, min=1e-8)
    return a / denom


def _compute_siglip_rollout(attentions: torch.Tensor, discard_ratio: float = 0.9, head_fusion: str = "mean") -> torch.Tensor:
    """Compute attention rollout with pooler weights."""
    if len(attentions.shape) != 4:
        raise ValueError(f"Expected attention tensor with 4 dimensions (num_layers, num_heads, N, N), got {attentions.shape}")
    device = attentions.device
    num_layers, _, N1, N2 = attentions.shape
    if N1 != N2:
        raise ValueError(f"Attention must be square, got {attentions.shape}")

    result = torch.eye(N1, device=device)
    with torch.no_grad():
        for layer_idx in range(num_layers):
            A = _fuse_heads(attentions[layer_idx], head_fusion)
            A = _discard_low_values_per_row(A, discard_ratio)
            A = _row_normalize_with_identity(A, identity_weight=1.0)
            result = A @ result
    return result


def _apply_threshold_mask(attention_map: torch.Tensor, threshold: float) -> torch.Tensor:
    flat = attention_map.flatten()
    flat = flat / (flat.sum() + 1e-8)
    val, sort_idx = torch.sort(flat)
    cumval = torch.cumsum(val, dim=0)
    masked_flat = cumval > (1 - threshold)
    sort_idx_inv = torch.argsort(sort_idx)
    masked_flat = masked_flat[sort_idx_inv]
    return masked_flat.reshape(attention_map.shape).float()


def _process_attentions_per_image(
    attention: torch.Tensor,
    image: torch.Tensor,
    image_output_dir: str,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    w_featmap: int,
    h_featmap: int,
    patch_size: int,
    threshold: float,
    head_fusion: Union[str, int],
    compute_rollout: bool,
    all_layer_attentions: Union[torch.Tensor, None] = None,
    rollout_discard_ratio: float = 0.9,
    pooler_attn: Optional[torch.Tensor] = None,
) -> None:
    """Process SigLIP attention maps for a single image/view (pooler-centric only)"""
    if pooler_attn is None:
        raise ValueError("SigLip attention visualization requires pooler_attn_for_view (per-head pooler weights).")

    # Save original image
    image_denorm = image * image_std.to(image.device) + image_mean.to(image.device)
    image_denorm = torch.clamp(image_denorm, 0, 1)
    ToPILImage()(image_denorm).save(os.path.join(image_output_dir, "original.png"))
    original_image = imread(os.path.join(image_output_dir, "original.png"))
    if original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]

    # Determine heads/maps to save
    selected_maps: list[torch.Tensor] = []
    heads_to_save: list[Union[int, str]] = []

    # pooler_attn: [H, N]
    num_pooler_heads = pooler_attn.shape[0]
    if head_fusion in ("mean", "max", "min"):
        if head_fusion == "mean":
            fused_pooler_attn = pooler_attn.mean(dim=0)
        elif head_fusion == "max":
            fused_pooler_attn = pooler_attn.max(dim=0)[0]
        else:
            fused_pooler_attn = pooler_attn.min(dim=0)[0]
        fused_pooler_attn_maps = fused_pooler_attn.reshape(w_featmap, h_featmap)
        fused_pooler_attn_maps_upsampled = nn.functional.interpolate(
            fused_pooler_attn_maps.unsqueeze(0).unsqueeze(0).float(),
            scale_factor=patch_size, mode="bilinear", align_corners=False)[0, 0]
        selected_maps = [fused_pooler_attn_maps_upsampled]
        heads_to_save = [f"{head_fusion}_fused"]
    elif isinstance(head_fusion, int):
        k = min(int(head_fusion), num_pooler_heads)
        if k > num_pooler_heads:
            logger.warning(f"Number of heads to save ({k}) exceeds number of heads ({num_pooler_heads}). Using all heads.")
        head_indices = random.sample(range(num_pooler_heads), k)
        for h in head_indices:
            attn_head_maps = pooler_attn[h].reshape(w_featmap, h_featmap)
            attn_head_maps_upsampled = nn.functional.interpolate(
                attn_head_maps.unsqueeze(0).unsqueeze(0).float(),
                scale_factor=patch_size, mode="bilinear", align_corners=False)[0, 0]
            selected_maps.append(attn_head_maps_upsampled)
        heads_to_save = head_indices
    else:
        raise ValueError(f"Invalid head_fusion: {head_fusion}")

    original_image_float = original_image.astype(np.float32) / 255.0

    # Save maps
    for map_idx, head_id in enumerate(heads_to_save):
        attention_map = selected_maps[map_idx]
        # Normalize for display
        att_t = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        # Save raw heatmap
        head_name = f"head_{head_id}" if isinstance(head_id, int) else head_id
        fname = os.path.join(image_output_dir, f"attn_{head_name}.png")
        plt.imsave(fname=fname, arr=att_t.detach().cpu().numpy(), format='png', cmap='viridis', dpi=300)
        # Masked overlay
        masked = _apply_threshold_mask(att_t, threshold)
        mask_fname = os.path.join(image_output_dir, f"masked_{threshold * 100:.0f}%_{head_name}.png")
        _display_instances(original_image, masked.cpu().numpy(), fname=mask_fname, blur=False, figsize=(8, 8))
        # Smooth colormap overlay
        overlay = smooth_attention_overlay(original_image_float, att_t.detach().cpu().numpy(), cmap="jet", alpha=0.5)
        overlay_fname = os.path.join(image_output_dir, f"overlay_{head_name}.png")
        plt.imsave(fname=overlay_fname, arr=overlay, format='png', dpi=300)

    # Rollout 
    if compute_rollout and all_layer_attentions is not None:
        # Compute rollout across encoder heads
        if not isinstance(head_fusion, str) or head_fusion not in ("mean", "max", "min"):
            raise ValueError("When computing rollout, head_fusion must be 'mean', 'max', or 'min'")
        rollout = _compute_siglip_rollout(all_layer_attentions, discard_ratio=rollout_discard_ratio, head_fusion=head_fusion)

        # fused_pooler_attn: [N]; rollout: [N,N] → pooled_rollout: [N]
        pooled_rollout = (fused_pooler_attn @ rollout).reshape(w_featmap, h_featmap)
        pooled_rollout = (pooled_rollout - pooled_rollout.min()) / (pooled_rollout.max() - pooled_rollout.min() + 1e-8)
        pooled_rollout_upsampled = nn.functional.interpolate(
            pooled_rollout.unsqueeze(0).unsqueeze(0).float(),
            scale_factor=patch_size, mode="bilinear", align_corners=False)[0, 0]
        rollout_fname = os.path.join(image_output_dir, f"rollout_{head_fusion}.png")
        plt.imsave(fname=rollout_fname, arr=pooled_rollout_upsampled.detach().cpu().numpy(), format='png', cmap='viridis', dpi=300)
        rollout_masked = _apply_threshold_mask(pooled_rollout_upsampled, threshold)
        rollout_mask_fname = os.path.join(image_output_dir, f"rollout_masked_{threshold * 100:.0f}%_{head_fusion}.png")
        _display_instances(original_image, rollout_masked.cpu().numpy(), fname=rollout_mask_fname, blur=False, figsize=(8, 8))
        # Smooth rollout overlay
        rollout_overlay = smooth_attention_overlay(
            original_image_float, pooled_rollout_upsampled.detach().cpu().numpy(), cmap="jet", alpha=0.5)
        rollout_overlay_fname = os.path.join(image_output_dir, f"rollout_overlay_{head_fusion}.png")
        plt.imsave(fname=rollout_overlay_fname, arr=rollout_overlay, format='png', dpi=300)


def visualize_siglip_attention_maps(
    attentions: torch.Tensor,
    images: torch.Tensor,
    image_ids,
    output_dir: str,
    accelerator,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    patch_size: int = 14,
    threshold: float = 0.6,
    head_fusion: Union[str, int] = "mean",
    compute_rollout: bool = False,
    rollout_discard_ratio: float = 0.9,
    pooler_attn_weights: Optional[torch.Tensor] = None,
):
    """
    Visualize attention maps for SigLIP vision encoder (no CLS token, MAP pooling).
    
    attentions: [L, B, H, N, N] or [L, B, 4, H, N, N] for multi-view
    images: [B, C, H, W] or [B, 4, C, H, W]
    """
    if not accelerator.is_main_process:
        return

    batch_size = images.shape[0]
    is_multi_view = len(images.shape) == 5 and images.shape[1] == 4

    logger.info(f"Processing SigLIP attention maps for {batch_size} images (multi_view={is_multi_view})...")

    last_attention = attentions[-1]
    if is_multi_view:
        num_views = last_attention.shape[1]
        view_names = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
    else:
        num_views = 1
        view_names = ['single']

    # For SigLIP: seq_len equals number of patches (no CLS)
    seq_len = last_attention.shape[-1]
    patch_dim = int(np.sqrt(seq_len))
    w_featmap = images.shape[-2] // patch_size
    h_featmap = images.shape[-1] // patch_size
    logger.info(f"siglip patch_dim: {patch_dim}, seq_len: {seq_len}")
    assert patch_dim == w_featmap == h_featmap, f"NotEqualError: patch_dim: {patch_dim}, width_featmap: {w_featmap}, height_featmap: {h_featmap}"

    for idx in range(batch_size):
        image_id = image_ids[idx]
        if is_multi_view:
            for view_idx in range(num_views):
                attn = last_attention[idx, view_idx]
                img = images[idx, view_idx]
                view_name = view_names[view_idx]
                image_output_dir = os.path.join(output_dir, f"attention_{image_id}_{view_name}")
                os.makedirs(image_output_dir, exist_ok=True)
                all_layer_attns = attentions[:, idx, view_idx] if compute_rollout else None
                pooler_view = None
                if pooler_attn_weights is not None:
                    # pooler_attn_weights expected [B, V, H, N]
                    pooler_view = pooler_attn_weights[idx, view_idx]
                _process_attentions_per_image(
                    attn, img, image_output_dir, image_mean, image_std,
                    w_featmap, h_featmap, patch_size, threshold, head_fusion,
                    compute_rollout, all_layer_attentions=all_layer_attns, rollout_discard_ratio=rollout_discard_ratio,
                    pooler_attn=pooler_view
                )
        else:
            attn = last_attention[idx]
            img = images[idx]
            image_output_dir = os.path.join(output_dir, f"attention_{image_id}")
            os.makedirs(image_output_dir, exist_ok=True)
            all_layer_attns = attentions[:, idx] if compute_rollout else None
            pooler_view = None
            if pooler_attn_weights is not None:
                # pooler_attn_weights expected [B, H, N]
                pooler_view = pooler_attn_weights[idx]
            _process_attentions_per_image(
                attn, img, image_output_dir, image_mean, image_std,
                w_featmap, h_featmap, patch_size, threshold, head_fusion,
                compute_rollout, all_layer_attentions=all_layer_attns, rollout_discard_ratio=rollout_discard_ratio,
                pooler_attn=pooler_view
            )

        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info(f"Processed SigLIP attention maps: {idx + 1}/{batch_size} images")

    logger.info(f"SigLIP attention visualization completed for all {batch_size} images. Results saved in {output_dir}")



