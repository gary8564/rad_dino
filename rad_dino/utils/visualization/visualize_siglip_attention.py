import os
import random
import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.io import imread
from torchvision.transforms import ToPILImage

from rad_dino.loggings.setup import init_logging
from rad_dino.utils.visualization.visualize_vit_attention import _display_instances

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
    """
    Attention rollout for models WITHOUT CLS token (e.g., SigLIP vision).
    Returns vector of length N (num_patches) representing per-patch importance, derived by
    averaging over all query positions.
    
    attentions: [num_layers, num_heads, N, N]
    """
    if len(attentions.shape) != 4:
        raise ValueError(f"Expected attention tensor with 4 dimensions (num_layers, num_heads, N, N), got {attentions.shape}")
    device = attentions.device
    num_layers, _, N, N2 = attentions.shape
    if N != N2:
        raise ValueError(f"Attention must be square, got {attentions.shape}")

    result = torch.eye(N, device=device)
    with torch.no_grad():
        for layer_idx in range(num_layers):
            A = _fuse_heads(attentions[layer_idx], head_fusion)
            A = _discard_low_values_per_row(A, discard_ratio)
            A = _row_normalize_with_identity(A, identity_weight=1.0)
            result = A @ result

    # Average over all queries to get per-key importance  
    # For SigLIP, we average over rows (queries) to get importance per key (patch)
    mask = result.mean(dim=0)  # [N]
    if torch.isnan(mask).any():
        raise RuntimeError("NaN detected in rollout mask for SigLIP.")
    if torch.allclose(mask, torch.zeros_like(mask)):
        raise RuntimeError("Rollout result is all zeros for SigLIP.")
    return mask


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
) -> None:
    """Process SigLIP attention maps for a single image/view (no CLS token)."""
    if compute_rollout and isinstance(head_fusion, int):
        raise ValueError("Attention rollout computation is only supported when head_fusion is 'mean'|'max'|'min'.")

    num_heads = attention.shape[0]

    # Per-head per-key map by averaging over all queries (no CLS)
    per_key = attention.mean(dim=1)  # [num_heads, N]
    per_key = per_key.reshape(num_heads, w_featmap, h_featmap)

    # Upsample to image grid for visualization
    per_key_up = nn.functional.interpolate(per_key.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0]
    per_key_np = per_key_up.detach().cpu().numpy()

    # Save original image
    image_denorm = image * image_std.to(image.device) + image_mean.to(image.device)
    image_denorm = torch.clamp(image_denorm, 0, 1)
    ToPILImage()(image_denorm).save(os.path.join(image_output_dir, "original.png"))
    original_image = imread(os.path.join(image_output_dir, "original.png"))
    if original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]

    # Determine heads/maps to save
    if head_fusion in ("mean", "max", "min"):
        if head_fusion == "mean":
            fused = np.mean(per_key_np, axis=0)
        elif head_fusion == "max":
            fused = np.max(per_key_np, axis=0)
        else:
            fused = np.min(per_key_np, axis=0)
        heads_to_save = [f"{head_fusion}_fused"]
        selected_maps = [fused]
    elif isinstance(head_fusion, int):
        k = min(int(head_fusion), num_heads)
        if k > num_heads:
            logger.warning(f"Number of heads to save ({k}) exceeds number of heads ({num_heads}). Using all heads.")
        head_indices = random.sample(range(num_heads), k)
        heads_to_save = head_indices
        selected_maps = [per_key_np[h] for h in head_indices]
    else:
        raise ValueError(f"Invalid head_fusion: {head_fusion}")

    # Save maps
    for map_idx, head_id in enumerate(heads_to_save):
        attention_map = selected_maps[map_idx]
        # Normalize for display
        att_t = torch.from_numpy(attention_map)
        att_t = (att_t - att_t.min()) / (att_t.max() - att_t.min() + 1e-8)
        # Save raw heatmap
        head_name = f"head_{head_id}" if isinstance(head_id, int) else head_id
        fname = os.path.join(image_output_dir, f"attn_{head_name}.png")
        plt.imsave(fname=fname, arr=att_t.cpu().numpy(), format='png', cmap='viridis', dpi=300)
        # Masked overlay
        masked = _apply_threshold_mask(att_t, threshold)
        mask_fname = os.path.join(image_output_dir, f"masked_{threshold * 100:.0f}%_{head_name}.png")
        _display_instances(original_image, masked.cpu().numpy(), fname=mask_fname, blur=False, figsize=(8, 8))

    # Rollout
    if compute_rollout and all_layer_attentions is not None:
        logger.info(f"Computing SigLIP attention rollout with {all_layer_attentions.shape[0]} layers and head_fusion={head_fusion}")
        rollout_vec = _compute_siglip_rollout(all_layer_attentions, discard_ratio=rollout_discard_ratio, head_fusion=head_fusion if isinstance(head_fusion, str) else "mean")
        width = int(rollout_vec.size(-1) ** 0.5)
        if width != w_featmap or width != h_featmap:
            raise ValueError(f"NotEqualError: width of rollout_mask: {width}, width_featmap: {w_featmap}, height_featmap: {h_featmap}")
        rollout_spatial = rollout_vec.reshape(width, width).unsqueeze(0).unsqueeze(0)
        rollout_up = nn.functional.interpolate(rollout_spatial, scale_factor=patch_size, mode="nearest")[0, 0]
        rollout_up = (rollout_up - rollout_up.min()) / (rollout_up.max() - rollout_up.min() + 1e-8)

        rollout_fname = os.path.join(image_output_dir, f"rollout_{head_fusion if isinstance(head_fusion, str) else 'mean'}.png")
        plt.imsave(fname=rollout_fname, arr=rollout_up.cpu().numpy(), format='png', cmap='viridis', dpi=300)

        rollout_masked = _apply_threshold_mask(rollout_up, threshold)
        rollout_mask_fname = os.path.join(image_output_dir, f"rollout_masked_{threshold * 100:.0f}%_{head_fusion if isinstance(head_fusion, str) else 'mean'}.png")
        _display_instances(original_image, rollout_masked.cpu().numpy(), fname=rollout_mask_fname, blur=False, figsize=(8, 8))


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
                _process_attentions_per_image(
                    attn, img, image_output_dir, image_mean, image_std,
                    w_featmap, h_featmap, patch_size, threshold, head_fusion,
                    compute_rollout, all_layer_attentions=all_layer_attns, rollout_discard_ratio=rollout_discard_ratio,
                )
        else:
            attn = last_attention[idx]
            img = images[idx]
            image_output_dir = os.path.join(output_dir, f"attention_{image_id}")
            os.makedirs(image_output_dir, exist_ok=True)
            all_layer_attns = attentions[:, idx] if compute_rollout else None
            _process_attentions_per_image(
                attn, img, image_output_dir, image_mean, image_std,
                w_featmap, h_featmap, patch_size, threshold, head_fusion,
                compute_rollout, all_layer_attentions=all_layer_attns, rollout_discard_ratio=rollout_discard_ratio,
            )

        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info(f"Processed SigLIP attention maps: {idx + 1}/{batch_size} images")

    logger.info(f"SigLIP attention visualization completed for all {batch_size} images. Results saved in {output_dir}")



