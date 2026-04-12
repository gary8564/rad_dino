import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread
from typing import Union, Optional
from torchvision.transforms import ToPILImage

from rad_dino.loggings.setup import init_logging
from rad_dino.models.siglip import (
    patch_siglip_encoder_eager,
    unpatch_siglip_encoder,
)
from rad_dino.utils.visualization.visualize_vit_attention import (
    _display_instances, _smooth_attention_overlay, _gradient_rollout,
    _normalize_heatmap, _save_gradient_rollout_visualizations,
    _grad_rollout_class_suffix,
)

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


def _compute_siglip_rollout(
    attentions: torch.Tensor,
    discard_ratio: float = 0.9,
    head_fusion: str = "mean",
    last_n_layers: int = 6,
) -> torch.Tensor:
    """Compute attention rollout for SigLIP encoder (patch-to-patch, no CLS).

    Because SigLIP has no CLS token, the full N-by-N rollout matrix is used
    (combined with pooler attention externally).  Products of stochastic
    matrices converge to a rank-1 (uniform) matrix, so using all 27 layers
    typically erases spatial information.  ``last_n_layers`` limits rollout
    to the final K encoder layers, which preserves meaningful variation.

    Args:
        attentions: [num_layers, num_heads, N, N]
        discard_ratio: Fraction of lowest attention values zeroed per row.
        head_fusion: How to fuse heads ("mean", "max", "min").
            last_n_layers: Only use the last K encoder layers for rollout. By default, use the last 6 layers.
    """
    if len(attentions.shape) != 4:
        raise ValueError(f"Expected attention tensor with 4 dimensions (num_layers, num_heads, N, N), got {attentions.shape}")
    device = attentions.device
    num_layers, _, N1, N2 = attentions.shape
    if N1 != N2:
        raise ValueError(f"Attention must be square, got {attentions.shape}")

    start_layer = 0
    if last_n_layers > num_layers:
        raise ValueError(f"last_n_layers ({last_n_layers}) is greater than the number of layers ({num_layers}).")
    start_layer = num_layers - last_n_layers
    logger.info(
        f"SigLIP rollout: using layers {start_layer}..{num_layers - 1} "
        f"(last {last_n_layers} of {num_layers}) to avoid convergence to uniform."
    )

    result = torch.eye(N1, device=device)
    with torch.no_grad():
        for layer_idx in range(start_layer, num_layers):
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
    rollout_last_n_layers: int = 6,
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
        overlay = _smooth_attention_overlay(original_image_float, att_t.detach().cpu().numpy())
        overlay_fname = os.path.join(image_output_dir, f"overlay_{head_name}.png")
        plt.imsave(fname=overlay_fname, arr=overlay, format='png', dpi=300)

    # Rollout 
    if compute_rollout and all_layer_attentions is not None:
        # Compute rollout across encoder heads
        if not isinstance(head_fusion, str) or head_fusion not in ("mean", "max", "min"):
            raise ValueError("When computing rollout, head_fusion must be 'mean', 'max', or 'min'")
        rollout = _compute_siglip_rollout(
            all_layer_attentions, discard_ratio=rollout_discard_ratio,
            head_fusion=head_fusion, last_n_layers=rollout_last_n_layers,
        )

        # fused_pooler_attn: [N]; rollout: [N,N] → pooled_rollout: [N]
        fused_pooler_attn = fused_pooler_attn.to(rollout.device)
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
        rollout_overlay = _smooth_attention_overlay(
            original_image_float, pooled_rollout_upsampled.detach().cpu().numpy())
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
    rollout_last_n_layers: int = 6,
):
    """
    Visualize attention maps for SigLIP vision encoder (no CLS token, MAP pooling).
    
    attentions: [L, B, H, N, N] or [L, B, 4, H, N, N] for multi-view
    images: [B, C, H, W] or [B, 4, C, H, W]
    """
    if not accelerator.is_main_process:
        return

    batch_size = images.shape[0]
    is_multi_view = len(images.shape) == 5 and images.shape[1] > 1

    logger.info(f"Processing SigLIP attention maps for {batch_size} images (multi_view={is_multi_view})...")

    last_attention = attentions[-1]
    if is_multi_view:
        num_views = last_attention.shape[1]
        view_names = (
            ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']
            if num_views == 4
            else [f"view_{i}" for i in range(num_views)]
        )
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
                    pooler_attn=pooler_view, rollout_last_n_layers=rollout_last_n_layers,
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
                pooler_attn=pooler_view, rollout_last_n_layers=rollout_last_n_layers,
            )

        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info(f"Processed SigLIP attention maps: {idx + 1}/{batch_size} images")

    logger.info(f"SigLIP attention visualization completed for all {batch_size} images. Results saved in {output_dir}")


# Gradient rollout for SigLIP
def compute_siglip_gradient_rollout(
    model,
    input_tensor,
    image_mean,
    image_std,
    category_index=None,
    discard_ratio=0.9,
    patch_size=14,
    output_dir=None,
    image_id=None,
):
    """
    End-to-end gradient attention rollout for a SigLIP/MedSigLIP model.

    SigLIP has no CLS token.  Information is aggregated by a MAP (Multi-head
    Attention Pooling) head.  Gradient rollout therefore combines:

    1. Gradient-weighted encoder rollout matrix ``R`` ([N, N])
    2. Gradient-weighted MAP pooler attention ``p`` ([N])

    into ``mask = p @ R`` to get per-patch importance.

    Args:
        model: A MedSigClassifier.
        input_tensor: [1, C, H, W].
        image_mean, image_std: For denormalisation.
        category_index: Target class (``None`` → predicted).
        discard_ratio: Fraction to discard in encoder rollout.
        patch_size: SigLIP patch size.
        output_dir: Save directory (``None`` → skip).
        image_id: Sample id for filenames.

    Returns:
        ``(rollout_map_np, category_index)``.
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device).requires_grad_(True)
    model.eval()

    # Monkey-patch encoder attention to eager mode so we can capture
    # per-layer attention weight tensors (SDPA silently returns None).
    attn_storage: list[torch.Tensor] = []
    encoder_patches = patch_siglip_encoder_eager(
        model.backbone.vision_model.encoder,
        storage=attn_storage,
        retain_grad=True,
    )

    try:
        vm = model.backbone.vision_model
        # 1. Patch embedding
        hidden_states = vm.embeddings(input_tensor)
        # 2. Encoder 
        encoder_outputs = vm.encoder(
            inputs_embeds=hidden_states,
            return_dict=True,
        )
        last_hidden_state = vm.post_layernorm(encoder_outputs.last_hidden_state)
        # 3. MAP pooler head
        head = vm.head
        mha = head.attention # nn.MultiheadAttention (batch_first=True)
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        head_dim = embed_dim // num_heads
        probe = head.probe.repeat(last_hidden_state.shape[0], 1, 1)  # [B, 1, D]
        # Project Q from probe, K/V from encoder hidden states
        qkv_same_weight = mha.in_proj_weight          # [3*D, D]
        qkv_same_bias = mha.in_proj_bias              # [3*D]
        q = F.linear(probe, qkv_same_weight[:embed_dim], qkv_same_bias[:embed_dim])
        k = F.linear(last_hidden_state, qkv_same_weight[embed_dim:2*embed_dim], qkv_same_bias[embed_dim:2*embed_dim])
        v = F.linear(last_hidden_state, qkv_same_weight[2*embed_dim:], qkv_same_bias[2*embed_dim:])
        bsz = probe.shape[0]
        q = q.view(bsz, 1, num_heads, head_dim).transpose(1, 2)           # [B, H, 1, d]
        k = k.view(bsz, -1, num_heads, head_dim).transpose(1, 2)          # [B, H, N, d]
        v = v.view(bsz, -1, num_heads, head_dim).transpose(1, 2)          # [B, H, N, d]
        pooler_attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        pooler_attn_weights = F.softmax(pooler_attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        pooler_attn_weights.retain_grad()                                  # [B, H, 1, N]
        attn_output = torch.matmul(pooler_attn_weights, v)                 # [B, H, 1, d]
        attn_output = attn_output.transpose(1, 2).reshape(bsz, 1, embed_dim)
        attn_output = mha.out_proj(attn_output)                            # [B, 1, D]

        residual = attn_output
        pooler_hidden = head.layernorm(attn_output)
        pooler_hidden = residual + head.mlp(pooler_hidden)
        pooler_output = pooler_hidden[:, 0]
        # 4. Classifier
        features = pooler_output / pooler_output.norm(dim=-1, keepdim=True)
        logits = model.classifier(features)
        num_classes = logits.shape[-1]
        if category_index is None:
            category_index = logits.argmax(dim=-1).item()
        # 5. Backward
        model.zero_grad()
        target = torch.zeros_like(logits)
        target[:, category_index] = 1.0
        (logits * target).sum().backward()
        # 6. Collect encoder attention and gradients from storage
        enc_attn_list = [a[0].detach().cpu() for a in attn_storage]
        enc_grad_list = [a.grad[0].detach().cpu() for a in attn_storage]
        # Gradient-weighted MAP pooler attention
        # pooler_attn_weights: [B, num_heads, 1, N]
        pooler_a = pooler_attn_weights[0, :, 0, :].detach().cpu()
        pooler_g = pooler_attn_weights.grad[0, :, 0, :].detach().cpu()

    finally:
        unpatch_siglip_encoder(encoder_patches)

    rollout_matrix = _gradient_rollout(
        enc_attn_list, enc_grad_list,
        discard_ratio=discard_ratio,
        num_prefix_tokens=0,
        return_full_matrix=True,
    )

    weighted_pooler = (pooler_a * pooler_g).mean(dim=0).clamp(min=0)
    if weighted_pooler.sum() > 0:
        weighted_pooler = weighted_pooler / weighted_pooler.sum()

    # pooler @ encoder rollout
    mask = (weighted_pooler @ rollout_matrix).float()
    if mask.max() > 0:
        mask = mask / mask.max()

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

