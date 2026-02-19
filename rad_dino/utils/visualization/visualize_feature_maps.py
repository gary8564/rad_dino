"""
Stage-wise Feature Map Visualization.

Visualizes intermediate feature activations at each stage of hierarchical
vision transformer backbones (DaViT, Swin Transformer). For each stage,
a spatial heatmap is produced by computing the L2 norm across channels
at every spatial position, then resize and overlay on the original image
using a smooth colormap.

Optionally, individual random feature channels can be visualized to show
what specific channels attend to.
"""

import os
import random
import logging
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from torchvision.transforms import ToPILImage
from skimage.io import imread

from rad_dino.utils.visualization.visualize_vit_attention import smooth_attention_overlay

logger = logging.getLogger(__name__)


def _features_to_spatial_map(
    features: torch.Tensor,
    spatial_size: tuple,
    mode: str = "l2_norm",
    channel_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert a feature tensor to a single-channel spatial activation map.

    Args:
        features: [N, C] flattened features for one image at one stage.
        spatial_size: (H, W) spatial dims of the feature grid.
        mode: "l2_norm" | "mean_abs" | "max" | "channel".
            - l2_norm: L2 norm across channel dim at each position.
            - mean_abs: Mean absolute value across channels.
            - max: Max absolute value across channels.
            - channel: Select a single channel (requires channel_idx).
        channel_idx: Required when mode="channel".

    Returns:
        [H, W] spatial activation map.
    """
    H, W = spatial_size
    N, C = features.shape
    assert N == H * W, f"Feature count {N} != H*W={H*W}"

    feat_2d = features.view(H, W, C)

    if mode == "l2_norm":
        return feat_2d.norm(dim=-1)
    elif mode == "mean_abs":
        return feat_2d.abs().mean(dim=-1)
    elif mode == "max":
        return feat_2d.abs().max(dim=-1)[0]
    elif mode == "channel":
        if channel_idx is None:
            raise ValueError("channel_idx required for mode='channel'")
        return feat_2d[:, :, channel_idx].abs()
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _normalize_map(spatial_map: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a spatial map to [0, 1]."""
    vmin = spatial_map.min()
    vmax = spatial_map.max()
    return (spatial_map - vmin) / (vmax - vmin + 1e-8)


def _resize_to_image(spatial_map: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
    """Bilinearly resize [H, W] map to (img_h, img_w)."""
    return F.interpolate(
        spatial_map.unsqueeze(0).unsqueeze(0),
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _process_feature_maps_per_image(
    stage_features: Dict[int, Dict],
    image: torch.Tensor,
    image_output_dir: str,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    num_random_channels: int = 3,
    activation_mode: str = "l2_norm",
):
    """
    Process and save feature map visualizations for one image across all stages.

    Args:
        stage_features: {stage_idx: {"features": [N, C], "spatial_size": (H, W), "embed_dim": int}}
        image: [C, H, W] normalized image tensor.
        image_output_dir: Directory to save outputs.
        image_mean, image_std: For denormalization.
        num_random_channels: How many random channels to visualize per stage.
        activation_mode: Default aggregation mode for the summary map.
    """
    image_denorm = image * image_std.to(image.device) + image_mean.to(image.device)
    image_denorm = torch.clamp(image_denorm, 0, 1)
    ToPILImage()(image_denorm).save(os.path.join(image_output_dir, "original.png"))
    original_image = imread(os.path.join(image_output_dir, "original.png"))
    if original_image.ndim == 3 and original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]
    original_image_float = original_image.astype(np.float32) / 255.0

    img_h, img_w = image.shape[-2:]

    for stage_idx in sorted(stage_features.keys()):
        info = stage_features[stage_idx]
        features = info["features"]       # [N, C]
        spatial_size = info["spatial_size"]  # (H, W)
        embed_dim = info["embed_dim"]

        stage_dir = os.path.join(image_output_dir, f"stage_{stage_idx}")
        os.makedirs(stage_dir, exist_ok=True)

        # Aggregated activation map (L2 norm by default)
        agg_map = _features_to_spatial_map(features, spatial_size, mode=activation_mode)
        agg_resized = _resize_to_image(agg_map, img_h, img_w)
        agg_norm = _normalize_map(agg_resized)

        agg_np = agg_norm.cpu().numpy()
        plt.imsave(
            os.path.join(stage_dir, f"{activation_mode}.png"),
            agg_np, format="png", cmap="viridis", dpi=300,
        )

        overlay = smooth_attention_overlay(original_image_float, agg_np, cmap="jet", alpha=0.5)
        plt.imsave(
            os.path.join(stage_dir, f"{activation_mode}_overlay.png"),
            overlay, format="png", dpi=300,
        )

        # Random channel visualizations
        k = min(num_random_channels, embed_dim)
        channel_indices = random.sample(range(embed_dim), k)

        for ch_idx in channel_indices:
            ch_map = _features_to_spatial_map(
                features, spatial_size, mode="channel", channel_idx=ch_idx,
            )
            ch_resized = _resize_to_image(ch_map, img_h, img_w)
            ch_norm = _normalize_map(ch_resized)

            ch_np = ch_norm.cpu().numpy()
            plt.imsave(
                os.path.join(stage_dir, f"channel_{ch_idx}.png"),
                ch_np, format="png", cmap="viridis", dpi=300,
            )

            overlay = smooth_attention_overlay(
                original_image_float, ch_np, cmap="jet", alpha=0.5,
            )
            plt.imsave(
                os.path.join(stage_dir, f"channel_{ch_idx}_overlay.png"),
                overlay, format="png", dpi=300,
            )

        logger.debug(
            "Stage %d: spatial_size=%s, embed_dim=%d, saved %d channel maps",
            stage_idx, spatial_size, embed_dim, k,
        )


def visualize_stage_feature_maps(
    stage_features: Dict[int, Dict],
    images: torch.Tensor,
    image_ids: List[str],
    output_dir: str,
    image_mean: torch.Tensor,
    image_std: torch.Tensor,
    num_random_channels: int = 3,
    activation_mode: str = "l2_norm",
):
    """
    Visualize stage-wise feature maps for a batch of images.

    Produces, for each image and each stage:
      - An aggregated spatial heatmap (L2 norm across channels by default).
      - A jet-colormap overlay on the original image.
      - ``num_random_channels`` individual channel activation maps + overlays.

    Args:
        stage_features: {stage_idx: {"features": [B, N, C], "spatial_size": (H, W), "embed_dim": C}}
            One entry per backbone stage.
        images: [B, C, H, W] batch of normalised images.
        image_ids: Per-image identifiers for naming output files.
        output_dir: Root output directory.
        image_mean, image_std: Denormalization constants (shape [3,1,1]).
        num_random_channels: Random channels to visualize per stage.
        activation_mode: Aggregation mode ("l2_norm", "mean_abs", "max").
    """
    batch_size = images.shape[0]
    num_stages = len(stage_features)
    logger.info(
        "Visualizing feature maps: %d stages, %d images, mode=%s",
        num_stages, batch_size, activation_mode,
    )

    for idx in range(batch_size):
        image_dir = os.path.join(output_dir, f"feature_maps_{image_ids[idx]}")
        os.makedirs(image_dir, exist_ok=True)

        per_image_features = {}
        for stage_idx, info in stage_features.items():
            per_image_features[stage_idx] = {
                "features": info["features"][idx],        # [N, C]
                "spatial_size": info["spatial_size"],
                "embed_dim": info["embed_dim"],
            }

        try:
            _process_feature_maps_per_image(
                stage_features=per_image_features,
                image=images[idx],
                image_output_dir=image_dir,
                image_mean=image_mean,
                image_std=image_std,
                num_random_channels=num_random_channels,
                activation_mode=activation_mode,
            )
        except Exception as e:
            logger.error("Failed to process feature maps for %s: %s", image_ids[idx], e)

        if (idx + 1) % 10 == 0 or idx == batch_size - 1:
            logger.info("Feature map visualization: %d/%d", idx + 1, batch_size)

    logger.info("Feature map visualizations saved to %s", output_dir)
