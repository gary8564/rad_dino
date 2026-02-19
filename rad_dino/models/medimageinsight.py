import os
import sys
import torch
import logging
from typing import Optional, Dict

from rad_dino.models.base import BaseClassifier
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def load_medimageinsight_model(model_dir: str, device: str = "cuda"):
    """
    Load the MedImageInsight UniCL model from a locally cloned lion-ai/MedImageInsights repo.

    The function adds ``model_dir`` to ``sys.path`` so that the package-internal
    imports inside the ``MedImageInsight`` package resolve correctly, then builds
    the full UniCL model (image encoder + language encoder + projections) and
    loads the pre-trained weights.

    Args:
        model_dir: Absolute path to the cloned ``lion-ai/MedImageInsights`` hugging-face repository
        device: Target device, i.e., "cuda" or "cpu".

    Returns:
        The loaded UniCLModel (nn.Module) moved to device.
    """
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from MedImageInsight.UniCLModel import build_unicl_model
    from MedImageInsight.Utils.Arguments import load_opt_from_config_files

    config_path = os.path.join(model_dir, "2024.09.27", "config.yaml")
    config = load_opt_from_config_files([config_path])

    config["UNICL_MODEL"]["PRETRAINED"] = os.path.join(
        model_dir, "2024.09.27", "vision_model", "medimageinsigt-v1.0.0.pt"
    )
    config["LANG_ENCODER"]["PRETRAINED_TOKENIZER"] = os.path.join(
        model_dir, "2024.09.27", "language_model", "clip_tokenizer_4.16.2"
    )

    unicl_model = build_unicl_model(config)
    unicl_model.to(device)
    logger.info(
        "Loaded MedImageInsight UniCL model from %s (device=%s)", model_dir, device
    )
    return unicl_model


# DaViT stage-wise feature map hooks
def _extract_davit_backbone(backbone):
    """Extract the DaViT image encoder from the UniCL backbone."""
    if hasattr(backbone, "image_encoder"):
        return backbone.image_encoder
    return None


def _stage_feature_map_hooks(storage: dict, stage_idx: int):
    """
    Create a forward hook for a DaViT stage block.
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            features, spatial_size = output
        else:
            features = output
            spatial_size = None

        storage[stage_idx] = {
            "features": features.detach(),
            "spatial_size": spatial_size,
            "embed_dim": features.shape[-1],
        }
    return hook_fn


class MedImageInsightClassifier(BaseClassifier):
    """
    Linear classifier with MedImageInsight (UniCL / DaViT) backbone.

    Features are extracted via ``backbone.encode_image(x, norm=True)`` which chains:
    1. ``image_encoder.forward_features(x)`` -> ``[B, 2048]``
    2. ``x @ image_projection`` -> ``[B, 1024]``
    3. L2 normalisation

    Attention visualization is NOT supported for DaViT due to its dual
    (spatial-window + channel-group) attention mechanism which is hard to
    interpret as a single spatial heatmap.  Use ``--show-feature-maps``
    instead to visualize stage-wise feature activations.
    """

    def __init__(
        self,
        backbone,
        num_classes: int,
        multi_view: bool = False,
        num_views: Optional[int] = None,
        view_fusion_type: Optional[str] = None,
        adapter_dim: Optional[int] = None,
        view_fusion_hidden_dim: Optional[int] = None,
        return_attentions: bool = False,
    ):
        # Infer embedding dimension from the learned projection matrix
        # image_projection shape: [2048, 1024]
        embed_dim = backbone.image_projection.shape[1]  # 1024

        super().__init__(
            backbone=backbone,
            embed_dim=embed_dim,
            num_classes=num_classes,
            multi_view=multi_view,
            num_views=num_views,
            view_fusion_type=view_fusion_type,
            adapter_dim=adapter_dim,
            view_fusion_hidden_dim=view_fusion_hidden_dim,
        )
        if return_attentions:
            logger.warning(
                "MedImageInsight (DaViT) does not support attention visualization. "
                "The return_attentions flag is ignored. Use --show-feature-maps instead."
            )
        self._feature_storage: dict = {}
        self._hook_handles: list = []

    # ------------------------------------------------------------------
    # Stage-wise feature map hooks
    # ------------------------------------------------------------------

    def _enable_feature_map_hooks(self):
        """Register forward hooks on each DaViT stage to capture features."""
        self._feature_storage.clear()
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

        image_encoder = _extract_davit_backbone(self.backbone)
        if image_encoder is None:
            logger.warning("Cannot find DaViT image_encoder — feature capture disabled")
            return

        if not hasattr(image_encoder, "blocks") or len(image_encoder.blocks) == 0:
            logger.warning("DaViT image_encoder has no blocks — feature capture disabled")
            return

        for stage_idx, block in enumerate(image_encoder.blocks):
            hook_fn = _stage_feature_map_hooks(self._feature_storage, stage_idx)
            handle = block.register_forward_hook(hook_fn)
            self._hook_handles.append(handle)

        logger.debug("Registered feature capture hooks on %d DaViT stages", len(image_encoder.blocks))

    def _disable_feature_map_hooks(self):
        """Remove all forward hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def _collect_stage_features(self) -> Optional[Dict[int, Dict]]:
        """
        Package captured stage features for the visualizer.

        Returns:
            {stage_idx: {"features": [B, N, C], "spatial_size": (H, W), "embed_dim": C}} or None if nothing was captured.
        """
        if not self._feature_storage:
            return None
        return dict(self._feature_storage)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor):
        """
        Extract features via the full UniCL encoding path (L2-normalised).

        Attention maps are returned as None (DaViT's dual attention is not
        amenable to standard attention-map visualization).
        """
        features = self.backbone.encode_image(x, norm=True)
        return features, None # return features, None because attention maps are not supported for DaViT 

    def extract_stage_feature_maps(self, x: torch.Tensor):
        """
        Run a forward pass with stage-wise feature capture hooks.

        Returns:
            {stage_idx: {"features": [B, N, C], "spatial_size": (H, W), "embed_dim": C}}
            or None if capture failed.
        """
        self._enable_feature_map_hooks()
        with torch.no_grad():
            self.backbone.encode_image(x, norm=True)
        stage_features = self._collect_stage_features()
        self._disable_feature_map_hooks()
        return stage_features
