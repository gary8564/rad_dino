import os
import sys
import torch
import logging
from typing import Optional

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
    # Make the cloned repo importable
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from MedImageInsight.UniCLModel import build_unicl_model
    from MedImageInsight.Utils.Arguments import load_opt_from_config_files

    config_path = os.path.join(model_dir, "2024.09.27", "config.yaml")
    config = load_opt_from_config_files([config_path])

    # Point to local weight files
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


class MedImageInsightClassifier(BaseClassifier):
    """
    Linear classifier with MedImageInsight (UniCL / DaViT) backbone.

    Features are extracted via ``backbone.encode_image(x, norm=True)`` which chains:
    1. ``image_encoder.forward_features(x)`` -> ``[B, 2048]``
    2. ``x @ image_projection`` -> ``[B, 1024]``
    3. L2 normalisation

    ``forward()`` returns ``(logits, None)`` — attention maps are not yet supported.
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

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor):
        """
        Extract features via the full UniCL encoding path (L2-normalised).

        Returns:
            ``(features, None)`` — attention maps not supported.
        """
        features = self.backbone.encode_image(x, norm=True)
        return features, None
