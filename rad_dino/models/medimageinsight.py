import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

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


class MedImageInsightClassifier(nn.Module):
    """
    Linear classifier with MedImageInsight (UniCL / DaViT) backbone.
    
    # TODO: Add attention maps for MedImageInsight (look into the attention maps in DaViT paper)
    forward() returns (logits, attention_maps) where attention_maps is always None for this architecture. 

    The backbone is the full UniCLModel. Features are extracted via backbone.encode_image(x, norm=True) which chains:
    1. image_encoder.forward_features(x) -> [B, 2048]
    2. x @ image_projection -> [B, 1024]
    3. L2 normalisation
    producing 1024-dim embeddings.
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
        """
        Args:
            backbone: UniCLModel instance.
            num_classes: Number of output classes for classification.
            multi_view: Whether to enable multi-view processing.
            num_views: Number of views (required when multi_view=True).
            view_fusion_type: Fusion strategy with options "mean", "weighted_mean", or "mlp_adapter".
            adapter_dim: Hidden dim for MLP adapters (only used when view_fusion_type="mlp_adapter").
            view_fusion_hidden_dim: Hidden dim for fusion MLP (only used when view_fusion_type="mlp_adapter").
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.multi_view = multi_view

        # Infer embedding dimension from the learned projection matrix
        # image_projection shape: [2048, 1024]
        self.embed_dim = backbone.image_projection.shape[1]  # 1024

        # Classification head
        self._init_classification_head()

        # Multi-view components (if needed)
        if self.multi_view:
            self._init_multi_view_components(
                num_views, view_fusion_type, adapter_dim, view_fusion_hidden_dim
            )

        # Strategy dispatch dictionaries
        self._init_strategy_dictionaries(view_fusion_type)

    def _init_classification_head(self):
        """Initialise the classification head."""
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def _init_multi_view_components(
        self,
        num_views: int,
        view_fusion_type: Optional[str],
        adapter_dim: Optional[int],
        view_fusion_hidden_dim: Optional[int],
    ):
        """Initialise multi-view components when multi_view=True."""
        assert num_views is not None, "Number of views is required for multi-view processing"
        assert view_fusion_type is not None, "View fusion type is required for multi-view processing"
        assert view_fusion_type in (
            "mean",
            "weighted_mean",
            "mlp_adapter",
        ), f"Invalid fusion type: {view_fusion_type}"

        self.num_views = num_views
        self.view_fusion_type = view_fusion_type

        if adapter_dim is None:
            adapter_dim = self.embed_dim
        if view_fusion_hidden_dim is None:
            view_fusion_hidden_dim = self.embed_dim

        if view_fusion_type == "mlp_adapter":
            self.view_adapters = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.embed_dim, adapter_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(adapter_dim, self.embed_dim),
                    )
                    for _ in range(num_views)
                ]
            )
            self.view_fusion_layer = nn.Sequential(
                nn.Linear(num_views * self.embed_dim, view_fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(view_fusion_hidden_dim, self.embed_dim),
                nn.ReLU(inplace=True),
            )
        elif view_fusion_type == "weighted_mean":
            self.view_scores = nn.Parameter(torch.zeros(num_views))
            self.view_fusion_layer = nn.Linear(self.embed_dim, self.embed_dim)

    def _init_strategy_dictionaries(self, view_fusion_type: Optional[str]):
        """Initialise strategy dictionaries for branch-free dispatch."""
        self.input_reshape_strategies = {
            True: self._multi_view_input_reshape,
            False: self._single_view_input_reshape,
        }
        self.normalization_strategies = {
            True: self._multi_view_normalization,
            False: self._single_view_normalization,
        }
        self.view_fusion_strategies = {
            "mean": self._mean_fusion,
            "weighted_mean": self._weighted_mean_fusion,
            "mlp_adapter": self._mlp_adapter_fusion,
        }

    # ------------------------------------------------------------------
    # Input reshaping
    # ------------------------------------------------------------------

    def _single_view_input_reshape(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Return ``(x, 1)`` unchanged for single-view input."""
        return x, 1

    def _multi_view_input_reshape(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Reshape ``[B, V, C, H, W]`` -> ``[B*V, C, H, W]``."""
        batch_size = x.shape[0]
        num_views = x.shape[1]
        x = x.view(batch_size * num_views, *x.shape[2:])
        return x, num_views

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _single_view_normalization(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def _multi_view_normalization(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0] // self.num_views
        return features.view(batch_size, self.num_views, -1)

    # ------------------------------------------------------------------
    # View fusion
    # ------------------------------------------------------------------

    def _single_view_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        return features

    def _mean_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        features = features.view(batch_size, num_views, self.embed_dim)
        return torch.mean(features, dim=1)

    def _weighted_mean_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        features = features.view(batch_size, num_views, self.embed_dim)
        weights = F.softmax(self.view_scores, dim=0)
        weighted = features * weights.unsqueeze(0).unsqueeze(-1)
        fused = torch.sum(weighted, dim=1)
        return self.view_fusion_layer(fused)

    def _mlp_adapter_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        features = features.view(batch_size, num_views, self.embed_dim)
        adapted = []
        for i in range(num_views):
            adapted.append(self.view_adapters[i](features[:, i, :]))
        concatenated = torch.cat(adapted, dim=1)
        return self.view_fusion_layer(concatenated)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the MedImageInsight classifier.

        Args:
            x: Input tensor.
                - Single-view: ``[B, C, H, W]``
                - Multi-view:  ``[B, V, C, H, W]``

        Returns:
            (logits, attention_maps) where attention_maps is always None.
            # TODO: Add attention maps for MedImageInsight (look into the attention maps in DaViT paper)
        """
        # Validate input shape matches model configuration
        if len(x.shape) == 5 and not self.multi_view:
            raise ValueError(
                "Model configured for single-view but received multi-view input"
            )
        if len(x.shape) == 4 and self.multi_view:
            raise ValueError(
                "Model configured for multi-view but received single-view input"
            )

        batch_size = x.shape[0]

        # Reshape input
        x_reshaped, num_views = self.input_reshape_strategies[self.multi_view](x)

        # Extract features via the full encoding path:
        # forward_features (2048) -> image_projection (1024) -> L2 norm
        features = self.backbone.encode_image(x_reshaped, norm=True)  # [B(*V), 1024]

        # View fusion
        fusion_strategy = self.view_fusion_strategies.get(
            getattr(self, "view_fusion_type", None), self._single_view_fusion
        )
        fused_features = fusion_strategy(
            features, batch_size, getattr(self, "num_views", 1)
        )

        # Classification
        logits = self.classifier(fused_features)  # [B, num_classes]

        return logits, None  # No attention maps for DaViT
