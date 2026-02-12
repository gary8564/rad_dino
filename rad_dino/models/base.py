"""
Base classifier module shared by all vision model classifiers.

Provides ``BaseClassifier`` — an abstract ``nn.Module`` that houses the
duplicated multi-view logic (input reshaping, view fusion strategies,
normalization, classification head) so that concrete classifiers only need
to implement :meth:`extract_features`.
"""

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional

from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


class BaseClassifier(nn.Module):
    """
    Abstract base class for all vision classifiers in the benchmark.

    Concrete subclasses must:
    1. Set ``self.embed_dim`` before calling ``super().__init__(...)``.
    2. Implement :meth:`extract_features` to return ``(features, attentions_or_none)``.

    Everything else — classification head, multi-view reshaping / fusion /
    normalization, and the shared ``forward()`` — lives here.
    """

    def __init__(
        self,
        backbone,
        embed_dim: int,
        num_classes: int,
        multi_view: bool = False,
        num_views: Optional[int] = None,
        view_fusion_type: Optional[str] = None,
        adapter_dim: Optional[int] = None,
        view_fusion_hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            backbone: The pre-trained backbone model (architecture-specific).
            embed_dim: Dimensionality of the feature vector produced by the backbone.
            num_classes: Number of output classes for classification.
            multi_view: Whether to enable multi-view processing.
            num_views: Number of views (required when ``multi_view=True``).
            view_fusion_type: Fusion strategy — ``"mean"``, ``"weighted_mean"``,
                or ``"mlp_adapter"``.
            adapter_dim: Hidden dim for per-view MLP adapters (``mlp_adapter`` only).
            view_fusion_hidden_dim: Hidden dim for the fusion MLP (``mlp_adapter`` only).
        """
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.multi_view = multi_view

        # Classification head
        self._init_classification_head()

        # Multi-view components (if needed)
        if self.multi_view:
            self._init_multi_view_components(
                num_views, view_fusion_type, adapter_dim, view_fusion_hidden_dim
            )

        # Strategy dispatch dictionaries for branch-free forward
        self._init_strategy_dictionaries(view_fusion_type)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_classification_head(self):
        """Create the linear classification head ``self.classifier``."""
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

    def _init_multi_view_components(
        self,
        num_views: int,
        view_fusion_type: Optional[str],
        adapter_dim: Optional[int],
        view_fusion_hidden_dim: Optional[int],
    ):
        """Initialise multi-view components when ``multi_view=True``."""
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
        elif view_fusion_type == "mean":
            self.view_fusion_layer = nn.Linear(self.embed_dim, self.embed_dim)

        # Layer normalization for feature stabilization
        self.layer_norm = nn.LayerNorm(self.embed_dim)

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
    # Input reshaping strategies
    # ------------------------------------------------------------------

    def _single_view_input_reshape(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Return ``(x, 1)`` unchanged for single-view input ``[B, C, H, W]``."""
        return x, 1

    def _multi_view_input_reshape(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        """Reshape ``[B, V, C, H, W]`` -> ``[B*V, C, H, W]``."""
        batch_size = x.shape[0]
        num_views = x.shape[1]
        assert num_views == self.num_views, (
            f"Expected {self.num_views} views, got {num_views}"
        )
        x = x.reshape(batch_size * num_views, *x.shape[2:])
        return x, num_views

    # ------------------------------------------------------------------
    # Normalization strategies
    # ------------------------------------------------------------------

    def _single_view_normalization(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """No-op normalization for single-view ``[B, D]`` features."""
        return features

    def _multi_view_normalization(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Apply LayerNorm for multi-view fused ``[B, D]`` features."""
        return self.layer_norm(features)

    # ------------------------------------------------------------------
    # View fusion strategies
    # ------------------------------------------------------------------

    def _single_view_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        """Return features unchanged for single-view input."""
        return features

    def _mean_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        """Simple average across views.

        ``[B*V, D]`` -> ``[B, V, D]`` -> mean -> ``[B, D]``
        """
        features = features.view(batch_size, num_views, self.embed_dim)
        return features.mean(dim=1)

    def _weighted_mean_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        """Learnable weighted average across views.

        ``[B*V, D]`` -> ``[B, V, D]`` -> weighted sum -> ``[B, D]``
        """
        features = features.view(batch_size, num_views, self.embed_dim)
        w = F.softmax(self.view_scores, dim=0)
        return (features * w[None, :, None]).sum(dim=1)

    def _mlp_adapter_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        """Per-view MLP adaptation then concatenation + fusion MLP.

        ``[B*V, D]`` -> per-view MLPs -> LayerNorm -> concat -> fusion MLP -> ``[B, D]``
        """
        features = features.view(batch_size, num_views, self.embed_dim)
        adapted = []
        for i in range(num_views):
            adapted.append(self.view_adapters[i](features[:, i, :]))
        adapted = torch.stack(adapted, dim=1)  # [B, V, D]
        adapted = self.layer_norm(adapted)
        return self.view_fusion_layer(
            adapted.view(batch_size, num_views * self.embed_dim)
        )

    # ------------------------------------------------------------------
    # Abstract feature extraction
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, any]:
        """
        Extract features from the backbone.

        Subclasses must implement this to handle backbone-specific logic.

        Args:
            x: Pre-processed input images ``[B(*V), C, H, W]``.

        Returns:
            ``(features, attentions)`` where:
            - ``features``: ``[B(*V), embed_dim]``
            - ``attentions``: Architecture-specific attention maps, or ``None``.
        """
        ...

    # ------------------------------------------------------------------
    # Shared forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Shared forward pass for all classifiers.

        Supports both single-view and multi-view inputs:
        - Single-view: ``x`` has shape ``[B, C, H, W]``
        - Multi-view:  ``x`` has shape ``[B, V, C, H, W]``

        Args:
            x: Input tensor.

        Returns:
            ``(logits, attentions)``
        """
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

        # Extract features (subclass-specific)
        features, attentions = self.extract_features(x_reshaped)

        # View fusion
        fusion_strategy = self.view_fusion_strategies.get(
            getattr(self, "view_fusion_type", None), self._single_view_fusion
        )
        fused_features = fusion_strategy(
            features, batch_size, getattr(self, "num_views", 1)
        )

        # Normalization
        fused_features = self.normalization_strategies[self.multi_view](
            fused_features
        )

        # Classification
        logits = self.classifier(fused_features)

        return logits, attentions
