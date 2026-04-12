"""Ark+ classifier (Swin Transformer Large) with pretrained multi-task checkpoint loading."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import logging
from typing import Optional
import timm.models.swin_transformer as swin
import timm
from rad_dino.models.base import BaseClassifier
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

TIMM_VERSION = timm.__version__

class SwinTransformer(swin.SwinTransformer):
    def __init__(self,
                 num_classes_list: list[int],
                 img_size: int = 768,
                 patch_size: int = 4,
                 window_size: int = 12,
                 embed_dim: int = 192,
                 depths: tuple = (2, 2, 18, 2),
                 num_heads: tuple = (6, 12, 24, 48),
                 projector_features: int = 1376,
                 use_mlp: bool = False,
                 grad_checkpointing: bool = False):
        """
        Initialize Swin Transformer for Ark.

        Args:
            num_classes_list: List of number of output classes for each pretrained classification task
            img_size: Input image size
            patch_size: Patch size for embedding
            window_size: Window size for Swin Transformer
            embed_dim: Embedding dimension
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            projector_features: Dimension for projector
            use_mlp: Whether to use MLP projector
            grad_checkpointing: Whether to enable gradient checkpointing
        """
        super().__init__(
            num_classes=0,
            img_size=img_size,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        assert num_classes_list is not None
        self.num_classes_list = num_classes_list
        self.grad_checkpointing = grad_checkpointing
        
        # Initialize projector
        self.encoder_features = self.num_features
        self.num_features = projector_features
        if use_mlp:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder_features, self.num_features),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_features, self.num_features)
            )
        else:
            self.projector = nn.Linear(self.encoder_features, self.num_features)
        
        # Initialize omini classification head
        self.omni_heads = []
        for num_classes in self.num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)
        # Freeze omni_heads so they don't participate in grads during downstream training
        for p in self.omni_heads.parameters():
            p.requires_grad = False

    def forward_features(self, x):
        """Extract features from the backbone."""
        if self.grad_checkpointing:
            # Use gradient checkpointing for memory efficiency
            # Patch embedding
            x = self.patch_embed(x)
            
            # Apply checkpointing to the entire layers module
            x = checkpoint.checkpoint(
                lambda x_input: self.layers(x_input), 
                x,
                use_reentrant=False
            )
            
            # Apply final normalization
            x = self.norm(x)
        else:
            x = super().forward_features(x)
        
        # Handle compatibility between timm v0.5.4 and latest version
        # timm 0.5.x -> (B, C)
        # timm >= 0.8.x -> (B, L, C) or (B, H, W, C)
        if x.ndim == 3:           # (B, L, C)
            logger.info(f"timm version {TIMM_VERSION}: (B, L, C) -> (B, C) need to be handled manually!")
            x = x.transpose(1, 2)         # (B, C, L)
            x = F.adaptive_avg_pool1d(x, 1)  # (B, C, 1)
            x = x.flatten(1)              # (B, C)
        elif x.ndim == 4:         # (B, H, W, C)
            logger.info(f"timm version {TIMM_VERSION}: (B, H, W, C) -> (B, C) need to be handled manually!")
            x = x.permute(0, 3, 1, 2)     # (B, C, H, W)
            x = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
            x = x.flatten(1)              # (B, C)
        
        return x
    
    def forward(self, x, head_n: Optional[int] = None):
        """Forward pass through the model."""
        x = self.forward_features(x)
        x = self.projector(x)
        
        if head_n is not None:
            return self.omni_heads[head_n](x), None
        else:
            outputs = [head(x) for head in self.omni_heads]
            return outputs, None

    # Stage-wise feature map hooks
    def _enable_feature_map_hooks(self):
        """Register forward hooks on each Swin stage to capture output features."""
        if not hasattr(self, '_feature_storage'):
            self._feature_storage: dict = {}
            self._feature_hook_handles: list = []
        self._feature_storage.clear()
        for h in self._feature_hook_handles:
            h.remove()
        self._feature_hook_handles.clear()

        for stage_idx, stage in enumerate(self.layers):
            input_resolution = stage.blocks[0].input_resolution  # (H, W)

            def hook_fn(module, input, output, s_idx=stage_idx, res=input_resolution):
                if output.ndim == 3:
                    feat = output  # [B, L, C]
                    L = feat.shape[1]
                    H_in, W_in = res
                    if L != H_in * W_in:
                        # PatchMerging downsample halves each spatial dim
                        res = (H_in // 2, W_in // 2)
                elif output.ndim == 4:
                    B, H, W, C = output.shape
                    feat = output.view(B, H * W, C)
                    res = (H, W)
                else:
                    return
                self._feature_storage[s_idx] = {
                    "features": feat.detach(),
                    "spatial_size": res,
                    "embed_dim": feat.shape[-1],
                }

            handle = stage.register_forward_hook(hook_fn)
            self._feature_hook_handles.append(handle)

        logger.debug("Registered feature capture hooks on %d Swin stages", len(self.layers))

    def _disable_feature_map_hooks(self):
        """Remove feature capture hooks."""
        if hasattr(self, '_feature_hook_handles'):
            for h in self._feature_hook_handles:
                h.remove()
            self._feature_hook_handles.clear()

    def _collect_stage_features(self) -> Optional[dict]:
        """Return captured stage features or None."""
        if not hasattr(self, '_feature_storage') or not self._feature_storage:
            return None
        return dict(self._feature_storage)

    def generate_embeddings(self, x, after_proj: bool = True):
        """
        Generate embeddings for downstream tasks.

        Args:
            x: Input tensor
            after_proj: Whether to apply projection after feature extraction

        Returns:
            ``(embeddings, None)``
        """
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x, None
    
    def get_feature_dimension(self, after_proj: bool = True) -> int:
        """
        Get the feature dimension for the current configuration.
        
        Args:
            after_proj: Whether to return dimension after projection
            
        Returns:
            Feature dimension
        """
        if after_proj:
            return self.num_features
        else:
            return self.encoder_features


class ArkClassifier(BaseClassifier):
    """
    Ark classifier built on a Swin Transformer backbone.

    Supports multi-view processing and various fusion strategies.
    """

    def __init__(
        self,
        backbone: SwinTransformer,
        num_classes: int,
        multi_view: bool = False,
        num_views: Optional[int] = None,
        view_fusion_type: Optional[str] = None,
        adapter_dim: Optional[int] = None,
        view_fusion_hidden_dim: Optional[int] = None,
        use_backbone_projector: bool = False,
    ):
        """
        Args:
            backbone: Pre-trained Ark Swin Transformer backbone.
            num_classes: Number of output classes for classification.
            multi_view: Whether to enable multi-view processing.
            num_views: Number of views (required when multi_view=True).
            view_fusion_type: Fusion strategy — "mean", "weighted_mean", or "mlp_adapter".
            adapter_dim: Hidden dim for MLP adapters.
            view_fusion_hidden_dim: Hidden dim for fusion MLP.
            use_backbone_projector: If True, use backbone features after projection
                (linear probing); otherwise use features before projection (fine-tuning)
        """
        self.use_backbone_projector = use_backbone_projector
        if use_backbone_projector:
            embed_dim = backbone.num_features
        else:
            embed_dim = (
                backbone.num_features
                if backbone.projector is None
                else backbone.projector.in_features
            )

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

    # Feature extraction
    def extract_features(self, x: torch.Tensor):
        """Extract features via backbone.generate_embeddings."""
        features, attention_maps = self.backbone.generate_embeddings(
            x, after_proj=self.use_backbone_projector
        )
        return features, attention_maps

    def extract_stage_feature_maps(self, x: torch.Tensor):
        """
        Run a forward pass with stage-wise feature capture hooks.

        Returns:
            {stage_idx: {"features": [B, N, C], "spatial_size": (H, W), "embed_dim": C}}
            or None if capture failed.
        """
        self.backbone._enable_feature_map_hooks()
        with torch.no_grad():
            self.backbone.forward_features(x)
        stage_features = self.backbone._collect_stage_features()
        self.backbone._disable_feature_map_hooks()
        return stage_features

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the Ark classifier.
        """
        batch_size = x.shape[0]

        # Input reshape
        x_reshaped, _ = self.input_reshape_strategies[self.multi_view](x)

        # Extract features
        features, attention_maps = self.extract_features(x_reshaped)

        # Normalize and reshape
        if self.multi_view:
            features = features.view(batch_size, self.num_views, -1)

        # Fusion
        fusion_strategy = self.view_fusion_strategies.get(
            getattr(self, "view_fusion_type", None), self._single_view_fusion
        )
        features = fusion_strategy(
            features, batch_size, getattr(self, "num_views", 1)
        )

        # Classification
        logits = self.classifier(features)

        return logits, None

    def _mean_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        """Mean fusion across views. Input: ``[B, V, D]``."""
        return torch.mean(features, dim=1)

    def _weighted_mean_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        """Weighted mean fusion across views. Input: ``[B, V, D]``."""
        weights = F.softmax(self.view_scores, dim=0)
        weighted = features * weights.unsqueeze(0).unsqueeze(-1)
        return self.view_fusion_layer(torch.sum(weighted, dim=1))

    def _mlp_adapter_fusion(
        self, features: torch.Tensor, batch_size: int, num_views: int
    ) -> torch.Tensor:
        """MLP adapter fusion across views. Input: ``[B, V, D]``."""
        adapted = []
        for i in range(num_views):
            adapted.append(self.view_adapters[i](features[:, i, :]))
        concatenated = torch.cat(adapted, dim=1)
        return self.view_fusion_layer(concatenated)

def load_pretrained_ark_model(checkpoint_path: str,
                   num_classes_list: list[int] | None = None,
                   img_size: int = 768,
                   patch_size: int = 4,
                   window_size: int = 12,
                   embed_dim: int = 192,
                   depths: tuple = (2, 2, 18, 2),
                   num_heads: tuple = (6, 12, 24, 48),
                   projector_features: int = 1376,
                   use_mlp: bool = False,
                   grad_checkpointing: bool = False,
                   device: str = "cpu") -> SwinTransformer:
    """
    Load a pre-trained Ark model from checkpoint.

    Args:
        checkpoint_path: Path to the Ark checkpoint file
        num_classes_list: List of number of output classes for each pretrained classification task
        img_size: Input image size
        patch_size: Patch size for embedding
        window_size: Window size for Swin Transformer
        embed_dim: Embedding dimension
        depths: Number of layers in each stage
        num_heads: Number of attention heads in each stage
        projector_features: Dimension for projector
        use_mlp: Whether to use MLP projector
        grad_checkpointing: Whether to enable gradient checkpointing
        device: Device to load the model on

    Returns:
        Loaded SwinTransformer model
    """
    if num_classes_list is None:
        num_classes_list = [14, 14, 14, 3, 6, 1]

    model = SwinTransformer(
        num_classes_list=num_classes_list,
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        projector_features=projector_features,
        use_mlp=use_mlp,
        grad_checkpointing=grad_checkpointing,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)    
    state_dict = checkpoint["teacher"]

    # Remove "module." prefix if present (for DataParallel models)
    if any([True if 'module.' in k else False for k in state_dict.keys()]):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}
    
    # Remove unnecessary keys
    keys_to_delete = []
    for k in state_dict.keys():
        if "attn_mask" in k or k in ["head.weight", "head.bias"]:
            keys_to_delete.append(k)
    # Delete identified keys
    for k in keys_to_delete:
        if k in state_dict: # Ensure the key exists
            del state_dict[k]
    
    # Handle compatibility between timm v0.5.4 and latest version:
    # Map old layer names to new layer names
    new_state_dict = swin.checkpoint_filter_fn(state_dict, model)
    
    # Load state dict
    msg = model.load_state_dict(new_state_dict, strict=False)
    logger.info(f'Loaded Ark model with msg: {msg}')
    return model

