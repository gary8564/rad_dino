import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Callable, Optional
import timm.models.swin_transformer as swin
import timm
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
                 use_mlp: bool = False):
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
            projector_features: Dimension for projector (if None, no projector)
            use_mlp: Whether to use MLP projector
        """
        super().__init__(
            num_classes=0,  # Handle classification separately
            img_size=img_size,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        assert num_classes_list is not None
        self.num_classes_list = num_classes_list
        
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
    
    def forward_features(self, x):
        """Extract features from the backbone."""
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
    
        
        
        
    
    def forward(self, x, head_n=None):
        """Forward pass through the model."""
        x = self.forward_features(x)
        x = self.projector(x)
        if head_n is not None:
            return x, self.omni_heads[head_n](x)
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj: bool = True):
        """Generate embeddings for downstream tasks."""
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        return x
    
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


class ArkClassifier(nn.Module):
    """
    Ark classifier that follows the same interface as DinoClassifier and MedSigClassifier.
    Supports multi-view processing and various fusion strategies.
    """
    def __init__(self, 
                 backbone: SwinTransformer, 
                 num_classes: int, 
                 multi_view: bool = False,
                 num_views: Optional[int] = None,
                 view_fusion_type: Optional[str] = None,
                 adapter_dim: Optional[int] = None,
                 view_fusion_hidden_dim: Optional[int] = None,
                 use_backbone_projector: bool = False):
        """
        Initialize the Ark classifier.
        
        Args:
            backbone: Pre-trained Ark Swin Transformer backbone
            num_classes: Number of output classes for classification
            multi_view: Whether to enable multi-view processing
            num_views: Number of views to process (only used if multi_view=True)
            view_fusion_type: Fusion strategy for multi-view processing (only used if multi_view=True)
                - "mean": Simple average across views
                - "weighted_mean": Learnable weighted average
                - "mlp_adapter": MLP-based feature adaptation and fusion
            adapter_dim: Hidden dimension for MLP adapters (only used if multi_view=True and fusion_type="mlp_adapter")
            view_fusion_hidden_dim: Hidden dimension for fusion MLP (only used if multi_view=True and fusion_type="mlp_adapter")
            use_backbone_projector: Whether to use the backbone projector. 
                                    For linear probing, use the feature dimension from the backbone after projection.
                                    For fine-tuning, use the feature dimension from the backbone before projection.
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.multi_view = multi_view
        self.use_backbone_projector = use_backbone_projector
        if use_backbone_projector:
            # For linear probing, use the feature dimension from the backbone after projection
            self.embed_dim = backbone.num_features
        else:
            # For fine-tuning, use the feature dimension from the backbone before projection
            self.embed_dim = backbone.num_features if backbone.projector is None else backbone.projector.in_features
        
        # Initialize classification head 
        self._init_classification_head()
        
        # Initialize multi-view components only if needed
        if self.multi_view:
            self._init_multi_view_components(num_views, view_fusion_type, adapter_dim, view_fusion_hidden_dim)
        
        # Initialize strategy function dictionaries for branch-free dispatch
        self._init_strategy_dictionaries(view_fusion_type)
        
    def _init_multi_view_components(self, 
                                    num_views: int, 
                                    view_fusion_type: str | None, 
                                    adapter_dim: int | None, 
                                    view_fusion_hidden_dim: int | None):
        """
        Initialize multi-view components when multi_view=True.
        Args:
            num_views: Number of views to process
            view_fusion_type: Type of fusion strategy to use
            adapter_dim: Hidden dimension for MLP adapters
            view_fusion_hidden_dim: Hidden dimension for fusion MLP
        """
        # Validate multi-view parameters
        assert num_views is not None, "Number of views is required for multi-view processing"
        assert view_fusion_type is not None, "View fusion type is required for multi-view processing"
        assert view_fusion_type in ["mean", "weighted_mean", "mlp_adapter"], f"Invalid fusion type: {view_fusion_type}"
        
        # Set multi-view configuration
        self.num_views = num_views
        self.view_fusion_type = view_fusion_type
        
        # Set default dimensions if not provided
        if adapter_dim is None:
            adapter_dim = self.embed_dim
        if view_fusion_hidden_dim is None:
            view_fusion_hidden_dim = self.embed_dim
            
        # Initialize components based on fusion type
        if view_fusion_type == "mlp_adapter":
            # MLP adapters for each view
            self.view_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embed_dim, adapter_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(adapter_dim, self.embed_dim),
                ) for _ in range(num_views)
            ])
            
            # Fusion layer for combining adapted features
            self.view_fusion_layer = nn.Sequential(
                nn.Linear(num_views * self.embed_dim, view_fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(view_fusion_hidden_dim, self.embed_dim),
                nn.ReLU(inplace=True),
            )
            
        elif view_fusion_type == "weighted_mean":
            # Learnable weights for weighted mean fusion
            self.view_scores = nn.Parameter(torch.zeros(num_views))
            # Simple linear layer for weighted mean
            self.view_fusion_layer = nn.Linear(self.embed_dim, self.embed_dim)
    
    def _init_classification_head(self):
        """Initialize the classification head."""
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
    
    def _init_strategy_dictionaries(self, view_fusion_type: str | None):
        """Initialize strategy dictionaries for branch-free dispatch in the forward pass."""
        # Feature extraction strategies (use_backbone_projector -> strategy)
        self.feature_extraction_strategies = {
            True: self._extract_features_with_projection,
            False: self._extract_features_without_projection
        }
        
        # Input reshape strategies (multi_view -> strategy)
        self.input_reshape_strategies = {
            True: self._multi_view_input_reshape,
            False: self._single_view_input_reshape
        }
        
        # Normalization strategies (multi_view -> strategy)
        self.normalization_strategies = {
            True: self._multi_view_normalization,
            False: self._single_view_normalization
        }
        
        # View fusion strategies (view_fusion_type -> strategy)
        self.view_fusion_strategies = {
            "mean": self._mean_fusion,
            "weighted_mean": self._weighted_mean_fusion,
            "mlp_adapter": self._mlp_adapter_fusion
        }

    def _extract_features_with_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with projection applied."""
        return self.backbone.generate_embeddings(x, after_proj=True)
    
    def _extract_features_without_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without projection."""
        return self.backbone.generate_embeddings(x, after_proj=False)
    
    def _single_view_input_reshape(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Reshape input for single view processing."""
        batch_size = x.shape[0]
        return x, batch_size
    
    def _multi_view_input_reshape(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Reshape input for multi-view processing."""
        # Expected input shape: [batch_size, num_views, channels, height, width]
        batch_size = x.shape[0]
        num_views = x.shape[1]
        
        # Reshape to [batch_size * num_views, channels, height, width]
        x = x.view(batch_size * num_views, *x.shape[2:])
        return x, batch_size
    
    def _single_view_normalization(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features for single view processing."""
        return features
    
    def _multi_view_normalization(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features for multi-view processing."""
        # Reshape back to [batch_size, num_views, embed_dim]
        batch_size = features.shape[0] // self.num_views
        features = features.view(batch_size, self.num_views, -1)
        return features
    
    def _single_view_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """Fusion strategy for single view processing."""
        return features
    
    def _mean_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """Mean fusion across views."""
        # features shape: [batch_size, num_views, embed_dim]
        return torch.mean(features, dim=1)  # [batch_size, embed_dim]
    
    def _weighted_mean_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """Weighted mean fusion across views."""
        # features shape: [batch_size, num_views, embed_dim]
        # Apply learnable weights
        weights = F.softmax(self.view_scores, dim=0)  # [num_views]
        weighted_features = features * weights.unsqueeze(0).unsqueeze(-1)  # [batch_size, num_views, embed_dim]
        fused_features = torch.sum(weighted_features, dim=1)  # [batch_size, embed_dim]
        return self.view_fusion_layer(fused_features)
    
    def _mlp_adapter_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """MLP adapter fusion across views."""
        # features shape: [batch_size, num_views, embed_dim]
        adapted_features = []
        for i in range(num_views):
            adapted = self.view_adapters[i](features[:, i, :])  # [batch_size, embed_dim]
            adapted_features.append(adapted)
        
        # Concatenate adapted features
        concatenated = torch.cat(adapted_features, dim=1)  # [batch_size, num_views * embed_dim]
        fused_features = self.view_fusion_layer(concatenated)  # [batch_size, embed_dim]
        return fused_features
    
    def forward(self, x):
        """
        Forward pass through the Ark classifier.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width] for single view
               or [batch_size, num_views, channels, height, width] for multi-view
        """
        # Reshape input
        x, batch_size = self.input_reshape_strategies[self.multi_view](x)
        
        # Extract features from backbone
        features = self.feature_extraction_strategies[self.use_backbone_projector](x)
        
        # Normalize features
        features = self.normalization_strategies[self.multi_view](features)
        
        # Apply fusion strategy
        fusion_strategy = self.view_fusion_strategies.get(getattr(self, 'view_fusion_type', None), self._single_view_fusion)
        features = fusion_strategy(features, batch_size, getattr(self, 'num_views', 1))
        
        # Apply classification head
        logits = self.classifier(features)
        
        return logits


def load_prtrained_ark_model(checkpoint_path: str, 
                   num_classes_list: list[int] = [14, 14, 14, 3, 6, 1],
                   img_size: int = 768,
                   patch_size: int = 4,
                   window_size: int = 12,
                   embed_dim: int = 192,
                   depths: tuple = (2, 2, 18, 2),
                   num_heads: tuple = (6, 12, 24, 48),
                   projector_features: int = 1376,
                   use_mlp: bool = False,
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
        projector_features: Dimension for projector (if None, no projector)
        use_mlp: Whether to use MLP projector
        device: Device to load the model on
    
    Returns:
        Loaded SwinTransformer model
    """
    # Initialize model
    model = SwinTransformer(
        num_classes_list=num_classes_list,
        img_size=img_size,
        patch_size=patch_size,
        window_size=window_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        projector_features=projector_features,
        use_mlp=use_mlp
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

if __name__ == "__main__":
    import os
    
    def unfreeze_layers(model: ArkClassifier, num_unfreeze_layers: int):
        """
        Unfreeze the last n layers of the Ark backbone.
        
        Args:
            model: ArkClassifier model
            num_unfreeze_layers: Number of layers to unfreeze from the end
        """
        # For Swin Transformer, we need to unfreeze specific stages
        # The backbone has 4 stages, each with multiple layers
        total_stages = len(model.backbone.layers)
        
        if num_unfreeze_layers > total_stages or num_unfreeze_layers < 1:
            raise ValueError(f"Number of unfreeze layers {num_unfreeze_layers} cannot be greater than the total number of stages {total_stages} or less than 1")
        
        # First freeze all backbone parameters
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        
        # Then unfreeze the specified stages from the end
        for i in range(total_stages - 1, total_stages - num_unfreeze_layers - 1, -1):
            for name, param in model.backbone.named_parameters():
                if f"layers.{i}" in name:
                    param.requires_grad = True
    
    # Load real Ark checkpoint
    checkpoint_path = "/hpcwork/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"
    
    # Load the pre-trained Ark model
    ark_backbone = load_prtrained_ark_model(
        checkpoint_path=checkpoint_path,
        num_classes_list=[14,14,14,3,6,1],  # This will be overridden by ArkClassifier
        img_size=768,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        projector_features=1376,
        use_mlp=False,
        device="cpu"
    )        
    
    # Test single-view classifier
    model_single = ArkClassifier(
        backbone=ark_backbone, 
        num_classes=10, 
        multi_view=False,
        use_backbone_projector=True
    )
    
    # Test multi-view classifier with different fusion types
    model_multi_mean = ArkClassifier(
        backbone=ark_backbone, 
        num_classes=10, 
        multi_view=True,
        num_views=4,
        view_fusion_type="mean",
        use_backbone_projector=True
    )
    
    model_multi_weighted = ArkClassifier(
        backbone=ark_backbone, 
        num_classes=10, 
        multi_view=True,
        num_views=4,
        view_fusion_type="weighted_mean",
        use_backbone_projector=True
    )
    
    model_multi_adapter = ArkClassifier(
        backbone=ark_backbone, 
        num_classes=10, 
        multi_view=True,
        num_views=4,
        view_fusion_type="mlp_adapter",
        adapter_dim=512,
        view_fusion_hidden_dim=512,
        use_backbone_projector=True
    )
    
    # Test unfreezing layers
    unfreeze_layers(model_multi_mean, 2)
    
    # Print trainable parameters for debugging
    for name, param in model_multi_mean.named_parameters():
        if 'backbone' in name:
            if param.requires_grad:
                print(f"Trainable backbone parameter: {name}")
        else:
            param.requires_grad = True
            print(f"Trainable parameter: {name}")
    
    # Print model statistics
    total_params = sum(p.numel() for p in model_multi_mean.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model_multi_mean.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # Test forward pass with dummy data    
    # Test single-view forward pass
    dummy_input_single = torch.randn(2, 3, 768, 768)
    try:
        logits_single = model_single(dummy_input_single)
        print(f"Single-view forward pass successful. Output shape: {logits_single.shape}")
    except Exception as e:
        raise RuntimeError(f"Single-view forward pass failed: {e}")
    
    # Test multi-view forward pass
    dummy_input_multi = torch.randn(2, 4, 3, 768, 768)  # [batch_size, num_views, channels, height, width]
    try:
        logits_multi = model_multi_mean(dummy_input_multi)
        print(f"Multi-view forward pass successful. Output shape: {logits_multi.shape}")
    except Exception as e:
        raise RuntimeError(f"Multi-view forward pass failed: {e}")
    
    print("All Ark model tests completed successfully!")
