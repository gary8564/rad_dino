import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import logging
from typing import Callable, Optional, List, Tuple
import timm.models.swin_transformer as swin
import timm
from rad_dino.models.base import BaseClassifier
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

TIMM_VERSION = timm.__version__

class WindowAttention(swin.WindowAttention):
    """
    Modified version of the original WindowAttention from timm that can return attention maps.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_attention = False
        self.attention_maps = []
        self.attention_metadata = []  # Store metadata for each attention map
    
    def forward(self, x, mask=None):
        """
        Forward pass with optional attention map return.
        
        Args:
            x: Input tensor
            mask: Attention mask
            
        Returns:
            Output tensor and attention maps if return_attention=True
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
        
        # Store attention maps if requested
        if self.return_attention:
            self.attention_maps.append(attn.detach())
            # Store metadata: (num_windows, window_size, window_size, num_heads)
            num_windows = B_ // (self.window_size[0] * self.window_size[1])
            window_size = self.window_size[0] * self.window_size[1]
            self.attention_metadata.append({
                'num_windows': num_windows,
                'window_size': window_size,
                'num_heads': self.num_heads,
                'seq_len': N
            })
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def get_attention_maps(self):
        """Get stored attention maps and clear the list."""
        maps = self.attention_maps.copy()
        metadata = self.attention_metadata.copy()
        self.attention_maps.clear()
        self.attention_metadata.clear()
        return maps, metadata

class SwinTransformerBlock(swin.SwinTransformerBlock):
    """
    SwinTransformerBlock that uses custom WindowAttention that can return attention maps.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the attention module with our custom one
        if hasattr(self, 'attn') and isinstance(self.attn, swin.WindowAttention):
            # Create a new WindowAttention with the same parameters
            old_attn = self.attn
            self.attn = WindowAttention(
                dim=old_attn.qkv.in_features,
                window_size=old_attn.window_size,
                num_heads=old_attn.num_heads,
                qkv_bias=old_attn.qkv.bias is not None,
                attn_drop=old_attn.attn_drop.p,
                proj_drop=old_attn.proj_drop.p
            )
            # Copy weights
            self.attn.qkv.weight.data = old_attn.qkv.weight.data
            if old_attn.qkv.bias is not None:
                self.attn.qkv.bias.data = old_attn.qkv.bias.data
            self.attn.proj.weight.data = old_attn.proj.weight.data
            if old_attn.proj.bias is not None:
                self.attn.proj.bias.data = old_attn.proj.bias.data
                
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
                 return_attention: bool = False,
                 grad_checkpointing: bool = False):
        """
        Initialize Swin Transformer for Ark with attention map support.
        
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
            return_attention: Whether to return attention maps
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
        self.return_attention = return_attention
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
        
        # Replace all transformer blocks with attention-enabled versions
        if self.return_attention:
            self._replace_attention_blocks()
    
    def _replace_attention_blocks(self):
        """Replace all transformer blocks with attention-enabled versions."""
        for stage in self.layers:
            for i, block in enumerate(stage.blocks):
                # Get input resolution from the block
                input_resolution = block.input_resolution
                
                # Create new block with attention maps
                new_block = SwinTransformerBlock(
                    dim=block.norm1.normalized_shape[0],
                    input_resolution=input_resolution,
                    num_heads=block.attn.num_heads,
                    window_size=block.attn.window_size,
                    shift_size=block.shift_size,
                    mlp_ratio=block.mlp_ratio,
                    qkv_bias=block.attn.qkv.bias is not None,
                    attn_drop=block.attn.attn_drop.p,
                    proj_drop=block.attn.proj_drop.p,
                    drop_path=block.drop_path2.p if hasattr(block.drop_path2, 'p') else 0.0,
                    norm_layer=type(block.norm1)
                )
                # Copy weights
                new_block.load_state_dict(block.state_dict())
                # Enable attention map collection
                new_block.attn.return_attention = True
                stage.blocks[i] = new_block
                
    def _collect_attention_maps(self):
        """Collect attention maps from all transformer blocks."""
        attention_maps = []
        attention_metadata = []
        stage_info = []
        
        for stage_idx, stage in enumerate(self.layers):
            stage_attention_maps = []
            stage_metadata = []
            
            for block_idx, block in enumerate(stage.blocks):
                if hasattr(block.attn, 'attention_maps'):
                    # Get attention maps 
                    maps = block.attn.attention_maps.copy()
                    if maps:  
                        stage_attention_maps.extend(maps)
                        # Create metadata for each attention map
                        for map_idx, attn_map in enumerate(maps):
                            meta = {
                                'stage': stage_idx,
                                'block': block_idx,
                                'map_idx': map_idx,
                                'shift_size': block.shift_size,
                                'window_size': block.attn.window_size[0] * block.attn.window_size[1] if hasattr(block.attn, 'window_size') else attn_map.shape[-1],
                                'num_heads': block.attn.num_heads,
                                'seq_len': attn_map.shape[-1]
                            }
                            stage_metadata.append(meta)
            
            if stage_attention_maps:
                attention_maps.extend(stage_attention_maps)
                attention_metadata.extend(stage_metadata)
                stage_info.append({
                    'stage_idx': stage_idx,
                    'num_blocks': len(stage.blocks),
                    'input_resolution': stage.blocks[0].input_resolution if stage.blocks else None,
                    'downsample': stage.downsample is not None
                })
        
        return attention_maps, attention_metadata, stage_info
    
    def _clear_attention_maps(self):
        """Clear all accumulated attention maps."""
        for stage in self.layers:
            for block in stage.blocks:
                if hasattr(block.attn, 'attention_maps'):
                    block.attn.attention_maps.clear()
                    block.attn.attention_metadata.clear()
    
    def forward_features(self, x):
        """Extract features from the backbone with optional attention maps."""
        # Ensure attention maps do not accumulate across forward calls
        if self.return_attention:
            self._clear_attention_maps()

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
            if self.return_attention:
                attention_maps, attention_metadata, stage_info = self._collect_attention_maps()
                return self.omni_heads[head_n](x), attention_maps
            else:
                return self.omni_heads[head_n](x), None
        else:
            outputs = [head(x) for head in self.omni_heads]
            if self.return_attention:
                attention_maps, attention_metadata, stage_info = self._collect_attention_maps()
                return outputs, attention_maps
            else:
                return outputs, None
    
    def get_hierarchical_attention_maps(self):
        """
        Get attention maps organized by stages for hierarchical visualization.
        
        Returns:
            dict: Dictionary with stage-wise attention maps and metadata
        """
        attention_maps, attention_metadata, stage_info = self._collect_attention_maps()
        
        # Organize attention maps by stage
        stage_attention_maps = {}
        for stage_idx in range(len(self.layers)):
            stage_attention_maps[stage_idx] = {
                'attention_maps': [],
                'metadata': [],
                'stage_info': stage_info[stage_idx] if stage_idx < len(stage_info) else None
            }
        
        # Populate stage-wise attention maps
        for attn_map, meta in zip(attention_maps, attention_metadata):
            stage_idx = meta['stage']
            if stage_idx in stage_attention_maps:
                stage_attention_maps[stage_idx]['attention_maps'].append(attn_map)
                stage_attention_maps[stage_idx]['metadata'].append(meta)
        
        return stage_attention_maps
    
    def get_last_layer_attention_maps(self):
        """
        Get attention maps from the last layer.
        
        Returns:
            tuple: (attention_maps, metadata) from the last stage
        """
        attention_maps, attention_metadata, _ = self._collect_attention_maps()
        
        if not attention_maps:
            return None, None
        
        # Get the last stage attention maps
        last_stage_maps = []
        last_stage_metadata = []
        
        for attn_map, meta in zip(attention_maps, attention_metadata):
            if meta['stage'] == len(self.layers) - 1:  # Last stage
                last_stage_maps.append(attn_map)
                last_stage_metadata.append(meta)
        
        return last_stage_maps, last_stage_metadata
    
    def generate_embeddings(self, x, after_proj: bool = True):
        """
        Generate embeddings for downstream tasks.
        
        Args:
            x: Input tensor
            after_proj: Whether to apply projection after feature extraction
            
        Returns:
            (embeddings, attention_maps) - attention_maps is None if return_attention=False
        """
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        
        if self.return_attention:
            attention_maps, attention_metadata, stage_info = self._collect_attention_maps()
            return x, attention_maps
        else:
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
            view_fusion_type: Fusion strategy — ``"mean"``, ``"weighted_mean"``, or ``"mlp_adapter"``.
            adapter_dim: Hidden dim for MLP adapters.
            view_fusion_hidden_dim: Hidden dim for fusion MLP.
            use_backbone_projector: If True, use backbone features after projection
                (linear probing); otherwise use features before projection (fine-tuning).
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

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor):
        """Extract features via ``backbone.generate_embeddings``."""
        features, attention_maps = self.backbone.generate_embeddings(
            x, after_proj=self.use_backbone_projector
        )
        return features, attention_maps

    # ------------------------------------------------------------------
    # Forward override — Ark has unique normalise-before-fusion flow
    # and list-based attention maps per Swin stage.
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the Ark classifier.

        Ark's flow differs from the base:
        1. Input reshape
        2. Feature extraction
        3. Normalise / reshape features (multi-view: ``[B*V, D]`` -> ``[B, V, D]``)
        4. Fusion (receives ``[B, V, D]`` for multi-view)
        5. Classification
        6. Reshape attention maps
        """
        batch_size = x.shape[0]

        # Input reshape
        x_reshaped, _ = self.input_reshape_strategies[self.multi_view](x)

        # Extract features
        features, attention_maps = self.extract_features(x_reshaped)

        # Normalise / reshape
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

        # Reshape attention maps for multi-view
        num_views = getattr(self, "num_views", 1)
        if attention_maps is not None and self.multi_view:
            reshaped_maps = []
            for attn_map in attention_maps:
                total = attn_map.shape[0]
                actual_bs = total // num_views
                reshaped_maps.append(
                    attn_map.reshape(actual_bs, num_views, *attn_map.shape[1:])
                )
            attention_maps = reshaped_maps

        return logits, attention_maps

    # ------------------------------------------------------------------
    # Ark-specific fusion (operates on [B, V, D] input, not [B*V, D])
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Hierarchical attention helpers
    # ------------------------------------------------------------------

    def get_hierarchical_attention_maps(self):
        """
        Get hierarchical attention maps organized by stages.
        
        Returns:
            dict: Stage-wise attention maps with metadata
        """
        if hasattr(self.backbone, 'get_hierarchical_attention_maps'):
            return self.backbone.get_hierarchical_attention_maps()
        return None
    
    def get_last_layer_attention_maps(self):
        """
        Get attention maps from the last layer only.
        
        Returns:
            tuple: (attention_maps, metadata) from the last stage
        """
        if hasattr(self.backbone, 'get_last_layer_attention_maps'):
            return self.backbone.get_last_layer_attention_maps()
        return None, None

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
                   return_attention: bool = False,
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
        projector_features: Dimension for projector (if None, no projector)
        use_mlp: Whether to use MLP projector
        return_attention: Whether to return attention maps
        grad_checkpointing: Whether to enable gradient checkpointing for memory efficiency
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
        use_mlp=use_mlp,
        return_attention=return_attention,
        grad_checkpointing=grad_checkpointing
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
        return_attention=True,  # Enable attention maps
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
    print("\n" + "="*50)
    print("Testing Single-view Forward Pass")
    print("="*50)
    dummy_input_single = torch.randn(2, 3, 768, 768)
    try:
        logits_single, attention_maps_single = model_single(dummy_input_single)
        print(f"Single-view forward pass successful. Output shape: {logits_single.shape}")
        if attention_maps_single is not None:
            print(f"Single-view attention maps: {len(attention_maps_single)} layers")
            if attention_maps_single:
                print(f"First attention map shape: {attention_maps_single[0].shape}")
        else:
            print("Single-view attention maps: None (not enabled)")
    except Exception as e:
        raise RuntimeError(f"Single-view forward pass failed: {e}")
    
    # Test multi-view forward pass
    print("\n" + "="*50)
    print("Testing Multi-view Forward Pass")
    print("="*50)
    dummy_input_multi = torch.randn(2, 4, 3, 768, 768)  # [batch_size, num_views, channels, height, width]
    try:
        logits_multi, attention_maps_multi = model_multi_mean(dummy_input_multi)
        print(f"Multi-view forward pass successful. Output shape: {logits_multi.shape}")
        if attention_maps_multi is not None:
            print(f"Multi-view attention maps: {len(attention_maps_multi)} layers")
            if attention_maps_multi:
                print(f"First attention map shape: {attention_maps_multi[0].shape}")
        else:
            print("Multi-view attention maps: None (not enabled)")
    except Exception as e:
        raise RuntimeError(f"Multi-view forward pass failed: {e}")
    
    # Test hierarchical attention maps (single-view backbone)
    print("\n" + "="*50)
    print("Testing Hierarchical Attention Maps (Single-view)")
    print("="*50)
    try:
        _ = model_single(dummy_input_single)
        hier_attn = model_single.get_hierarchical_attention_maps()
        if not hier_attn:
            raise RuntimeError("Failed to retrieve hierarchical attention maps: returned None/empty")
        print(f"Hierarchical attention maps: {len(hier_attn)} stages")
        for stage_idx, stage_data in hier_attn.items():
            num_maps = len(stage_data['attention_maps']) if stage_data and 'attention_maps' in stage_data else 0
            print(f"Stage {stage_idx}: {num_maps} attention maps")
            if stage_data and stage_data.get('stage_info'):
                print(f"  - Input resolution: {stage_data['stage_info']['input_resolution']}")
                print(f"  - Downsample present: {stage_data['stage_info']['downsample']}")
        
        last_maps, last_meta = model_single.get_last_layer_attention_maps()
        if last_maps is None:
            raise RuntimeError("Failed to retrieve last layer attention maps: returned None")
        print(f"Last layer attention maps: {len(last_maps)} maps")
        if last_meta:
            print(f"  - Sample meta: stage={last_meta[0].get('stage')}, block={last_meta[0].get('block')}, heads={last_meta[0].get('num_heads')}")
    except Exception as e:
        raise RuntimeError(f"Hierarchical attention test failed: {e}")
    
    print("All Ark model tests completed successfully!")
