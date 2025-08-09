from transformers import AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Callable, Optional

class MedSigClassifier(nn.Module):
    def __init__(self, 
                 backbone: AutoModel, 
                 num_classes: int,
                 multi_view: bool = False,
                 num_views: Optional[int] = None,
                 view_fusion_type: Optional[str] = None,
                 adapter_dim: Optional[int] = None,
                 view_fusion_hidden_dim: Optional[int] = None):
        """
        Initialize the MedSigLIP classifier.
        
        Args:
            backbone: Pre-trained MedSigLIP backbone model
            num_classes: Number of output classes for classification
            multi_view: Whether to enable multi-view processing
            num_views: Number of views to process (only used if multi_view=True)
            view_fusion_type: Fusion strategy for multi-view processing (only used if multi_view=True)
                - "mean": Simple average across views
                - "weighted_mean": Learnable weighted average
                - "mlp_adapter": MLP-based feature adaptation and fusion
            adapter_dim: Hidden dimension for MLP adapters (only used if multi_view=True and view_fusion_type="mlp_adapter")
            view_fusion_hidden_dim: Hidden dimension for fusion MLP (only used if multi_view=True and view_fusion_type="mlp_adapter")
        """
        super().__init__()
        self.backbone = backbone
        self.feat_dim = self.backbone.config.text_config.projection_size  # MedSigLIP's feature dim
        self.num_classes = num_classes
        self.multi_view = multi_view
        
        # Initialize classification head
        self._init_classification_head()
        
        # Initialize multi-view components only if needed
        if self.multi_view:
            self._init_multi_view_components(num_views, view_fusion_type, adapter_dim, view_fusion_hidden_dim)
        
        # Initialize strategy dictionaries for branch-free dispatch
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
        assert view_fusion_type is not None, "view_fusion_type is required for multi-view processing"
        assert view_fusion_type in ["mean", "weighted_mean", "mlp_adapter"], f"Invalid view_fusion_type: {view_fusion_type}"
        
        # Set multi-view configuration
        self.num_views = num_views
        self.view_fusion_type = view_fusion_type
        
        # Set default dimensions if not provided
        if adapter_dim is None:
            adapter_dim = self.feat_dim
        if view_fusion_hidden_dim is None:
            view_fusion_hidden_dim = self.feat_dim
            
        # Initialize components based on fusion type
        if view_fusion_type == "mlp_adapter":
            # MLP adapters for each view
            self.view_adapters = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.feat_dim, adapter_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(adapter_dim, self.feat_dim),
                ) for _ in range(num_views)
            ])
            
            # Fusion layer for combining adapted features
            self.fusion_layer = nn.Sequential(
                nn.Linear(num_views * self.feat_dim, view_fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(view_fusion_hidden_dim, self.feat_dim),
                nn.ReLU(inplace=True),
            )
            
        elif view_fusion_type == "weighted_mean":
            # Learnable weights for weighted mean fusion
            self.view_scores = nn.Parameter(torch.zeros(num_views))
            # Simple linear layer for weighted mean
            self.fusion_layer = nn.Linear(self.feat_dim, self.feat_dim)
            
        elif view_fusion_type == "mean":
            # Simple linear layer for mean fusion
            self.fusion_layer = nn.Linear(self.feat_dim, self.feat_dim)
        
        # Layer normalization for feature stabilization (used in all multi-view cases)
        self.layer_norm = nn.LayerNorm(self.feat_dim)
        
    def _init_classification_head(self):
        """Initialize classification head"""
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes)
        )
    
    def _init_strategy_dictionaries(self, view_fusion_type: str | None):
        """Initialize strategy dictionaries for branch-free dispatch, eliminating conditional branches in the forward pass."""
        # Input reshape strategies (multi_view -> strategy)
        self.input_reshape_strategies = {
            True: self._multi_view_input_reshape,
            False: self._single_view_input_reshape
        }
        
        # Attention reshape strategies (multi_view -> strategy)
        self.attention_reshape_strategies = {
            True: self._multi_view_attention_reshape,
            False: self._single_view_attention_reshape
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
    
    def _single_view_input_reshape(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Single view input reshaping strategy.
        
        Args:
            pixel_values: Input tensor [B, C, H, W]
            
        Returns:
            tuple: (reshaped_tensor, num_views)
        """
        return pixel_values, 1
    
    def _multi_view_input_reshape(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Multi-view input reshaping strategy.
        
        Args:
            pixel_values: Input tensor [B, V, C, H, W]
            
        Returns:
            tuple: (reshaped_tensor, num_views)
        """
        batch_size = pixel_values.shape[0]
        num_views = pixel_values.shape[1]
        assert num_views == self.num_views, f"Expected {self.num_views} views, got {num_views}"
        # Reshape: [B, V, C, H, W] -> [B*V, C, H, W]
        pixel_values_reshaped = pixel_values.reshape(batch_size * num_views, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4])
        return pixel_values_reshaped, num_views
    
    def _single_view_attention_reshape(self, stacked_attns: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Single view attention reshaping strategy.
        
        Args:
            stacked_attns: Attention maps [L, B, N_heads, N_seq, N_seq]
            batch_size: Batch size
            num_views: Number of views (always 1)
            
        Returns:
            Unmodified attention maps
        """
        return stacked_attns
    
    def _multi_view_attention_reshape(self, stacked_attns: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Multi-view attention reshaping strategy.
        
        Args:
            stacked_attns: Attention maps [L, B*V, N_heads, N_seq, N_seq]
            batch_size: Batch size
            num_views: Number of views
            
        Returns:
            Reshaped attention maps [L, B, V, N_heads, N_seq, N_seq]
        """
        return stacked_attns.reshape(-1, batch_size, num_views, *stacked_attns.shape[2:])
    
    def _single_view_normalization(self, features: torch.Tensor) -> torch.Tensor:
        """
        Single view normalization strategy (no normalization).
        
        Args:
            features: Features to normalize [B, D]
            
        Returns:
            Original features without any normalization [B, D]
        """
        return features
    
    def _multi_view_normalization(self, features: torch.Tensor) -> torch.Tensor:
        """
        Multi-view normalization strategy (LayerNorm).
        
        Args:
            features: Features to normalize [B, D]
            
        Returns:
            Normalized features [B, D]
        """
        return self.layer_norm(features)
    
    def _single_view_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Single view fusion - return the features directly.
        
        Args:
            features: Features from backbone [B, D]
            batch_size: Batch size
            num_views: Number of views (always 1 for single view)
            
        Returns:
            Original features [B, D]
        """
        return features
    
    def _mean_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Mean fusion across views - simple average.
        
        Args:
            features: Features from backbone [B*V, D]
            batch_size: Batch size
            num_views: Number of views (always > 1 for multi-view)
            
        Returns:
            Averaged features [B, D]
        """
        # Reshape: [B*V, D] -> [B, V, D]
        features = features.view(batch_size, num_views, self.feat_dim)
        return features.mean(dim=1)
    
    def _weighted_mean_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        Weighted mean fusion across views using learnable weights.
        
        Args:
            features: Features from backbone [B*V, D]
            batch_size: Batch size
            num_views: Number of views (always > 1 for multi-view)
            
        Returns:
            Weighted averaged features [B, D]
        """
        # Reshape: [B*V, D] -> [B, V, D]
        features = features.view(batch_size, num_views, self.feat_dim)
        w = F.softmax(self.view_scores, dim=0)
        return (features * w[None,:,None]).sum(dim=1)
    
    def _mlp_adapter_fusion(self, features: torch.Tensor, batch_size: int, num_views: int) -> torch.Tensor:
        """
        MLP adapter fusion - adapt each view with MLP then fuse.
        
        This strategy:
        1. Applies separate MLP adapters to each view
        2. Normalizes adapted features with LayerNorm
        3. Concatenates all adapted features
        4. Applies final fusion MLP
        
        Args:
            features: Features from backbone [B*V, D]
            batch_size: Batch size
            num_views: Number of views (always > 1 for multi-view)
            
        Returns:
            Fused features [B, D]
        """
        # Reshape: [B*V, D] -> [B, V, D]
        features = features.view(batch_size, num_views, self.feat_dim)
        
        # Adapt each view with its own MLP adapter
        adapted_features = []
        for i in range(num_views):
            adapted_features.append(self.view_adapters[i](features[:,i,:]))
        adapted_features = torch.stack(adapted_features, dim=1)  # [B, V, D]
        
        # Apply LayerNorm to normalized adapted features
        adapted_features = self.layer_norm(adapted_features)
        
        # Concatenate and apply fusion MLP layer
        return self.fusion_layer(adapted_features.view(batch_size, num_views * self.feat_dim))

    def forward(self, pixel_values):
        """
        Forward pass through the MedSigLIP classifier.
        
        Supports both single-view and multi-view inputs:
        - Single-view: pixel_values has shape [B, C, H, W]
        - Multi-view: pixel_values has shape [B, V, C, H, W] where V is number of views
        
        Args:
            pixel_values: Input tensor, either single-view or multi-view
            
        Returns:
            tuple: (logits, attention_maps)
                - logits: Classification logits [B, num_classes]
                - attention_maps: Attention maps from all transformer layers
        """
        # Validate input matches model configuration
        if len(pixel_values.shape) == 5 and not self.multi_view:
            raise ValueError("Model configured for single-view but received multi-view input")
        if len(pixel_values.shape) == 4 and self.multi_view:
            raise ValueError("Model configured for multi-view but received single-view input")
        
        batch_size = pixel_values.shape[0]
        
        # Apply input reshaping strategy
        pixel_values_reshaped, num_views = self.input_reshape_strategies[self.multi_view](pixel_values)
        
        # Process through vision model with attention maps
        vision_outputs = self.backbone.vision_model(
            pixel_values=pixel_values_reshaped,
            output_attentions=True,
            return_dict=True
        )
        
        # Extract features and normalize
        features = vision_outputs.pooler_output / vision_outputs.pooler_output.norm(dim=-1, keepdim=True)
        
        # Apply view fusion strategy
        fusion_strategy = self.view_fusion_strategies.get(getattr(self, 'view_fusion_type', None), self._single_view_fusion)
        fused_features = fusion_strategy(features, batch_size, getattr(self, 'num_views', 1))
        
        # Apply normalization strategy
        fused_features = self.normalization_strategies[self.multi_view](fused_features)
        
        # Classification
        logits = self.head(fused_features)
        
        # Handle attention maps
        attentions = vision_outputs.attentions
        stacked_attns = torch.stack(attentions, dim=0)
        stacked_attns = self.attention_reshape_strategies[self.multi_view](stacked_attns, batch_size, num_views)
        
        return logits, stacked_attns

if __name__ == "__main__":
    import os
    from transformers import AutoModel
    from dotenv import load_dotenv, find_dotenv
    from huggingface_hub import login
    load_dotenv(find_dotenv())
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)
    
    def unfreeze_layers(model, num_unfreeze_layers):
        # MedSigLIP has separate vision and text encoders, use vision config for unfreezing
        # Vision encoder has 27 layers (vision_model.encoder.layers.0 to .26)
        num_total_layers = model.backbone.config.vision_config.num_hidden_layers
        assert num_unfreeze_layers <= num_total_layers, "Number of unfreeze layers cannot be greater than the total number of layers"
        # First freeze all backbone parameters
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        # Then unfreeze the specified layers (vision encoder layers)
        for i in range(num_total_layers - 1, num_total_layers - num_unfreeze_layers - 1, -1):
            for name, param in model.backbone.named_parameters():
                if f"vision_model.encoder.layers.{i}" in name:
                    param.requires_grad = True
                    
    # Load pre-trained MedSigLIP backbone
    backbone = AutoModel.from_pretrained('google/medsiglip-448')
    
    # Test single-view
    model_single = MedSigClassifier(backbone, num_classes=10, multi_view=False)
    print("Single-view model created successfully")
    
    # Test multi-view
    model_multi = MedSigClassifier(backbone, num_classes=10, multi_view=True, num_views=4, view_fusion_type="mean")
    print("Multi-view model created successfully")
    
    # Test different fusion types
    model_weighted = MedSigClassifier(backbone, num_classes=10, multi_view=True, num_views=4, view_fusion_type="weighted_mean")
    model_adapter = MedSigClassifier(backbone, num_classes=10, multi_view=True, num_views=4, view_fusion_type="mlp_adapter")
    print("All fusion types created successfully")
    
    # Test unfreezing layers
    unfreeze_layers(model_multi, 2)
    for name, param in model_multi.named_parameters():
        if 'backbone' in name:
            if param.requires_grad:
                print(f"Parameter name: {name}")
        else:
            param.requires_grad = True
            print(f"Parameter name: {name}")
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model_multi.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model_multi.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # Test forward pass with attention maps
    # Create dummy input for single-view
    dummy_input_single = torch.randn(2, 3, 448, 448)  # [B, C, H, W]
    logits_single, attns_single = model_single(dummy_input_single)
    print(f"Single-view output shapes: logits {logits_single.shape}, attention maps {attns_single.shape}")
    
    # Create dummy input for multi-view
    dummy_input_multi = torch.randn(2, 4, 3, 448, 448)  # [B, V, C, H, W]
    logits_multi, attns_multi = model_multi(dummy_input_multi)
    print(f"Multi-view output shapes: logits {logits_multi.shape}, attention maps {attns_multi.shape}")
