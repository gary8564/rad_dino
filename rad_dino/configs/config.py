from pydantic import BaseModel, Field
from typing import Optional, Any
from dataclasses import dataclass
import torch
from rad_dino.models.base import BaseClassifier

class MultiViewConfig(BaseModel):
    """Configuration for multi-view processing"""
    num_views: int = Field(default=4, description="Number of views to process")
    view_fusion_type: str = Field(default="mean", description="Fusion strategy: mean, weighted_mean, or mlp_adapter")
    adapter_dim: Optional[int] = Field(default=None, description="Hidden dimension for MLP adapters")
    view_fusion_hidden_dim: Optional[int] = Field(default=None, description="Hidden dimension for fusion MLP")

class DataConfig(BaseModel):
    data_root_folder: str = Field(..., description="Root folder containing the dataset")
    num_workers: int = Field(..., description="Number of workers for data loading")
    multi_view: Optional[MultiViewConfig] = Field(default=None, description="Multi-view configuration")

    @property
    def is_multi_view(self) -> bool:
        """Auto-detect whether this dataset requires multi-view processing from the config."""
        return self.multi_view is not None

class ClassificationDataConfig(DataConfig):
    pass

class RegressionDataConfig(DataConfig):
    pass

# Training config
class OptimizerConfig(BaseModel):
    base_lr: float = Field(..., description="Base learning rate")
    weight_decay: float = Field(..., description="Weight decay for optimizer")

class LRSchedulerConfig(BaseModel):
    warmup_ratio: float = Field(..., description="Ratio of warmup steps to total steps")

class EarlyStoppingConfig(BaseModel):
    patience: int = Field(..., description="Number of epochs to wait before early stopping")
    min_delta: float = Field(default=0.0, description="Minimum change in metric to qualify as improvement")
    mode: str = Field(default="max", description="Metric optimization mode ('min' or 'max')")

class TrainConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for training")
    epochs: int = Field(..., description="Number of training epochs")
    optim: OptimizerConfig
    lr_scheduler: Optional[LRSchedulerConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None

# Inference config
@dataclass
class InferenceConfig:
    """Configuration class for inference parameters"""
    task: str
    data: str
    model: str
    model_path: str
    output_path: str
    batch_size: int = 16
    optimize_compute: bool = False
    compile: bool = False
    show_attention: bool = False
    show_gradcam: bool = False
    show_feature_maps: bool = False
    attention_threshold: Optional[float] = None
    save_heads: Optional[str] = None
    compute_rollout: bool = False
    compute_gradient_rollout: bool = False
    max_visualization_samples: int = 24
    min_positive_visualization_labels: int = 20
    visualization_sample_ids: Optional[str] = None
    medimageinsight_path: Optional[str] = None

@dataclass
class ModelWrapper:
    """Wrapper for model information"""
    model: Optional[BaseClassifier] = None
    config: Optional[Any] = None
    device: Optional[torch.device] = None
    multi_view: bool = False

@dataclass
class OutputPaths:
    """Output directory paths"""
    base: str
    figs: str
    table: str
    gradcam: Optional[str] = None
    attention: Optional[str] = None
    feature_maps: Optional[str] = None
