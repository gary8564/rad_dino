from pydantic import BaseModel, Field
from typing import Optional, List, Any
from dataclasses import dataclass
import onnxruntime
import torch
from rad_dino.models.dino import DinoClassifier

# ------------ multi-view config ------------
class MultiViewConfig(BaseModel):
    """Configuration for multi-view processing"""
    num_views: int = Field(default=4, description="Number of views to process")
    view_fusion_type: str = Field(default="mean", description="Fusion strategy: mean, weighted_mean, or mlp_adapter")
    adapter_dim: Optional[int] = Field(default=None, description="Hidden dimension for MLP adapters")
    view_fusion_hidden_dim: Optional[int] = Field(default=None, description="Hidden dimension for fusion MLP")
    data_root_folder_multi_view: Optional[str] = Field(default=None, description="Data root folder for multi-view data")

# ------------ data config ------------
class DataConfig(BaseModel):
    data_root_folder: str = Field(..., description="Root folder containing the dataset")
    num_workers: int = Field(..., description="Number of workers for data loading")
    multi_view: Optional[MultiViewConfig] = Field(default=None, description="Multi-view configuration")
    
    def get_data_root_folder(self, use_multi_view: bool = False) -> str:
        """Get the appropriate data root folder based on multi-view setting"""
        if use_multi_view and self.multi_view and self.multi_view.data_root_folder_multi_view:
            return self.multi_view.data_root_folder_multi_view
        return self.data_root_folder
    
    def get_multi_view_config(self, use_multi_view: bool = False) -> Optional[MultiViewConfig]:
        """Get multi-view configuration if enabled"""
        if use_multi_view and self.multi_view:
            return self.multi_view
        return None

class ClassificationDataConfig(DataConfig):
    pass

class RegressionDataConfig(DataConfig):
    pass

# ------------ training config ------------
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

# ------------ inference config ------------
@dataclass
class InferenceConfig:
    """Configuration class for inference parameters"""
    task: str
    data: str
    model: str
    model_path: str
    output_path: str
    batch_size: int = 16
    fusion_type: str = 'mean'
    multi_view: bool = False
    optimize_compute: bool = False
    show_attention: bool = False
    show_lrp: bool = False
    show_gradcam: bool = False
    attention_threshold: Optional[float] = None
    save_heads: Optional[str] = None
    compute_rollout: bool = False

@dataclass
class ModelWrapper:
    """Wrapper for model information"""
    model_type: str  # 'onnx' or 'pytorch'
    session: Optional[onnxruntime.InferenceSession] = None
    model: Optional[DinoClassifier] = None
    input_name: Optional[str] = None
    output_names: Optional[List[str]] = None
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
    lrp: Optional[str] = None
