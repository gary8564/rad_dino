from pydantic import BaseModel, Field
from typing import Optional, List

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

class DataConfig(BaseModel):
    data_root_folder: str = Field(..., description="Root folder containing the dataset")
    num_workers: int = Field(..., description="Number of workers for data loading")
    
class ClassificationDataConfig(DataConfig):
    pass

class RegressionDataConfig(DataConfig):
    pass

class OrdinalDataConfig(DataConfig):
    pass

class SegmentationDataConfig(DataConfig):
    pass

class TextGenerationDataConfig(DataConfig):
    pass
