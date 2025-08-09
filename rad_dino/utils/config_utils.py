import os
import logging
import yaml
from typing import Dict, Any
from pathlib import Path
from rad_dino.configs.config import *
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_configs(dataset_name: str, task: str) -> tuple[DataConfig, TrainConfig]:
    # Configuration settings
    data_config_path = os.path.join(CURR_DIR, "../configs/data_config.yaml")
    train_config_path = os.path.join(CURR_DIR, "../configs/train_config.yaml")
    
    # Load configurations
    with open(data_config_path, 'r') as file:
        data_config_raw = yaml.safe_load(file)
    with open(train_config_path, 'r') as file:
        train_config_raw = yaml.safe_load(file)
    
    # Select the appropriate dataset configuration
    data_config_raw = data_config_raw.get(dataset_name, None)
    if data_config_raw is None:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported")
    
    # Validate configurations
    try:
        if task in ["multilabel", "multiclass", "binary"]:
            data_config = ClassificationDataConfig(**data_config_raw)
        else:
            raise NotImplementedError(f"Task {task} is currently not supported.")
        train_config = TrainConfig(**train_config_raw)
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    return data_config, train_config

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model configurations from the YAML file.
    
    Returns:
        Dict containing model-specific configuration
    """
    current_dir = Path(__file__).parent
    configs_dir = current_dir.parent / "configs"
    model_config_path = configs_dir / "model_config.yaml"
    
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found: {model_config_path}")
    
    try:
        with open(model_config_path, 'r') as file:
            model_configs = yaml.safe_load(file)
        if "dinov2" in model_name: # For dinov2-base and dinov2-small, use the same dinov2 config
            model_name = "dinov2"
        return model_configs[model_name].copy()
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing model configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading model configuration file: {e}")