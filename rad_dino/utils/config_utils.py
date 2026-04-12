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

MODEL_REPOS: Dict[str, str] = {
    "rad-dino": "microsoft/rad-dino",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-large-reg": "facebook/dinov2-with-registers-large",
    "dinov3-small-plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "medsiglip": "google/medsiglip-448",
    "ark": "microsoft/swin-large-patch4-window12-384-in22k",
}


# Configuration settings
DATA_CONFIG_PATH = os.path.join(CURR_DIR, "../configs/data_config.yaml")
TRAIN_CONFIG_PATH = os.path.join(CURR_DIR, "../configs/train_config.yaml")

def get_available_datasets() -> list[str]:
    """Return the dataset names defined in ``data_config.yaml``."""
    with open(DATA_CONFIG_PATH, 'r') as file:
        return list(yaml.safe_load(file).keys())

def validate_dataset(dataset_name: str) -> None:
    """Raise ``ValueError`` if *dataset_name* is not in ``data_config.yaml``."""
    available = get_available_datasets()
    if dataset_name not in available:
        raise ValueError(
            f"Dataset '{dataset_name}' is not configured. "
            f"Available datasets (from data_config.yaml): {available}"
        )

def setup_configs(dataset_name: str, task: str) -> tuple[DataConfig, TrainConfig]:
    # Load configurations
    with open(DATA_CONFIG_PATH, 'r') as file:
        data_config_raw = yaml.safe_load(file)
    with open(TRAIN_CONFIG_PATH, 'r') as file:
        train_config_raw = yaml.safe_load(file)
    
    # Select the appropriate dataset configuration
    data_config_raw = data_config_raw.get(dataset_name, None)
    if data_config_raw is None:
        validate_dataset(dataset_name)
    
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
        elif "dinov3" in model_name: # For dinov3-small-plus, dinov3-base and dinov3-large, use the same dinov3 config
            model_name = "dinov3"
        return model_configs[model_name].copy()
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing model configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading model configuration file: {e}")