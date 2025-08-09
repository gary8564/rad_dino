import torch
from torch.utils.data import Dataset
import logging
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def get_class_weights(task: str, dataset: Dataset):
    """
    Calculate pos_weight or class weights for weighted loss functions.
    
    For binary classification:
        Returns pos_weight for BCEWithLogitsLoss using the formula:
        pos_weight = num_negative_samples / num_positive_samples
        
    For multilabel classification:
        Returns pos_weight tensor for each class using the formula:
        pos_weight[i] = num_negative_samples[i] / num_positive_samples[i]
        
    For multiclass classification:
        Returns class weights using the formula:
        weight_for_class_i = total_samples / (num_samples_in_class_i)
    
    Args:
        task: Type of classification task
        dataset: Dataset to calculate weights from
        
    Returns:
        torch.Tensor or None: Weights for the loss function
    """
    if task == "binary":
        # For binary classification, calculate pos_weight
        class_counts = dataset.df["label"].value_counts().to_dict()
        # Get counts for negative (0) and positive (1) classes
        num_negative = class_counts.get(0, 0)
        num_positive = class_counts.get(1, 0)
        if num_positive == 0:
            logger.warning("No positive samples found in dataset for binary classification")
            return None
        # Calculate pos_weight = num_negative / num_positive
        pos_weight = torch.tensor([num_negative / num_positive], dtype=torch.float32)
        return pos_weight
    elif task == "multilabel":
        # For multilabel classification, calculate pos_weight for each class
        # Assuming dataset.df contains binary columns for each class
        pos_weights = []
        for col in dataset.df.columns:
            if col != "image_id":  # Skip non-label columns
                class_counts = dataset.df[col].value_counts().to_dict()
                num_negative = class_counts.get(0, 0)
                num_positive = class_counts.get(1, 0)
                
                if num_positive == 0:
                    logger.warning(f"No positive samples found for class {col}")
                    pos_weight = 1.0  # Default weight
                else:
                    pos_weight = num_negative / num_positive
                    
                pos_weights.append(pos_weight)        
        if pos_weights:
            return torch.tensor(pos_weights, dtype=torch.float32)
        else:
            logger.warning("No pos_weights calculated for multilabel classification")
            return None
    elif task == "multiclass":
        # For multiclass classification, calculate class weights
        # Using formula: weight_for_class_i = total_samples / (num_samples_in_class_i)
        class_weights = []
        total_samples = len(dataset.df)
        # Get all columns except image_id
        class_columns = [col for col in dataset.df.columns if col != "image_id"]
        # Calculate weight for each class column
        for class_idx, col in enumerate(class_columns):
            # Count positive samples for this class (assuming binary encoding: 0/1)
            class_counts = dataset.df[col].value_counts().to_dict()
            num_positive = class_counts.get(1, 0)
            if num_positive == 0:
                logger.warning(f"No positive samples found for class {col}, using default weight 1.0")
                weight = 1.0
            else:
                weight = total_samples / num_positive
            class_weights.append(weight)
        return torch.tensor(class_weights, dtype=torch.float32)
    else:
        raise ValueError(f"Task {task} is not supported for weight calculation") 
    
