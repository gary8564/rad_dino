from typing import List

# Dataset-specific label mappings
MULTI_CLASS_DATASET_LABEL_MAPPINGS = {
    "VinDr-Mammo": {
        # Map integer indices to BIRADS labels for visualization
        0: "BIRADS_1",
        1: "BIRADS_2", 
        2: "BIRADS_3",
        3: "BIRADS_4",
        4: "BIRADS_5"
    }
    # Add more datasets here as needed
    # "New-Dataset": {
    #     0: "Class_A", 
    #     1: "Class_B", ...
    # }
}

def class_labels_mapping(dataset_name: str, class_labels: List) -> List:
    """
    Prepare class labels for metrics computation based on dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "VinDr-Mammo", "VinDr-CXR", "RSNA-Pneumonia")
        class_labels: Original class labels (integer indices for multiclass)
        
    Returns:
        List of formatted class labels for metrics visualization
    """
    if dataset_name in MULTI_CLASS_DATASET_LABEL_MAPPINGS:
        mapping = MULTI_CLASS_DATASET_LABEL_MAPPINGS[dataset_name]
        return [mapping.get(i, f"Class_{i}") for i in class_labels]
    else:
        return class_labels
