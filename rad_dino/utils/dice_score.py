import numpy as np
import os
BBOX_CONFIG_TYPE = {
    'VinDr-CXR': ['x_min', 'y_min', 'x_max', 'y_max'],
    'VinDr-Mammo': ['xmin', 'ymin', 'xmax', 'ymax'],
    'RSNA-Pneumonia': ['x', 'y', 'width', 'height'],
}

def _get_bbox_by_image_id(image_id, df_annot, dataset_name):
    """
    Get the bounding box from the image id.
    
    Args:
        image_id: The image id.
        df_annot: The dataframe with the annotations.
        dataset_name: The name of the dataset.
        
    Returns:
        Bounding box.
    """
    df_image_id = df_annot[df_annot['image_id'] == image_id]
    bbox = df_image_id[BBOX_CONFIG_TYPE[dataset_name]].to_numpy()
    return bbox

def _get_attention_map_by_image_id(image_id, root_folder, dataset_name):
    """
    Get the attention map from the image id.
    
    Args:
        image_id: The image id.
    """
    path = os.path.join(root_folder, dataset_name, image_id)
    

def _bbox_to_binary_mask(bbox, shape):
    """
    Convert a bounding box to a binary mask.
    """
    mask = np.zeros(shape)
    mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
    return mask

def compute_dice_score(y_true, y_pred):
    """
    Compute the Dice score between two binary masks.
    """
    y_true = y_true.astype(np.bool)
    y_pred = y_pred.astype(np.bool)
    
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return 2 * intersection / union