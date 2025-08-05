import numpy as np
import logging
import os
import cv2
import supervision as sv
import matplotlib.pyplot as plt
from PIL import Image
from rad_dino.utils.preprocessing_utils import dicom2array
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

BBOX_TYPE_CONFIG = {
    'VinDr-CXR': ['x_min', 'y_min', 'x_max', 'y_max'],
    'VinDr-Mammo': ['xmin', 'ymin', 'xmax', 'ymax'],
    'RSNA-Pneumonia': ['x', 'y', 'width', 'height'],
}

IMAGE_ID_CONFIG = {
    'VinDr-CXR': 'image_id',
    'VinDr-Mammo': 'image_id',
    'RSNA-Pneumonia': 'patientId',
}

CLASS_LABES_CONFIG = {
    'VinDr-CXR': ['Architectural Distortion', 
                  'Asymmetry', 
                  'Mass', 
                  'Skin Thickening', 
                  'Suspicious Calcification', 
                  'Suspicious Lymph Node'],
    'VinDr-Mammo': ['Architectural Distortion', 
                    'Asymmetry', 
                    'Focal Asymmetry', 
                    'Global Asymmetry', 
                    'Mass', 
                    'Nipple Retraction', 
                    'No Finding', 
                    'Skin Retraction', 
                    'Skin Thickening', 
                    'Suspicious Calcification', 
                    'Suspicious Lymph Node'],
    'RSNA-Pneumonia': ['Lung Opacity'],
}

def _convert_xywh_to_xyxy(xywh):
    """
    Convert bounding box from [x, y, width, height] to [x_min, y_min, x_max, y_max].
    """
    if len(xywh.shape) == 1:
        # Single bounding box
        x, y, w, h = xywh
        xyxy = [int(x), int(y), int(x + w), int(y + h)]
        return xyxy
    else:
        # Array of bounding boxes
        x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
        xyxy = np.column_stack([x, y, x + w, y + h]).astype(int)
        return xyxy

def _get_image_by_image_id(image_id, root_folder, df_annot, dataset_name):
    """
    Get the image from the image id.
    """
    df_img = df_annot[df_annot[IMAGE_ID_CONFIG[dataset_name]] == image_id]
    if dataset_name == 'RSNA-Pneumonia':
        img = cv2.cvtColor(dicom2array(os.path.join(root_folder, "stage_2_train_images", f"{image_id}.dcm")), cv2.COLOR_GRAY2RGB)
        return img, df_img
    elif dataset_name == "VinDr-CXR":
        img = cv2.cvtColor(dicom2array(os.path.join(root_folder, "train", f"{image_id}.dicom")),cv2.COLOR_GRAY2RGB)
        return img, df_img
    elif dataset_name == "VinDr-Mammo":
        # Get study_id from annotation dataframe
        if len(df_img) == 0:
            raise ValueError(f"No annotation found for image_id: {image_id}")
        study_id = df_img['study_id'].iloc[0]
        img = cv2.cvtColor(dicom2array(os.path.join(root_folder, f"images/{study_id}/{image_id}.dicom")),cv2.COLOR_GRAY2RGB)
        return img, df_img
    else:
        raise ValueError(f"Dataset name {dataset_name} not supported")

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
    df_image_id = df_annot[df_annot[IMAGE_ID_CONFIG[dataset_name]] == image_id]
    if len(df_image_id) == 0:
        return None
    bbox = df_image_id[BBOX_TYPE_CONFIG[dataset_name]].to_numpy()
    return bbox

def _resize_bbox(bbox, raw_img_size, attn_map_size):
    """Rescale the bounding box from raw image size to be matched with attention map size."""
    if len(raw_img_size) > 2:
        raise ValueError("Raw image size should be specified as (height, width).")
    if len(attn_map_size) > 2:
        raise ValueError("Attention map size should be specified as (height, width).")
    raw_img_height, raw_img_width = raw_img_size
    attn_map_height, attn_map_width = attn_map_size
    scale_x = attn_map_width / raw_img_width
    scale_y = attn_map_height / raw_img_height
    
    if len(bbox.shape) == 1:
        # Single bounding box
        x_min, y_min, x_max, y_max = bbox
        x_min = int(round(x_min * scale_x))
        y_min = int(round(y_min * scale_y))
        x_max = int(round(x_max * scale_x))
        y_max = int(round(y_max * scale_y))

        # Clamp to valid range [0, attn_W−1] × [0, attn_H−1]
        x_min = max(0, min(attn_map_width - 1, x_min))
        y_min = max(0, min(attn_map_height - 1, y_min))
        x_max = max(0, min(attn_map_width - 1, x_max))
        y_max = max(0, min(attn_map_height - 1, y_max))
        return x_min, y_min, x_max, y_max
    else:
        # Array of bounding boxes
        x_min, y_min, x_max, y_max = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        x_min = np.round(x_min * scale_x).astype(int)
        y_min = np.round(y_min * scale_y).astype(int)
        x_max = np.round(x_max * scale_x).astype(int)
        y_max = np.round(y_max * scale_y).astype(int)

        # Clamp to valid range [0, attn_W−1] × [0, attn_H−1]
        x_min = np.clip(x_min, 0, attn_map_width - 1)
        y_min = np.clip(y_min, 0, attn_map_height - 1)
        x_max = np.clip(x_max, 0, attn_map_width - 1)
        y_max = np.clip(y_max, 0, attn_map_height - 1)
        return np.column_stack([x_min, y_min, x_max, y_max])

def _get_attention_map_by_image_id(image_id,
                                   root_folder,
                                   attn_map_method="raw",
                                   attn_fusion_method="mean",
                                   display_attn_map=False):
    """
    Get the attention map from the image id.
    
    Args:
        image_id: The image id.
        root_folder: Root folder containing attention maps.
        attn_map_method: The method to get attention map. 
                         ["raw", "rollout", "lrp"]
                         "raw" means raw attention map;
                         "rollout" means attention rollout;
                         "lrp" means Layer-wise Relevance Propagation.
        attn_fusion_method: The fusion method of attention map results.
                            ["mean", "max", "min"]
        display_attn_map: Whether to display the attention map. 
                          If True, the attention map will be visualized.
    Returns:
        Attention map as numpy array.
    """
    assert attn_fusion_method in ["mean", "max", "min"], "Invalid attention map fusion method"
    
    if attn_map_method == "raw":
        attention_map_filename = f"attn_head_{attn_fusion_method}_fused.png"
        attention_map_title = f"Raw Attention ({attn_fusion_method} fused)"
    elif attn_map_method == "rollout":
        attention_map_filename = f"{attn_map_method}_{attn_fusion_method}.png"
        attention_map_title = f"Attention Rollout ({attn_fusion_method} fused)"
    elif attn_map_method == "lrp":
        attention_map_filename = f"{attn_map_method}_{attn_fusion_method}.png"
        attention_map_title = f"TransLRP ({attn_fusion_method} fused)"
    else:
        raise AttributeError(f"Invalid attention map method: {attn_map_method}")
    
    attention_map_path = os.path.join(root_folder, 
                                      f"attention_{image_id}",
                                      attention_map_filename)

    # Load the attention map
    attention_map = np.array(Image.open(attention_map_path).convert('L'))
    if display_attn_map:
        plt.imshow(attention_map)
        plt.title(attention_map_title)
        plt.show()
    return attention_map

def _bbox_to_binary_mask(bbox, img_size, display_bbox_mask=False):
    """
    Convert a bounding box to a binary mask.
    
    Args:
        bbox: Bounding box coordinates in xyxy format (single box or array of boxes).
        img_size: Shape of the output mask (height, width).
        display_bbox_mask: Whether to display the bounding box binary mask.
        
    Returns:
        Binary mask.
    """
    bbox_mask = np.zeros(img_size, dtype=np.uint8)
    
    if len(bbox.shape) == 1:
        # Single bounding box
        x_min, y_min, x_max, y_max = bbox
        
        # Ensure coordinates are within bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_size[1] - 1, x_max)
        y_max = min(img_size[0] - 1, y_max)
        
        bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1
    else:
        # Array of bounding boxes
        for box in bbox:
            x_min, y_min, x_max, y_max = box
            
            # Ensure coordinates are within bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_size[1] - 1, x_max)
            y_max = min(img_size[0] - 1, y_max)
            
            bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1
    if display_bbox_mask:
        plt.imshow(bbox_mask)
        plt.title("Bounding Box Binary Mask")
        plt.show()
    return bbox_mask

def _attention_map_to_binary_mask(attention_map, threshold=0.5, display_attn_mask=False):
    """
    Convert attention map to binary mask using thresholding.
    
    Args:
        attention_map: Attention map as numpy array.
        threshold: Threshold value for binarization.
        display_attn_mask: Whether to display the attention map binary mask.
        
    Returns:
        Binary mask.
    """
    # Normalize attention map to [0, 1] if needed
    if attention_map.max() > 1.0:
        attention_map = attention_map / attention_map.max()
    
    # Apply threshold
    binary_mask = attention_map >= threshold
    if display_attn_mask:
        plt.imshow(binary_mask)
        plt.title("Attention Map Binary Mask")
        plt.show()
    return binary_mask.astype(np.uint8)

def _compute_dice_score(y_true, y_pred):
    """
    Compute the Dice score between two binary masks.
    
    Args:
        y_true: Ground truth binary mask.
        y_pred: Predicted binary mask.
        
    Returns:
        Dice score.
    """
    y_true = y_true.astype(np.bool)
    y_pred = y_pred.astype(np.bool)
    
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    
    if union == 0:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    
    return 2 * intersection / union

def plot_annotated_bbox(image_id, root_folder, df_annot, dataset_name):
    """
    Plot the annotated bounding box on the image.
    """
    img, df_img = _get_image_by_image_id(image_id, root_folder, df_annot, dataset_name)
    bbox = _get_bbox_by_image_id(image_id, df_annot, dataset_name)
    
    if bbox is None:
        return None
    class_names = CLASS_LABES_CONFIG[dataset_name]
    if dataset_name == "RSNA-Pneumonia":
        xyxy = sv.xywh_to_xyxy(xywh=bbox) # convert to xyxy
        class_ids = np.array(df_img.Target.to_list()).astype(int)
        class_names = class_names * len(class_ids)
    elif dataset_name == "VinDr-CXR":
        xyxy = bbox
        class2idx = {c: i for i, c in enumerate(class_names)}
        class_ids = np.array([class2idx[cls] for cls in class_names]).astype(int)
    elif dataset_name == "VinDr-Mammo":
        xyxy = bbox
        class2idx = {c: i for i, c in enumerate(class_names)}
        class_ids = np.array([class2idx[cls] for cls in class_names]).astype(int)
    
    confidences = np.ones(len(xyxy), dtype=float)  # no scores in EDA, so set to 1.0

    # Create the Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        class_id=class_ids,
        confidence=confidences,
        data={"class_name": class_names}
    )

    # Annotate
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=2.5, text_thickness=3)

    # draw boxes and labels
    annotated = box_annotator.annotate(scene=img.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=class_names)
    sv.plot_image(annotated)

def compute_dice_score_per_image(image_id, 
                                 df_annot, 
                                 attn_map_folder, 
                                 image_root_folder,
                                 dataset_name, 
                                 attention_threshold=0.5,
                                 attn_map_method="raw",
                                 attn_fusion_method="mean",
                                 display_attn_map=False,
                                 display_binary_mask=False):
    """
    Compute dice score for a single image.
    
    Args:
        image_id: Image identifier.
        df_annot: DataFrame containing annotations.
        attn_map_folder: Root folder containing attention maps.
        image_root_folder: Root folder containing original images.
        dataset_name: Name of the dataset.
        attention_threshold: Threshold for binarizing attention map.
        attn_map_method: The method to get attention map. ["raw", "rollout", "lrp"]
        attn_fusion_method: The fusion method of attention map results. ["mean", "max", "min"]
        display_attn_map: Whether to display the attention map. 
                          If True, the attention map will be visualized.
        display_binary_mask: Whether to display the binary mask. 
                             If True, the binary mask will be visualized.
                             
    Returns:
        Dice score for the image, or None if no annotation found.
    """
    # Get bounding box
    bbox = _get_bbox_by_image_id(image_id, df_annot, dataset_name)
    if bbox is None:
        return None
    
    try:
        # Get attention map
        attention_map = _get_attention_map_by_image_id(image_id, 
                                                       attn_map_folder, 
                                                       attn_map_method, 
                                                       attn_fusion_method,
                                                       display_attn_map)
        
        # Get image to determine original dimensions
        img, _ = _get_image_by_image_id(image_id, image_root_folder, df_annot, dataset_name)
        raw_img_size = (img.shape[0], img.shape[1])  # (height, width)
        
        # Convert bbox from xywh to xyxy if needed
        if dataset_name == "RSNA-Pneumonia":
            bbox = _convert_xywh_to_xyxy(bbox)
        
        # Resize bbox to match attention map shape
        bbox = _resize_bbox(bbox, raw_img_size, (attention_map.shape[0], attention_map.shape[1]))
        
        # Convert bbox to binary mask
        bbox_mask = _bbox_to_binary_mask(bbox, attention_map.shape, display_binary_mask)
        
        # Convert attention map to binary mask
        attention_mask = _attention_map_to_binary_mask(attention_map, attention_threshold, display_binary_mask)
        
        # Compute Dice score
        dice_score = _compute_dice_score(bbox_mask, attention_mask)
        
        return dice_score
        
    except Exception as e:
        raise RuntimeError(f"Error computing Dice score for image_id {image_id}: {str(e)}")
