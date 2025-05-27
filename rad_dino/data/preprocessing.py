import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut
import matplotlib.pyplot as plt
import cv2
import glob
import os
import pandas as pd
import logging
from typing import Union
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def filter_subset_annot_labels(train_annot_path: str, test_annot_path: str, labels: Union[int, list[str]]):
    """A subset of class labels is selected, given the long-tailed distribution in the original dataset.
    
    Args:
        train_annot_path: Path to annotation csv file of training dataset
        test_annot_path: Path to annotation csv file of test dataset
        labels: the class labels to be filtered
    Returns:
        A list of the class labels considered for image classification task
    
    """
    df_train = pd.read_csv(train_annot_path)
    df_test  = pd.read_csv(test_annot_path)
    annot_dir = os.path.dirname(train_annot_path)
    df_train_annot = pd.read_csv(os.path.join(annot_dir, "annotations_train.csv"))
    df_test_annot = pd.read_csv(os.path.join(annot_dir, "annotations_test.csv"))
    if isinstance(labels, int):
        # find top-k classes in the training dataset
        class_labels = (
            df_train_annot['class_name']
            .value_counts()
            .head(8)
            .index
            .tolist()
        )
    else:
        class_labels = list(labels)
    # filter both train & test to only those classes
    df_train_filtered = df_train[ ["image_id"] + class_labels ]
    df_test_filtered = df_test[ ["image_id"] + class_labels ]
    df_train_annot_filtered = df_train_annot[df_train_annot['class_name'].isin(class_labels)].copy()
    df_test_annot_filtered  = df_test_annot[df_test_annot['class_name'].isin(class_labels)].copy()
    df_train_annot_filtered.reset_index(drop=True, inplace=True)
    df_test_annot_filtered.reset_index(drop=True,  inplace=True)
    
    # output the filtered annotated csv 
    train_out = os.path.join(annot_dir, "filtered_image_labels_train.csv")
    test_out  = os.path.join(annot_dir, "filtered_image_labels_test.csv")
    train_annot_out = os.path.join(annot_dir, "filtered_annotations_train.csv")
    test_annot_out  = os.path.join(annot_dir, "filtered_annotations_test.csv")
    df_train_filtered.to_csv(train_out, index=False)
    df_test_filtered.to_csv(test_out,  index=False)
    df_train_annot_filtered.to_csv(train_annot_out, index=False)
    df_test_annot_filtered.to_csv(test_annot_out,  index=False)
    logger.debug(f"Filtered csv are saved!")
    return class_labels

def dicom2array(path, voi_lut = True, fix_monochrome = True):
    """
    Read and process a DICOM X-ray image.

    Args:
        path (str): Path to the DICOM file.
        voi_lut (bool): Apply VOI LUT transformation if available. Defaults to True.
        fix_monochrome (bool): Invert image if PhotometricInterpretation is MONOCHROME1. Defaults to True.

    Returns:
        np.ndarray: Processed image as a uint8 array scaled to [0, 255].

    Raises:
        FileNotFoundError: If the DICOM file cannot be found.
        AttributeError: If required DICOM attributes (e.g., pixel_array) are missing.
        ValueError: If pixel data processing fails.
    """
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(path)

        # Check if pixel data exists
        if not hasattr(dicom, 'pixel_array'):
            raise AttributeError("DICOM file does not contain pixel data.")

        # Apply VOI LUT if requested and available to transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array

        # Fix inverted X-ray if MONOCHROME1
        if fix_monochrome and getattr(dicom, 'PhotometricInterpretation', None) == "MONOCHROME1":
            data = np.amax(data) - data

        # Normalize the image array to [0, 255]
        # Normalize the image array 
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8) 

        return data

    except FileNotFoundError:
        raise FileNotFoundError(f"DICOM file not found at: {path}")
    except Exception as e:
        raise ValueError(f"Error processing DICOM file: {str(e)}")

def plot_image(img, title="", figsize=(8,8), cmap=None):
    plt.figure(figsize=figsize)
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title, fontweight="bold")
    plt.axis(False)
    plt.show()  
    
def get_image_id(path):
    """ Function to return the image-id from a path """
    return path.rsplit("/", 1)[1].rsplit(".", 1)[0]

def draw_bboxes(img, tl, br, rgb, label="", label_location="tl", opacity=0.1, line_thickness=0):
    """ draw bounding boxes of the class labels in the image 
    
    Args:
        img
        tl: top-left
        br: bottom-right
        rgb: color
        label
        label_location
        
    Returns:
        img: annotated image 
    """
    rect = np.uint8(np.ones((br[1]-tl[1], br[0]-tl[0], 3))*rgb)
    sub_combo = cv2.addWeighted(img[tl[1]:br[1],tl[0]:br[0],:], 1-opacity, rect, opacity, 1.0)    
    img[tl[1]:br[1],tl[0]:br[0],:] = sub_combo

    if line_thickness>0:
        img = cv2.rectangle(img, tuple(tl), tuple(br), rgb, line_thickness)
        
    if label:
        # DEFAULTS
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.666
        FONT_THICKNESS = 3
        FONT_LINE_TYPE = cv2.LINE_AA
        
        if type(label)==str:
            LABEL = label.upper().replace(" ", "_")
        else:
            LABEL = f"CLASS_{label:02}"
        
        text_width, text_height = cv2.getTextSize(LABEL, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        
        label_origin = {"tl":tl, "br":br, "tr":(br[0],tl[1]), "bl":(tl[0],br[1])}[label_location]
        label_offset = {
            "tl":np.array([0, -10]), "br":np.array([-text_width, text_height+10]), 
            "tr":np.array([-text_width, -10]), "bl":np.array([0, text_height+10])
        }[label_location]
        img = cv2.putText(img, LABEL, tuple(label_origin+label_offset), 
                          FONT, FONT_SCALE, rgb, FONT_THICKNESS, FONT_LINE_TYPE)
    
    return img