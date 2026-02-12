import numpy as np
import logging
import pydicom
import matplotlib.pyplot as plt
import cv2
import os
from pydicom.pixels import apply_rescale, apply_voi_lut
from PIL import Image
from typing import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def create_symlinks_parallel(
    symlink_pairs: Sequence[tuple[str, str]],
    max_workers: int = 16,
    raise_on_missing: bool = False,
) -> None:
    """
    Create symlinks in parallel using a thread pool.

    Args:
        symlink_pairs: Sequence of (src, dst) path pairs.
        max_workers: Number of threads to use.
        raise_on_missing: If True, raise FileNotFoundError when a source file
            is missing. If False, log a warning and skip.
    """
    def _create_one(src: str, dst: str) -> str | None:
        if not os.path.exists(src):
            if raise_on_missing:
                raise FileNotFoundError(f"Source image not found: {src}")
            return f"Source image not found: {src}"
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_create_one, src, dst): src
            for src, dst in symlink_pairs
        }
        for future in as_completed(futures):
            warning = future.result()
            if warning:
                logger.warning(warning)

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
            data = dicom.pixel_array
            data = apply_rescale(data, dicom) 
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

def plot_image(img, title="", figsize=(8,8), cmap=None, visualize=True, output_path=None):
    """
    Plot radiology image.

    Args:
        img: image array
        title: title of the image
        figsize: figure size
        cmap: colormap
        visualize: whether to visualize the image
        output_path: path to save the image
    """
    plt.figure(figsize=figsize)
    if cmap:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title, fontweight="bold")
    plt.axis(False)
    if visualize:
        plt.show()
    if output_path is not None:
        plt.savefig(output_path)
    plt.close()
    
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

def uint16_to_uint8(img_path):
    """
    Convert a 16-bit grayscale image to 8-bit grayscale image.

    Args:
        img_path: path to the image

    Returns:
        Image: 8-bit grayscale image
    """
    img = Image.open(img_path)
    # Read as grayscale
    if img.mode in ("I;16", "I"):
        img_arr = np.array(img, dtype=np.float32)             # 16-bit -> float
        img_arr = img_arr - img_arr.min()
        denom = max(1e-6, img_arr.max())
        img_arr = (img_arr / denom * 255.0).astype(np.uint8)  # min-max to [0,255]
        return Image.fromarray(img_arr, mode="L")
    else:
        return img.convert("L")  # already 8-bit; ensure grayscale