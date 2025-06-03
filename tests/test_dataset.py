import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from torchvision import transforms
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
import shutil

from rad_dino.data.dataset import RadImageClassificationDataset

def create_mock_dicom():
    """Create a mock DICOM file with proper metadata and pixel data."""
    # Create file meta info
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    
    # Create the dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Add required DICOM attributes
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    
    # Create and add pixel data
    pixel_array = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    ds.Rows, ds.Columns = pixel_array.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()
    
    return ds

class TestRadImageClassificationDataset(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.root = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.root, "images", "train"), exist_ok=True)
        
        # Create mock DICOM files
        for i in range(5):
            ds = create_mock_dicom()
            ds.save_as(os.path.join(self.root, "images", "train", f"img{i}.dcm"))
        
        # Create binary classification data
        self.binary_train = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(5)],
            "label": [1, 0, 1, 0, 1]
        }).set_index("image_id")
        
        # Create multilabel classification data
        self.multilabel_train = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(5)],
            "label1": [1, 0, 1, 0, 1],
            "label2": [0, 1, 1, 0, 0]
        }).set_index("image_id")

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.root)

    def test_dataset_initialization(self):
        """Test dataset initialization and basic functionality."""
        # Test invalid parameters
        with self.assertRaises(AttributeError):
            RadImageClassificationDataset(self.root, split="invalid", task="binary")
        with self.assertRaises(AttributeError):
            RadImageClassificationDataset(self.root, split="train", task="invalid_task")

        # Test binary classification
        self.binary_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="binary")
        self.assertEqual(len(dataset), 5)
        img, target, img_id = dataset[0]
        self.assertIsInstance(img, np.ndarray)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(img.shape, (224, 224))
        self.assertEqual(target.shape, (1,))
        self.assertTrue(torch.all((target == 0) | (target == 1)))

        # Test multilabel classification
        self.multilabel_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="multilabel")
        img, target, img_id = dataset[0]
        self.assertEqual(target.shape, (2,))

    def test_transforms(self):
        """Test that transforms are correctly applied."""
        # Ensure we're using binary classification data
        self.binary_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert single channel to 3 channels
        ])
        dataset = RadImageClassificationDataset(
            self.root, split="train", task="binary", transform=transform
        )
        img, _, _ = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (3, 224, 224))

if __name__ == "__main__":
    unittest.main() 