import unittest
import tempfile
import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import shutil

from rad_dino.data.dataset import RadImageClassificationDataset


def create_mock_png(filepath: str):
    """Create a mock PNG image file that SimpleITK can read."""
    # Create a random grayscale image with variation (min != max for normalization)
    pixel_array = np.random.randint(10, 245, (256, 256), dtype=np.uint8)
    pixel_array[0, 0] = 0
    pixel_array[0, 1] = 255
    img = Image.fromarray(pixel_array, mode='L')
    img.save(filepath)


class TestRadImageClassificationDataset(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.root = tempfile.mkdtemp(dir="/tmp")
        os.makedirs(os.path.join(self.root, "images", "train"), exist_ok=True)
        
        # Create mock PNG files for single-view
        for i in range(5):
            create_mock_png(os.path.join(self.root, "images", "train", f"img{i}.png"))
        
        # Create multi-view directory structure
        os.makedirs(os.path.join(self.root, "images", "train", "study0"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "images", "train", "study1"), exist_ok=True)
        
        # Create mock PNG files for multi-view (4 views per study)
        view_files = ['L_CC.png', 'L_MLO.png', 'R_CC.png', 'R_MLO.png']
        for study_id in ['study0', 'study1']:
            for view_file in view_files:
                create_mock_png(os.path.join(self.root, "images", "train", study_id, view_file))
        
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
        
        # Create multi-view classification data
        self.multiview_train = pd.DataFrame({
            "study_id": ["study0", "study1"],
            "label1": [1, 0],
            "label2": [0, 1]
        }).set_index("study_id")

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.root)

    def test_dataset_initialization(self):
        """Test dataset initialization and basic functionality."""
        # Test invalid parameters
        with self.assertRaises(AttributeError):
            RadImageClassificationDataset(self.root, split="invalid", task="binary", model_name="rad-dino")
        with self.assertRaises(AttributeError):
            RadImageClassificationDataset(self.root, split="train", task="invalid_task", model_name="rad-dino")
        with self.assertRaises(AttributeError):
            RadImageClassificationDataset(self.root, split="train", task="binary")  # No model_name when transform=None

        # Test binary classification
        self.binary_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="binary", model_name="rad-dino")
        self.assertEqual(len(dataset), 5)
        img, target, img_id = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)
        self.assertEqual(img.shape, (3, 518, 518))
        self.assertEqual(target.shape, (1,))  # [1] for BCEWithLogitsLoss
        self.assertTrue(torch.all((target == 0) | (target == 1)))

        # Test multilabel classification
        self.multilabel_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="multilabel", model_name="rad-dino")
        img, target, img_id = dataset[0]
        self.assertEqual(target.shape, (2,))

    def test_multi_view_dataset(self):
        """Test multi-view dataset functionality."""
        # Test multi-view multilabel classification
        self.multiview_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(
            self.root, split="train", task="multilabel", multi_view=True, model_name="rad-dino"
        )
        
        self.assertEqual(len(dataset), 2)  # 2 studies
        
        # Test multi-view data loading
        img, target, study_id = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (4, 3, 518, 518))
        self.assertEqual(target.shape, (2,))  # 2 labels
        self.assertIn(study_id, ["study0", "study1"])
        
        # Test that all 4 views are loaded
        for view_idx in range(4):
            view_img = img[view_idx]
            self.assertEqual(view_img.shape, (3, 518, 518))

    def test_multi_view_with_transforms(self):
        """Test multi-view dataset with transforms."""
        self.multiview_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        dataset = RadImageClassificationDataset(
            self.root, split="train", task="multilabel", 
            transform=transform, multi_view=True
        )
        
        img, target, study_id = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        # When transforms are applied, we expect the exact shape from transforms
        self.assertEqual(img.shape, (4, 3, 224, 224))
        self.assertEqual(target.shape, (2,))

    def test_multiclass_classification(self):
        """Test multiclass classification (e.g., BIRAD classification)."""
        # Create multiclass data (e.g., BIRAD 0-5)
        multiclass_train = pd.DataFrame({
            "image_id": [f"img{i}" for i in range(5)],
            "label": [0, 1, 2, 3, 4]  # BIRAD categories
        }).set_index("image_id")
        
        multiclass_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        
        dataset = RadImageClassificationDataset(
            self.root, split="train", task="multiclass", model_name="rad-dino"
        )
        
        self.assertEqual(len(dataset), 5)
        img, target, img_id = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (3, 518, 518))
        self.assertEqual(target.shape, ())  # Scalar tensor for multiclass classification
        self.assertTrue(torch.all((target >= 0) & (target <= 4)))  # BIRAD 0-4

    def test_multiclass_multi_view(self):
        """Test multiclass classification with multi-view data."""
        # Create multiclass multi-view data
        multiclass_multiview_train = pd.DataFrame({
            "study_id": ["study0", "study1"],
            "label": [2, 3]  # BIRAD categories
        }).set_index("study_id")
        
        multiclass_multiview_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        
        dataset = RadImageClassificationDataset(
            self.root, split="train", task="multiclass",
            model_name="rad-dino", multi_view=True
        )
        
        self.assertEqual(len(dataset), 2)
        img, target, study_id = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (4, 3, 518, 518))
        self.assertEqual(target.shape, ())  # Scalar tensor for multiclass classification

    def test_transforms(self):
        """Test that transforms are correctly applied."""
        self.binary_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        dataset = RadImageClassificationDataset(
            self.root, split="train", task="binary", transform=transform
        )
        img, _, _ = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (3, 224, 224))

    def test_model_name_and_transform_warning(self):
        """Test that a warning is issued when both model_name and transform are specified."""
        self.binary_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # This should issue a warning but not raise an error
        with self.assertLogs(level='WARNING') as log:
            dataset = RadImageClassificationDataset(
                self.root, split="train", task="binary", 
                transform=transform, model_name="rad-dino"
            )
        
        # Check that the warning was logged
        self.assertTrue(any("model_name" in record.message for record in log.records))
        
        # Should still work correctly
        self.assertEqual(len(dataset), 5)
        img, target, img_id = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape, (3, 224, 224))  # Transform should take priority

    def test_multi_view_missing_files(self):
        """Test that multi-view dataset raises error for missing view files."""
        self.multiview_train.to_csv(os.path.join(self.root, "train_labels.csv"))
    
        # Remove one view file to test error handling
        os.remove(os.path.join(self.root, "images", "train", "study0", "L_CC.png"))
    
        dataset = RadImageClassificationDataset(
            self.root, split="train", task="multilabel", multi_view=True, model_name="rad-dino"
        )
        
        with self.assertRaises(FileNotFoundError):
            _ = dataset[0]  # This should raise the error when trying to load the missing file

    def test_dataset_labels_property(self):
        """Test that dataset correctly exposes labels property."""
        self.multilabel_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="multilabel", model_name="rad-dino")
        self.assertEqual(dataset.labels, ["label1", "label2"])

    def test_dataset_len(self):
        """Test dataset length for different configurations."""
        # Single-view dataset
        self.binary_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="binary", model_name="rad-dino")
        self.assertEqual(len(dataset), 5)
        
        # Multi-view dataset
        self.multiview_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="multilabel", multi_view=True, model_name="rad-dino")
        self.assertEqual(len(dataset), 2)

    def test_dataset_getitem_consistency(self):
        """Test that dataset returns consistent data types and shapes."""
        self.binary_train.to_csv(os.path.join(self.root, "train_labels.csv"))
        dataset = RadImageClassificationDataset(self.root, split="train", task="binary", model_name="rad-dino")
        
        # Test consistency across multiple calls
        for i in range(len(dataset)):
            img, target, img_id = dataset[i]
            self.assertIsInstance(img, torch.Tensor)
            self.assertIsInstance(target, torch.Tensor)
            self.assertEqual(img.shape, (3, 518, 518))
            self.assertEqual(target.shape, (1,))  # [1] for BCEWithLogitsLoss
            self.assertIsInstance(img_id, str)

if __name__ == "__main__":
    unittest.main()
