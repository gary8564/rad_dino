import unittest
import unittest.mock
import torch
import numpy as np
from rad_dino.utils.data_utils import collate_fn

class TestDataUtils(unittest.TestCase):
    
    def test_collate_fn_single_view(self):
        """Test collate function with single-view data."""
        # Create batch of single-view samples
        batch = [
            (torch.randn(3, 224, 224), torch.tensor([1.0]), "img1"),
            (torch.randn(3, 224, 224), torch.tensor([0.0]), "img2"),
            (torch.randn(3, 224, 224), torch.tensor([1.0]), "img3"),
        ]
        
        result = collate_fn(batch)
        pixel_values = result["pixel_values"]
        targets = result["labels"]
        sample_ids = result["sample_ids"]
        
        # Check shapes
        self.assertEqual(pixel_values.shape, (3, 3, 224, 224))  # [B, C, H, W]
        self.assertEqual(targets.shape, (3, 1))  # [B, num_classes]
        self.assertEqual(len(sample_ids), 3)
        
        # Check data types
        self.assertIsInstance(pixel_values, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertIsInstance(sample_ids, tuple)
        
        # Check sample IDs
        self.assertEqual(sample_ids, ("img1", "img2", "img3"))
    
    def test_collate_fn_multi_view(self):
        """Test collate function with multi-view data."""
        # Create batch of multi-view samples
        batch = [
            (torch.randn(4, 3, 224, 224), torch.tensor([1.0, 0.0]), "study1"),
            (torch.randn(4, 3, 224, 224), torch.tensor([0.0, 1.0]), "study2"),
        ]
        
        result = collate_fn(batch)
        pixel_values = result["pixel_values"]
        targets = result["labels"]
        sample_ids = result["sample_ids"]
        
        # Check shapes
        self.assertEqual(pixel_values.shape, (2, 4, 3, 224, 224))  # [B, 4, C, H, W]
        self.assertEqual(targets.shape, (2, 2))  # [B, num_classes]
        self.assertEqual(len(sample_ids), 2)
        
        # Check data types
        self.assertIsInstance(pixel_values, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertIsInstance(sample_ids, tuple)
        
        # Check sample IDs
        self.assertEqual(sample_ids, ("study1", "study2"))
        
        # Check that each sample has 4 views
        for i in range(2):
            sample_views = pixel_values[i]  # [4, C, H, W]
            self.assertEqual(sample_views.shape, (4, 3, 224, 224))
    
    def test_collate_fn_mixed_batch(self):
        """Test that collate function handles consistent batch shapes."""
        # All samples in a batch should have the same shape
        batch = [
            (torch.randn(3, 224, 224), torch.tensor([1.0]), "img1"),
            (torch.randn(3, 224, 224), torch.tensor([0.0]), "img2"),
        ]
        
        result = collate_fn(batch)
        self.assertEqual(result["pixel_values"].shape, (2, 3, 224, 224))
        
        # Multi-view batch
        batch = [
            (torch.randn(4, 3, 224, 224), torch.tensor([1.0, 0.0]), "study1"),
            (torch.randn(4, 3, 224, 224), torch.tensor([0.0, 1.0]), "study2"),
        ]
        
        result = collate_fn(batch)
        self.assertEqual(result["pixel_values"].shape, (2, 4, 3, 224, 224))
    
    def test_collate_fn_returns_dict(self):
        """Test that collate_fn returns a dict with expected keys."""
        batch = [
            (torch.randn(3, 224, 224), torch.tensor([1.0]), "img1"),
        ]
        result = collate_fn(batch)
        self.assertIsInstance(result, dict)
        self.assertIn("pixel_values", result)
        self.assertIn("labels", result)
        self.assertIn("sample_ids", result)
    
    def test_collate_fn_empty_batch(self):
        """Test collate function with empty batch."""
        batch = []
        
        with self.assertRaises(ValueError):
            collate_fn(batch)

    def test_collate_fn_inconsistent_shapes(self):
        """Test collate_fn with inconsistent image shapes (should fail)."""
        batch = [
            (torch.randn(3, 224, 224), torch.tensor([1]), "img1"),
            (torch.randn(3, 256, 256), torch.tensor([0]), "img2"),  # Different shape
        ]
        with self.assertRaises(RuntimeError):
            collate_fn(batch)

    def test_collate_fn_different_target_shapes(self):
        """Test collate_fn with different target shapes (should fail)."""
        batch = [
            (torch.randn(3, 224, 224), torch.tensor([1]), "img1"),
            (torch.randn(3, 224, 224), torch.tensor([0, 1]), "img2"),  # Different target shape
        ]
        with self.assertRaises(RuntimeError):
            collate_fn(batch)

if __name__ == "__main__":
    unittest.main()
