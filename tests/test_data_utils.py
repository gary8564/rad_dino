import unittest
import unittest.mock
import torch
import numpy as np
from rad_dino.utils.data_utils import collate_fn, get_transforms

class TestDataUtils(unittest.TestCase):
    
    def test_collate_fn_single_view(self):
        """Test collate function with single-view data."""
        # Create batch of single-view samples
        batch = [
            (torch.randn(3, 224, 224), torch.tensor([1.0]), "img1"),
            (torch.randn(3, 224, 224), torch.tensor([0.0]), "img2"),
            (torch.randn(3, 224, 224), torch.tensor([1.0]), "img3"),
        ]
        
        pixel_values, targets, sample_ids = collate_fn(batch)
        
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
        
        pixel_values, targets, sample_ids = collate_fn(batch)
        
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
        
        pixel_values, targets, sample_ids = collate_fn(batch)
        self.assertEqual(pixel_values.shape, (2, 3, 224, 224))
        
        # Multi-view batch
        batch = [
            (torch.randn(4, 3, 224, 224), torch.tensor([1.0, 0.0]), "study1"),
            (torch.randn(4, 3, 224, 224), torch.tensor([0.0, 1.0]), "study2"),
        ]
        
        pixel_values, targets, sample_ids = collate_fn(batch)
        self.assertEqual(pixel_values.shape, (2, 4, 3, 224, 224))
    
    def test_get_transforms(self):
        """Test that transforms are created correctly."""
        # Mock model path
        with unittest.mock.patch('rad_dino.utils.data_utils.AutoImageProcessor') as mock_processor:
            # Mock the image processor
            mock_processor.from_pretrained.return_value = unittest.mock.MagicMock(
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                resample=3,  # BICUBIC
                crop_size={"height": 224, "width": 224},
                size={"shortest_edge": 256}
            )
            
            train_transforms, val_transforms = get_transforms("facebook/dinov2-base")
            
            # Check that transforms are created
            self.assertIsNotNone(train_transforms)
            self.assertIsNotNone(val_transforms)
            
            # Check that Grayscale transform is included
            train_transform_names = [type(t).__name__ for t in train_transforms.transforms]
            val_transform_names = [type(t).__name__ for t in val_transforms.transforms]
            
            self.assertIn('Grayscale', train_transform_names)
            self.assertIn('Grayscale', val_transform_names)
            
            # Test that transforms work on dummy data
            dummy_input = torch.randn(224, 224)
            train_output = train_transforms(dummy_input)
            val_output = val_transforms(dummy_input)
            
            # Should output 3-channel tensors
            self.assertEqual(train_output.shape, (3, 224, 224))
            self.assertEqual(val_output.shape, (3, 224, 224))
    
    def test_transforms_3channel_output(self):
        """Test that transforms always output 3-channel tensors."""
        with unittest.mock.patch('rad_dino.utils.data_utils.AutoImageProcessor') as mock_processor:
            mock_processor.from_pretrained.return_value = unittest.mock.MagicMock(
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                resample=3,
                crop_size={"height": 224, "width": 224},
                size={"shortest_edge": 256}
            )
            
            train_transforms, val_transforms = get_transforms("facebook/dinov2-base")
            
            # Test with grayscale input (2D tensor)
            grayscale_input = torch.randn(224, 224)
            
            train_output = train_transforms(grayscale_input)
            val_output = val_transforms(grayscale_input)
            
            # Both should output 3-channel tensors
            self.assertEqual(train_output.shape, (3, 224, 224))
            self.assertEqual(val_output.shape, (3, 224, 224))
            
            # Test with RGB input (3D tensor)
            rgb_input = torch.randn(3, 224, 224)
            
            train_output_rgb = train_transforms(rgb_input)
            val_output_rgb = val_transforms(rgb_input)
            
            # Should still output 3-channel tensors
            self.assertEqual(train_output_rgb.shape, (3, 224, 224))
            self.assertEqual(val_output_rgb.shape, (3, 224, 224))

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

    def test_get_transforms_different_models(self):
        """Test get_transforms with different model repositories."""
        with unittest.mock.patch('rad_dino.utils.data_utils.AutoImageProcessor') as mock_processor:
            mock_processor.from_pretrained.return_value = unittest.mock.MagicMock(
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                resample=3,
                crop_size={"height": 224, "width": 224},
                size={"shortest_edge": 256}
            )
            
            # Test with different model repositories
            models = ["facebook/dinov2-base", "facebook/dinov2-small", "microsoft/rad-dino"]
            
            for model in models:
                train_transforms, val_transforms = get_transforms(model)
                self.assertIsNotNone(train_transforms)
                self.assertIsNotNone(val_transforms)
                
                # Test that transforms work
                dummy_input = torch.randn(224, 224)
                train_output = train_transforms(dummy_input)
                val_output = val_transforms(dummy_input)
                
                self.assertEqual(train_output.shape, (3, 224, 224))
                self.assertEqual(val_output.shape, (3, 224, 224))

    def test_get_transforms_ark_model(self):
        """Test get_transforms for Ark models with 768x768 image size."""
        with unittest.mock.patch('rad_dino.utils.data_utils.AutoImageProcessor') as mock_processor:
            mock_processor.from_pretrained.return_value = unittest.mock.MagicMock(
                image_mean=[0.485, 0.456, 0.406],
                image_std=[0.229, 0.224, 0.225],
                resample=3,
                crop_size={"height": 768, "width": 768},
                size={"shortest_edge": 800}
            )
            
            train_transforms, val_transforms = get_transforms("ark-swin-large-768")
        
        # Test that transforms are created
        self.assertIsNotNone(train_transforms)
        self.assertIsNotNone(val_transforms)
        
        # Test with a dummy image to ensure transforms work
        dummy_image = torch.randn(3, 224, 224)
        
        # Test train transforms
        try:
            transformed_train = train_transforms(dummy_image)
            self.assertEqual(transformed_train.shape[1:], (768, 768))  # Should be 768x768
        except Exception as e:
            self.fail(f"Train transforms failed: {e}")
        
        # Test val transforms
        try:
            transformed_val = val_transforms(dummy_image)
            self.assertEqual(transformed_val.shape[1:], (768, 768))  # Should be 768x768
        except Exception as e:
            self.fail(f"Val transforms failed: {e}")
        
        # Test that normalization uses ImageNet values for Ark
        # The normalized values should be in a reasonable range for ImageNet normalization
        self.assertTrue(torch.all(transformed_val >= -3.0) and torch.all(transformed_val <= 3.0))

if __name__ == "__main__":
    unittest.main() 