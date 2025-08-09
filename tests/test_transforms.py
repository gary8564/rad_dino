import unittest
import unittest.mock
import torch
import torchvision.transforms.v2 as transforms
from rad_dino.utils.transforms import get_transforms

class TestTransforms(unittest.TestCase):
    
    def test_get_transforms(self):
        """Test that transforms are created correctly."""
        # Mock the config_utils.get_model_config function
        with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
            # Mock the configuration
            mock_get_config.return_value = {
                "crop_size": [224, 224],
                "size": [256, 256],
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "interpolation": 3  # BICUBIC
            }
            
            train_transforms, val_transforms = get_transforms("dinov2")
            
            # Check that transforms are created
            self.assertIsNotNone(train_transforms)
            self.assertIsNotNone(val_transforms)
            
            # Check that transforms are Compose objects
            self.assertIsInstance(train_transforms, transforms.Compose)
            self.assertIsInstance(val_transforms, transforms.Compose)
            
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
        with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
            mock_get_config.return_value = {
                "crop_size": [224, 224],
                "size": [256, 256],
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "interpolation": 3
            }
            
            train_transforms, val_transforms = get_transforms("dinov2")
            
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

    def test_get_transforms_different_models(self):
        """Test get_transforms with different model identifiers."""
        # Test with different model identifiers and their expected output sizes
        model_configs = [
            ("dinov2", (224, 224)),
            ("rad_dino", (518, 518)),
            ("medsiglip", (448, 448))
        ]
        
        for model_name, expected_size in model_configs:
            with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
                mock_get_config.return_value = {
                    "crop_size": list(expected_size),
                    "size": [expected_size[0] + 32, expected_size[1] + 32],  # Slightly larger
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                    "interpolation": 3
                }
                
                train_transforms, val_transforms = get_transforms(model_name)
                self.assertIsNotNone(train_transforms)
                self.assertIsNotNone(val_transforms)
                
                # Test that transforms work
                dummy_input = torch.randn(224, 224)
                train_output = train_transforms(dummy_input)
                val_output = val_transforms(dummy_input)
                
                # Check that output has the correct size for each model
                self.assertEqual(train_output.shape[1:], expected_size)
                self.assertEqual(val_output.shape[1:], expected_size)

    def test_get_transforms_ark_model(self):
        """Test get_transforms for Ark models with 768x768 image size."""
        with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
            mock_get_config.return_value = {
                "crop_size": [768, 768],
                "size": [800, 800],
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "interpolation": 3
            }
            
            train_transforms, val_transforms = get_transforms("ark")
        
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

    def test_transforms_augmentation_components(self):
        """Test that train transforms include expected augmentation components."""
        with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
            mock_get_config.return_value = {
                "crop_size": [224, 224],
                "size": [256, 256],
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "interpolation": 3
            }
            
            train_transforms, val_transforms = get_transforms("dinov2")
            
            # Get transform names
            train_transform_names = [type(t).__name__ for t in train_transforms.transforms]
            val_transform_names = [type(t).__name__ for t in val_transforms.transforms]
            
            # Check that train transforms include augmentation components
            expected_train_components = [
                'ToPILImage', 'Grayscale', 'RandomResizedCrop', 'RandomHorizontalFlip',
                'RandomApply', 'RandomGrayscale', 'RandomAffine', 'ToTensor', 'Normalize'
            ]
            
            for component in expected_train_components:
                self.assertIn(component, train_transform_names, 
                            f"Expected {component} in train transforms")
            
            # Check that val transforms are simpler (no augmentation)
            expected_val_components = [
                'ToPILImage', 'Grayscale', 'Resize', 'CenterCrop', 'ToTensor', 'Normalize'
            ]
            
            for component in expected_val_components:
                self.assertIn(component, val_transform_names,
                            f"Expected {component} in val transforms")

    def test_transforms_normalization(self):
        """Test that normalization is applied correctly."""
        with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
            mock_get_config.return_value = {
                "crop_size": [224, 224],
                "size": [256, 256],
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "interpolation": 3
            }
            
            train_transforms, val_transforms = get_transforms("dinov2")
            
            # Create a dummy tensor with values in [0, 1] range
            dummy_input = torch.rand(224, 224)
            
            # Apply transforms
            train_output = train_transforms(dummy_input)
            val_output = val_transforms(dummy_input)
            
            # Check that outputs are normalized (should be in reasonable range)
            # Normalized values should be roughly in [-3, 3] range for ImageNet normalization
            self.assertTrue(torch.all(train_output >= -3.0) and torch.all(train_output <= 3.0))
            self.assertTrue(torch.all(val_output >= -3.0) and torch.all(val_output <= 3.0))

    def test_transforms_grayscale_conversion(self):
        """Test that grayscale conversion works correctly."""
        with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
            mock_get_config.return_value = {
                "crop_size": [224, 224],
                "size": [256, 256],
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "interpolation": 3
            }
            
            train_transforms, val_transforms = get_transforms("dinov2")
            
            # Test with grayscale input
            grayscale_input = torch.randn(224, 224)
            
            train_output = train_transforms(grayscale_input)
            val_output = val_transforms(grayscale_input)
            
            # Should output 3-channel tensors (grayscale converted to RGB)
            self.assertEqual(train_output.shape, (3, 224, 224))
            self.assertEqual(val_output.shape, (3, 224, 224))
            
            # All channels should be identical for grayscale input
            self.assertTrue(torch.allclose(train_output[0], train_output[1], atol=1e-5))
            self.assertTrue(torch.allclose(train_output[1], train_output[2], atol=1e-5))
            self.assertTrue(torch.allclose(val_output[0], val_output[1], atol=1e-5))
            self.assertTrue(torch.allclose(val_output[1], val_output[2], atol=1e-5))

if __name__ == "__main__":
    unittest.main() 