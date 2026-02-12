import unittest
import unittest.mock
import torch
import numpy as np
import torchvision.transforms.v2 as transforms
from PIL import Image
from rad_dino.utils.transforms import get_transforms


def _make_pil_grayscale(size=(224, 224)):
    """Create a random grayscale PIL Image for testing."""
    arr = np.random.randint(0, 256, size, dtype=np.uint8)
    return Image.fromarray(arr, mode='L')


def _make_pil_rgb(size=(224, 224)):
    """Create a random RGB PIL Image for testing."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode='RGB')


class TestTransforms(unittest.TestCase):
    
    def test_get_transforms(self):
        """Test that transforms are created correctly."""
        with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
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
            
            # Test that transforms work on PIL Image input
            # (transforms no longer include ToPILImage, so input must be PIL)
            dummy_input = _make_pil_rgb()
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
            
            # Test with grayscale PIL Image (mode 'L')
            grayscale_input = _make_pil_grayscale()
            
            train_output = train_transforms(grayscale_input)
            val_output = val_transforms(grayscale_input)
            
            # Both should output 3-channel tensors
            self.assertEqual(train_output.shape, (3, 224, 224))
            self.assertEqual(val_output.shape, (3, 224, 224))
            
            # Test with RGB PIL Image (mode 'RGB')
            rgb_input = _make_pil_rgb()
            
            train_output_rgb = train_transforms(rgb_input)
            val_output_rgb = val_transforms(rgb_input)
            
            # Should still output 3-channel tensors
            self.assertEqual(train_output_rgb.shape, (3, 224, 224))
            self.assertEqual(val_output_rgb.shape, (3, 224, 224))

    def test_get_transforms_different_models(self):
        """Test get_transforms with different model identifiers."""
        model_configs = [
            ("dinov2", (224, 224)),
            ("rad_dino", (518, 518)),
            ("medsiglip", (448, 448))
        ]
        
        for model_name, expected_size in model_configs:
            with unittest.mock.patch('rad_dino.utils.transforms.get_model_config') as mock_get_config:
                mock_get_config.return_value = {
                    "crop_size": list(expected_size),
                    "size": [expected_size[0] + 32, expected_size[1] + 32],
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225],
                    "interpolation": 3
                }
                
                train_transforms, val_transforms = get_transforms(model_name)
                self.assertIsNotNone(train_transforms)
                self.assertIsNotNone(val_transforms)
                
                # Test that transforms work with PIL Image input
                dummy_input = _make_pil_rgb()
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
        
        # Test with a PIL Image to ensure transforms work
        dummy_image = _make_pil_rgb(size=(224, 224))
        
        # Test train transforms
        transformed_train = train_transforms(dummy_image)
        self.assertEqual(transformed_train.shape[1:], (768, 768))
        
        # Test val transforms
        transformed_val = val_transforms(dummy_image)
        self.assertEqual(transformed_val.shape[1:], (768, 768))
        
        # Test that normalization produces values in a reasonable range for ImageNet
        # After normalization, values are typically in [-3, 3] range
        self.assertTrue(transformed_val.min() >= -5.0 and transformed_val.max() <= 5.0)

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
            # (ToPILImage was removed since input is now PIL from dataset)
            expected_train_components = [
                'Grayscale', 'RandomResizedCrop', 'RandomHorizontalFlip',
                'RandomApply', 'RandomAffine', 'ToTensor', 'Normalize'
            ]
            
            for component in expected_train_components:
                self.assertIn(component, train_transform_names, 
                            f"Expected {component} in train transforms")
            
            # Check that val transforms are simpler (no augmentation)
            expected_val_components = [
                'Grayscale', 'Resize', 'CenterCrop', 'ToTensor', 'Normalize'
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
            
            # Create a PIL Image with values in [0, 255]
            dummy_input = _make_pil_rgb()
            
            # Apply transforms
            train_output = train_transforms(dummy_input)
            val_output = val_transforms(dummy_input)
            
            # Check that outputs are normalized (should be in reasonable range)
            # Normalized values should be roughly in [-3, 3] range for ImageNet normalization
            self.assertTrue(train_output.min() >= -5.0 and train_output.max() <= 5.0)
            self.assertTrue(val_output.min() >= -5.0 and val_output.max() <= 5.0)

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
            
            # Test with grayscale PIL Image
            grayscale_input = _make_pil_grayscale()
            
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
