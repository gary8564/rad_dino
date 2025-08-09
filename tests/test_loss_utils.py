import unittest
import unittest.mock
import torch
import pandas as pd
from torch.utils.data import Dataset
from rad_dino.utils.loss_utils import get_class_weights

class MockDataset(Dataset):
    """Mock dataset for testing class weight calculation."""
    
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)

class TestLossUtils(unittest.TestCase):
    
    def test_get_class_weights_binary_balanced(self):
        """Test class weight calculation for balanced binary classification."""
        # Create balanced binary dataset
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3', 'img4'],
            'label': [0, 0, 1, 1]  # 2 negative, 2 positive
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("binary", dataset)
        
        # pos_weight should be num_negative / num_positive = 2/2 = 1.0
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (1,))
        self.assertEqual(weights.item(), 1.0)
    
    def test_get_class_weights_binary_imbalanced(self):
        """Test class weight calculation for imbalanced binary classification."""
        # Create imbalanced binary dataset
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3', 'img4', 'img5'],
            'label': [0, 0, 0, 1, 1]  # 3 negative, 2 positive
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("binary", dataset)
        
        # pos_weight should be num_negative / num_positive = 3/2 = 1.5
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (1,))
        self.assertEqual(weights.item(), 1.5)
    
    def test_get_class_weights_binary_no_positive(self):
        """Test class weight calculation when no positive samples exist."""
        # Create dataset with only negative samples
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3'],
            'label': [0, 0, 0]  # All negative
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("binary", dataset)
        
        # Should return None when no positive samples
        self.assertIsNone(weights)
    
    def test_get_class_weights_multilabel(self):
        """Test class weight calculation for multilabel classification."""
        # Create multilabel dataset
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3', 'img4'],
            'class1': [0, 0, 1, 1],  # 2 negative, 2 positive
            'class2': [0, 1, 0, 1],  # 2 negative, 2 positive
            'class3': [0, 0, 0, 1]   # 3 negative, 1 positive
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("multilabel", dataset)
        
        # Should return weights for each class
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (3,))  # 3 classes
        
        # Expected weights: [1.0, 1.0, 3.0] (num_negative / num_positive for each class)
        expected_weights = torch.tensor([1.0, 1.0, 3.0], dtype=torch.float32)
        torch.testing.assert_close(weights, expected_weights)
    
    def test_get_class_weights_multilabel_no_positive(self):
        """Test class weight calculation for multilabel with no positive samples for some classes."""
        # Create multilabel dataset with some classes having no positive samples
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3'],
            'class1': [0, 0, 1],  # 2 negative, 1 positive
            'class2': [0, 0, 0],  # All negative
            'class3': [1, 1, 1]   # All positive
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("multilabel", dataset)
        
        # Should return weights for each class
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (3,))  # 3 classes
        
        # Expected weights: [2.0, 1.0, 0.0] (default weight 1.0 for class with no positive)
        expected_weights = torch.tensor([2.0, 1.0, 0.0], dtype=torch.float32)
        torch.testing.assert_close(weights, expected_weights)
    
    def test_get_class_weights_multiclass(self):
        """Test class weight calculation for multiclass classification."""
        # Create multiclass dataset
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3', 'img4', 'img5', 'img6'],
            'class1': [1, 1, 0, 0, 0, 0],  # 2 positive
            'class2': [0, 0, 1, 1, 0, 0],  # 2 positive
            'class3': [0, 0, 0, 0, 1, 1]   # 2 positive
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("multiclass", dataset)
        
        # Should return weights for each class
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (3,))  # 3 classes
        
        # Expected weights: [3.0, 3.0, 3.0] (total_samples / num_positive for each class)
        # total_samples = 6, num_positive for each class = 2, so weight = 6/2 = 3.0
        expected_weights = torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32)
        torch.testing.assert_close(weights, expected_weights)
    
    def test_get_class_weights_multiclass_imbalanced(self):
        """Test class weight calculation for imbalanced multiclass classification."""
        # Create imbalanced multiclass dataset
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3', 'img4', 'img5'],
            'class1': [1, 1, 1, 0, 0],  # 3 positive
            'class2': [0, 0, 0, 1, 0],  # 1 positive
            'class3': [0, 0, 0, 0, 1]   # 1 positive
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("multiclass", dataset)
        
        # Should return weights for each class
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (3,))  # 3 classes
        
        # Expected weights: [1.67, 5.0, 5.0] (total_samples / num_positive for each class)
        # total_samples = 5, num_positive: [3, 1, 1], so weights: [5/3, 5/1, 5/1]
        expected_weights = torch.tensor([5/3, 5.0, 5.0], dtype=torch.float32)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)
    
    def test_get_class_weights_multiclass_no_positive(self):
        """Test class weight calculation for multiclass with no positive samples for some classes."""
        # Create multiclass dataset with some classes having no positive samples
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3'],
            'class1': [1, 0, 0],  # 1 positive
            'class2': [0, 0, 0],  # No positive
            'class3': [0, 1, 0]   # 1 positive
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("multiclass", dataset)
        
        # Should return weights for each class
        self.assertIsInstance(weights, torch.Tensor)
        self.assertEqual(weights.shape, (3,))  # 3 classes
        
        # Expected weights: [3.0, 1.0, 3.0] (default weight 1.0 for class with no positive)
        expected_weights = torch.tensor([3.0, 1.0, 3.0], dtype=torch.float32)
        torch.testing.assert_close(weights, expected_weights)
    
    def test_get_class_weights_invalid_task(self):
        """Test class weight calculation with invalid task."""
        df = pd.DataFrame({
            'image_id': ['img1', 'img2'],
            'label': [0, 1]
        })
        dataset = MockDataset(df)
        
        with self.assertRaises(ValueError):
            get_class_weights("invalid_task", dataset)
    
    def test_get_class_weights_empty_dataset(self):
        """Test class weight calculation with empty dataset."""
        df = pd.DataFrame(columns=['image_id', 'label'])
        dataset = MockDataset(df)
        
        weights = get_class_weights("binary", dataset)
        
        # Should return None for empty dataset
        self.assertIsNone(weights)
    
    def test_get_class_weights_data_types(self):
        """Test that returned weights have correct data type."""
        df = pd.DataFrame({
            'image_id': ['img1', 'img2', 'img3'],
            'label': [0, 1, 1]
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("binary", dataset)
        
        # Should be torch.float32
        self.assertEqual(weights.dtype, torch.float32)
    
    def test_get_class_weights_multilabel_empty_weights(self):
        """Test class weight calculation for multilabel with no valid weights."""
        # Create dataset with only image_id column
        df = pd.DataFrame({
            'image_id': ['img1', 'img2']
        })
        dataset = MockDataset(df)
        
        weights = get_class_weights("multilabel", dataset)
        
        # Should return None when no label columns exist
        self.assertIsNone(weights)

if __name__ == "__main__":
    unittest.main() 