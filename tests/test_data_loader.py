import unittest
import torch
import numpy as np
from rad_dino.data.data_loader import create_train_loader, load_data

class TestDataLoader(unittest.TestCase):
    
    @unittest.mock.patch('rad_dino.data.data_loader.RadImageClassificationDataset')
    def test_create_train_loader(self, mock_dataset_class):
        """Test create_train_loader function."""
        # Mock the dataset
        mock_dataset = unittest.mock.MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.df = unittest.mock.MagicMock()
        mock_dataset.sample_ids = list(range(10))
        mock_dataset_class.return_value = mock_dataset
        
        # Mock transforms
        mock_train_transform = unittest.mock.MagicMock()
        mock_val_transform = unittest.mock.MagicMock()
        
        # Test parameters
        data_root_folder = "/test/data"
        task = "binary"
        train_idx = [0, 1, 2, 3, 4, 5, 6]
        val_idx = [7, 8, 9]
        mini_batch_size = 4
        batch_size = 8
        num_workers = 2
        
        # Call create_train_loader
        train_loader, val_loader = create_train_loader(
            data_root_folder=data_root_folder,
            task=task,
            train_idx=train_idx,
            val_idx=val_idx,
            train_transforms=mock_train_transform,
            val_transforms=mock_val_transform,
            mini_batch_size=mini_batch_size,
            batch_size=batch_size,
            num_workers=num_workers,
            multi_view=False
        )
        
        # Check that result is a tuple (train_loader, val_loader)
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader)
        
        # Check batch sizes
        self.assertEqual(train_loader.batch_size, mini_batch_size)
        self.assertEqual(val_loader.batch_size, batch_size)
        self.assertEqual(train_loader.num_workers, num_workers)
        self.assertEqual(val_loader.num_workers, num_workers)
        # Note: DataLoader doesn't expose shuffle as a public attribute
        # The shuffle parameter is handled internally by DataLoader

    @unittest.mock.patch('rad_dino.data.data_loader.RadImageClassificationDataset')
    def test_load_data_single_split(self, mock_dataset_class):
        """Test load_data function with single split."""
        # Mock the dataset
        mock_dataset = unittest.mock.MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.df = unittest.mock.MagicMock()
        mock_dataset.sample_ids = list(range(10))
        mock_dataset_class.return_value = mock_dataset
        
        # Mock transforms
        mock_train_transform = unittest.mock.MagicMock()
        mock_val_transform = unittest.mock.MagicMock()
        
        # Test parameters
        data_root_folder = "/test/data"
        task = "binary"
        batch_size = 8
        gradient_accumulation_steps = 2
        train_size = 0.7
        seed = 42
        
        # Call load_data
        result = load_data(
            data_root_folder=data_root_folder,
            task=task,
            batch_size=batch_size,
            train_transforms=mock_train_transform,
            val_transforms=mock_val_transform,
            num_workers=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            kfold=None,  # Single split
            train_size=train_size,
            seed=seed,
            multi_view=False
        )
        
        # Check that result is a tuple (train_loader, val_loader)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        # Check that dataset was created
        mock_dataset_class.assert_called()

    @unittest.mock.patch('rad_dino.data.data_loader.RadImageClassificationDataset')
    def test_load_data_kfold_binary(self, mock_dataset_class):
        """Test load_data function with k-fold for binary classification."""
        # Mock the dataset
        mock_dataset = unittest.mock.MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.df = unittest.mock.MagicMock()
        mock_dataset.df.__getitem__.return_value = unittest.mock.MagicMock()
        mock_dataset.df.__getitem__().to_numpy.return_value = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        mock_dataset.sample_ids = list(range(10))
        mock_dataset_class.return_value = mock_dataset
        
        # Mock transforms
        mock_train_transform = unittest.mock.MagicMock()
        mock_val_transform = unittest.mock.MagicMock()
        
        # Test parameters
        data_root_folder = "/test/data"
        task = "binary"
        batch_size = 8
        gradient_accumulation_steps = 2
        kfold = 3
        
        # Call load_data
        result = load_data(
            data_root_folder=data_root_folder,
            task=task,
            batch_size=batch_size,
            train_transforms=mock_train_transform,
            val_transforms=mock_val_transform,
            num_workers=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            kfold=kfold,
            multi_view=False
        )
        
        # Check that result is a list of tuples (k-fold results)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), kfold)
        
        for fold_result in result:
            self.assertIsInstance(fold_result, tuple)
            self.assertEqual(len(fold_result), 2)

    @unittest.mock.patch('rad_dino.data.data_loader.RadImageClassificationDataset')
    def test_load_data_invalid_batch_size(self, mock_dataset_class):
        """Test load_data with invalid batch size that's not divisible by gradient_accumulation_steps."""
        # Mock the dataset
        mock_dataset = unittest.mock.MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset.df = unittest.mock.MagicMock()
        mock_dataset.sample_ids = list(range(10))
        mock_dataset_class.return_value = mock_dataset
        
        with self.assertRaises(ValueError):
            load_data(
                data_root_folder="/test/data",
                task="binary",
                batch_size=7,  # Not divisible by 2
                train_transforms=unittest.mock.MagicMock(),
                val_transforms=unittest.mock.MagicMock(),
                num_workers=2,
                gradient_accumulation_steps=2,
                kfold=None
            )

if __name__ == "__main__":
    unittest.main() 