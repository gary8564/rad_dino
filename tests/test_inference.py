import unittest
import os
import tempfile
import shutil
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import argparse
import yaml
import sys
from rad_dino.run.inference import (
    get_args_parser,
    validate_args,
    create_output_directories
)
from rad_dino.configs.config import InferenceConfig, OutputPaths
from accelerate import Accelerator

class TestInference(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp(dir="/tmp")
        self.mock_accelerator = Mock()
        self.mock_accelerator.device = torch.device('cpu')
        self.mock_accelerator.is_main_process = True
        
        # Sample data
        self.class_labels = ['label1', 'label2', 'label3']
        self.num_classes = len(self.class_labels)
        self.batch_size = 2
        self.image_ids = ['img1', 'img2']
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_args_parser(self):
        """Test argument parser creation and default values"""
        parser = get_args_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)
        
        # Test required arguments for different models
        test_cases = [
            {
                'args': ['--task', 'multilabel', '--data', 'VinDr-CXR', '--model', 'rad-dino', '--model-path', 'runs/checkpoints_rad_dino_2025_05_18_233958', '--output-path', '../../experiments'],
                'expected_model': 'rad-dino'
            },
            {
                'args': ['--task', 'multiclass', '--data', 'VinDr-Mammo', '--model', 'medsiglip', '--model-path', 'runs/checkpoints_medsiglip_2025_05_18_233958', '--output-path', '../../experiments'],
                'expected_model': 'medsiglip'
            },
            {
                'args': ['--task', 'binary', '--data', 'RSNA-Pneumonia', '--model', 'ark', '--model-path', 'runs/checkpoints_ark_2025_05_18_233958', '--output-path', '../../experiments'],
                'expected_model': 'ark'
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(model=test_case['expected_model']):
                with patch.object(sys, 'argv', ['prog'] + test_case['args']):
                    args = parser.parse_args(test_case['args'])
                    self.assertEqual(args.model, test_case['expected_model'])
                    self.assertEqual(args.batch_size, 16)  # Default value

    def test_validate_args(self):
        """Test argument validation"""
        # Test valid config
        valid_config = InferenceConfig(
            task="multilabel",
            data="VinDr-Mammo",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            multi_view=True
        )
        # Should not raise any exception
        validate_args(valid_config)
        
        # Test invalid multi-view with non-mammo data
        invalid_config = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            multi_view=True
        )
        with self.assertRaises(ValueError):
            validate_args(invalid_config)
        
        # Test attention visualization without required params
        attention_config = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            show_attention=True
        )
        with self.assertRaises(ValueError):
            validate_args(attention_config)

    def test_create_output_directories(self):
        """Test output directory creation"""
        output_paths = create_output_directories(self.temp_dir, self.mock_accelerator)
        
        self.assertIsInstance(output_paths, OutputPaths)
        self.assertEqual(output_paths.base, self.temp_dir)
        self.assertEqual(output_paths.figs, f"{self.temp_dir}/figs")
        self.assertEqual(output_paths.table, f"{self.temp_dir}/table")
        self.assertEqual(output_paths.gradcam, f"{self.temp_dir}/gradcam")
        self.assertEqual(output_paths.attention, f"{self.temp_dir}/attention")
        self.assertEqual(output_paths.lrp, f"{self.temp_dir}/lrp")
        
        # Check that directories were created
        self.assertTrue(os.path.exists(f"{self.temp_dir}/figs"))
        self.assertTrue(os.path.exists(f"{self.temp_dir}/table"))
        self.assertTrue(os.path.exists(f"{self.temp_dir}/gradcam"))
        self.assertTrue(os.path.exists(f"{self.temp_dir}/attention"))
        self.assertTrue(os.path.exists(f"{self.temp_dir}/lrp"))

    def test_multi_view_validation(self):
        """Test multi-view validation logic"""
        # Test valid multi-view with mammo data
        valid_config = InferenceConfig(
            task="multiclass",
            data="VinDr-Mammo",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            multi_view=True
        )
        validate_args(valid_config)  # Should not raise
        
        # Test invalid multi-view with non-mammo data
        invalid_config = InferenceConfig(
            task="multiclass",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            multi_view=True
        )
        with self.assertRaises(ValueError):
            validate_args(invalid_config)

    def test_attention_visualization_validation(self):
        """Test attention visualization validation"""
        # Test valid attention config
        valid_config = InferenceConfig(
            task="multiclass",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            show_attention=True,
            attention_threshold=0.5,
            save_heads="mean"
        )
        validate_args(valid_config)  # Should not raise
        
        # Test invalid attention config (missing threshold)
        invalid_config = InferenceConfig(
            task="multiclass",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            show_attention=True,
            save_heads="mean"
        )
        with self.assertRaises(ValueError):
            validate_args(invalid_config)

    @patch('rad_dino.run.inference.load_model')
    @patch('rad_dino.run.inference.run_inference')
    @patch('rad_dino.run.inference.get_transforms')
    @patch('rad_dino.run.inference.RadImageClassificationDataset')
    @patch('rad_dino.run.inference.DataLoader')
    @patch('rad_dino.run.inference.Accelerator')
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=MagicMock)
    def test_main_function_flow(self, mock_open, mock_yaml_load, mock_accelerator, 
                               mock_dataloader, mock_dataset, mock_transforms, 
                               mock_run_inference, mock_load_model):
        """Test main function flow with mocked dependencies"""
        # Mock yaml config
        mock_yaml_load.return_value = {
            "VinDr-CXR": {"data_root_folder": "/test/data"}
        }
        
        # Mock accelerator
        mock_acc = Mock()
        mock_accelerator.return_value = mock_acc
        
        # Mock dataset
        mock_ds = Mock()
        mock_ds.labels = ['label1', 'label2']
        mock_dataset.return_value = mock_ds
        
        # Mock the dataset's labels property properly
        mock_ds.__getitem__ = Mock(return_value=mock_ds)
        mock_ds.__len__ = Mock(return_value=10)
        
        # Ensure labels is a real list, not a Mock
        mock_ds.labels = ['label1', 'label2']
        
        # Mock data loader
        mock_loader = Mock()
        mock_dataloader.return_value = mock_loader
        
        # Mock transforms
        mock_transforms.return_value = (None, None)
        
        # Mock model wrapper
        mock_model_wrapper = Mock()
        mock_load_model.return_value = mock_model_wrapper
        
        # Test with sys.argv
        test_args = [
            '--task', 'multilabel',
            '--data', 'VinDr-CXR',
            '--model', 'rad-dino',
            '--model-path', 'test/checkpoint',
            '--output-path', 'test/output'
        ]
        
        with patch.object(sys, 'argv', ['prog'] + test_args):
            from rad_dino.run.inference import main
            main()
            
            # Verify key function calls
            mock_load_model.assert_called_once()
            mock_run_inference.assert_called_once()

    @patch('rad_dino.run.inference.load_model')
    def test_model_repository_mapping(self, mock_load_model):
        """Test model repository mapping"""
        from rad_dino.run.inference import MODEL_REPOS
        
        expected_repos = {
            "rad-dino": "microsoft/rad-dino",
            "dinov2-base": "facebook/dinov2-base", 
            "dinov2-small": "facebook/dinov2-small",
            "medsiglip": "google/medsiglip-448",
            "ark": "microsoft/swin-large-patch4-window12-384-in22k"
        }
        
        self.assertEqual(MODEL_REPOS, expected_repos)

    def test_fusion_type_validation(self):
        """Test fusion type validation"""
        # Test valid fusion types
        valid_fusion_types = ['mean', 'weighted_mean', 'mlp_adapter']
        
        for fusion_type in valid_fusion_types:
            with self.subTest(fusion_type=fusion_type):
                config = InferenceConfig(
                    task="multilabel",
                    data="VinDr-CXR",
                    model="rad-dino",
                    model_path="test/path",
                    output_path="test/output",
                    fusion_type=fusion_type
                )
                # Should not raise
                self.assertEqual(config.fusion_type, fusion_type)

    def test_compute_rollout_validation(self):
        """Test compute rollout validation"""
        # Test valid config with attention and rollout
        valid_config = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            show_attention=True,
            attention_threshold=0.5,
            save_heads="mean",
            compute_rollout=True
        )
        validate_args(valid_config)  # Should not raise
        
        # Test invalid config with rollout but no attention
        invalid_config = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            show_attention=False,
            compute_rollout=True
        )
        validate_args(invalid_config) 

if __name__ == "__main__":
    unittest.main()
