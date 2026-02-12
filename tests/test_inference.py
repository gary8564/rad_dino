import unittest
import os
import tempfile
import shutil
import torch
from unittest.mock import Mock, patch, MagicMock
import argparse
import sys
from rad_dino.run.inference import (
    get_args_parser,
    validate_args,
    create_output_directories
)
from rad_dino.configs.config import InferenceConfig, OutputPaths

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
        """Test output directory creation with all visualization flags enabled"""
        config = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            show_gradcam=True,
            show_attention=True,
            show_lrp=True,
        )
        output_paths = create_output_directories(self.temp_dir, self.mock_accelerator, config)
        
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
    
    def test_create_output_directories_no_visualizations(self):
        """Test output directory creation without visualization flags"""
        config = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
        )
        output_paths = create_output_directories(self.temp_dir, self.mock_accelerator, config)
        
        self.assertIsInstance(output_paths, OutputPaths)
        self.assertEqual(output_paths.base, self.temp_dir)
        self.assertEqual(output_paths.figs, f"{self.temp_dir}/figs")
        self.assertEqual(output_paths.table, f"{self.temp_dir}/table")
        self.assertIsNone(output_paths.gradcam)
        self.assertIsNone(output_paths.attention)
        self.assertIsNone(output_paths.lrp)
        
        # Only figs and table should exist
        self.assertTrue(os.path.exists(f"{self.temp_dir}/figs"))
        self.assertTrue(os.path.exists(f"{self.temp_dir}/table"))
        self.assertFalse(os.path.exists(f"{self.temp_dir}/gradcam"))
        self.assertFalse(os.path.exists(f"{self.temp_dir}/attention"))
        self.assertFalse(os.path.exists(f"{self.temp_dir}/lrp"))

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

    @patch('rad_dino.run.inference.run_inference')
    @patch('rad_dino.run.inference.setup_model')
    @patch('rad_dino.run.inference.determine_class_info')
    @patch('rad_dino.run.inference.setup_data_loader')
    @patch('rad_dino.run.inference.Accelerator')
    @patch('os.makedirs')
    def test_main_function_flow(self, mock_makedirs, mock_accelerator, 
                               mock_setup_data_loader, mock_determine_class_info,
                               mock_setup_model, mock_run_inference):
        """Test main function flow with mocked dependencies"""
        # Mock accelerator
        mock_acc = Mock()
        mock_acc.is_main_process = True
        mock_accelerator.return_value = mock_acc
        
        # Mock setup_data_loader to return a mock dataset and loader
        mock_ds = Mock()
        mock_ds.labels = ['label1', 'label2']
        mock_loader = Mock()
        mock_setup_data_loader.return_value = (mock_ds, mock_loader)
        
        # Mock determine_class_info
        mock_determine_class_info.return_value = (['label1', 'label2'], 2)
        
        # Mock model wrapper
        mock_model_wrapper = Mock()
        mock_setup_model.return_value = mock_model_wrapper
        
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
            mock_setup_data_loader.assert_called_once()
            mock_setup_model.assert_called_once()
            mock_run_inference.assert_called_once()

    @patch('rad_dino.run.inference.load_model')
    def test_model_repository_mapping(self, mock_load_model):
        """Test model repository mapping contains expected keys"""
        from rad_dino.run.inference import MODEL_REPOS
        
        # Check that all expected model keys are present
        expected_keys = {
            "rad-dino", "dinov2-base", "dinov2-small", "dinov2-large",
            "dinov3-large", "dinov3-base", "dinov3-small-plus",
            "medsiglip", "ark",
        }
        self.assertTrue(expected_keys.issubset(set(MODEL_REPOS.keys())),
                        f"Missing keys: {expected_keys - set(MODEL_REPOS.keys())}")
        
        # Spot-check a few well-known repos
        self.assertEqual(MODEL_REPOS["rad-dino"], "microsoft/rad-dino")
        self.assertEqual(MODEL_REPOS["medsiglip"], "google/medsiglip-448")

    def test_compile_flag_in_config(self):
        """Test that compile flag is available in InferenceConfig"""
        # Default: compile is False
        config = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
        )
        self.assertFalse(config.compile)
        
        # Explicit compile=True
        config_compiled = InferenceConfig(
            task="multilabel",
            data="VinDr-CXR",
            model="rad-dino",
            model_path="test/path",
            output_path="test/output",
            compile=True,
        )
        self.assertTrue(config_compiled.compile)

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
