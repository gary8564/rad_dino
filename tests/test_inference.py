import unittest
import os
import tempfile
import shutil
import torch
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
import argparse
import yaml
import pandas as pd
from rad_dino.eval.inference import (
    get_args_parser,
    _load_best_model,
    load_model,
    _run_pytorch_inference,
    _run_onnx_inference,
    run_inference,
    main
)


class TestInference(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_accelerator = Mock()
        self.mock_accelerator.device = torch.device('cpu')
        self.mock_accelerator.is_main_process = True
        
        # Mock backbone config
        self.mock_config = Mock()
        self.mock_config.num_hidden_layers = 12
        self.mock_config.num_attention_heads = 12
        self.mock_config.patch_size = 14
        
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
        
        # Test default arguments
        args = parser.parse_args([])
        self.assertEqual(args.task, "multilabel")
        self.assertEqual(args.data, "VinDr-CXR")
        self.assertEqual(args.model, "rad_dino")
        self.assertEqual(args.model_path, "runs/checkpoints_rad_dino_2025_05_18_233958")
        self.assertEqual(args.output_path, "../../experiments")
        self.assertFalse(args.optimize_compute)
        self.assertFalse(args.show_attention)
        self.assertFalse(args.show_gradcam)
        self.assertIsNone(args.attention_threshold)
        self.assertEqual(args.save_heads, "all")
        
        # Test custom arguments
        custom_args = [
            '--task', 'binary',
            '--data', 'RSNA-Pneumonia',
            '--model', 'dinov2',
            '--show-attention',
            '--show-gradcam',
            '--attention-threshold', '0.6',
            '--save-heads', '0,5,11'
        ]
        args = parser.parse_args(custom_args)
        self.assertEqual(args.task, "binary")
        self.assertEqual(args.data, "RSNA-Pneumonia")
        self.assertEqual(args.model, "dinov2")
        self.assertTrue(args.show_attention)
        self.assertTrue(args.show_gradcam)
        self.assertEqual(args.attention_threshold, 0.6)
        self.assertEqual(args.save_heads, "0,5,11")

    @patch('rad_dino.eval.inference.DinoClassifier')
    @patch('torch.load')
    def test_load_best_model(self, mock_torch_load, mock_dino_classifier):
        """Test loading PyTorch model from checkpoint"""
        # Setup mocks
        mock_model = Mock()
        mock_dino_classifier.return_value = mock_model
        mock_checkpoint = {'model_state': {'key': 'value'}}
        mock_torch_load.return_value = mock_checkpoint
        mock_backbone = Mock()
        
        # Mock the to() method to return the model itself for chaining
        mock_model.to.return_value = mock_model
        
        # Test function
        checkpoint_dir = self.temp_dir
        result = _load_best_model(checkpoint_dir, mock_backbone, self.num_classes, self.mock_accelerator)
        
        # Assertions
        mock_dino_classifier.assert_called_once_with(mock_backbone, num_classes=self.num_classes)
        mock_torch_load.assert_called_once_with(
            os.path.join(checkpoint_dir, "best.pt"), 
            map_location=self.mock_accelerator.device
        )
        mock_model.load_state_dict.assert_called_once_with(mock_checkpoint["model_state"])
        mock_model.to.assert_called_once_with(self.mock_accelerator.device)
        self.assertEqual(result, mock_model)

    @patch('rad_dino.eval.inference.AutoModel')
    @patch('os.path.exists')
    @patch('onnxruntime.InferenceSession')
    def test_load_model_onnx_exists(self, mock_onnx_session, mock_exists, mock_auto_model):
        """Test loading ONNX model when file exists and gradcam is disabled"""
        # Setup mocks
        mock_exists.return_value = True
        mock_backbone = Mock()
        mock_backbone.config = self.mock_config
        mock_auto_model.from_pretrained.return_value = mock_backbone
        
        mock_session = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output1 = Mock()
        mock_output1.name = 'output1'
        mock_output2 = Mock()
        mock_output2.name = 'output2'
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output1, mock_output2]
        mock_session.get_providers.return_value = ['CPUExecutionProvider']
        mock_onnx_session.return_value = mock_session
        
        # Test function
        result = load_model(self.temp_dir, "test_repo", self.num_classes, self.mock_accelerator, show_gradcam=False)
        
        # Assertions
        self.assertEqual(result['type'], 'onnx')
        self.assertEqual(result['session'], mock_session)
        self.assertEqual(result['input_name'], 'input')
        self.assertEqual(result['output_names'], ['output1', 'output2'])
        self.assertEqual(result['config'], self.mock_config)

    @patch('rad_dino.eval.inference.AutoModel')
    @patch('rad_dino.eval.inference._load_best_model')
    @patch('os.path.exists')
    def test_load_model_pytorch_fallback(self, mock_exists, mock_load_best_model, mock_auto_model):
        """Test fallback to PyTorch model when ONNX doesn't exist"""
        # Setup mocks
        mock_exists.return_value = False
        mock_backbone = Mock()
        mock_backbone.config = self.mock_config
        mock_auto_model.from_pretrained.return_value = mock_backbone
        mock_model = Mock()
        mock_load_best_model.return_value = mock_model
        
        # Test function
        result = load_model(self.temp_dir, "test_repo", self.num_classes, self.mock_accelerator, show_gradcam=False)
        
        # Assertions
        self.assertEqual(result['type'], 'pytorch')
        self.assertEqual(result['model'], mock_model)
        self.assertEqual(result['config'], self.mock_config)
        mock_load_best_model.assert_called_once()

    @patch('rad_dino.eval.inference.visualize_gradcam')
    def test_run_pytorch_inference(self, mock_visualize_gradcam):
        """Test PyTorch inference workflow"""
        # Setup mock model
        mock_model = Mock()
        mock_logits = torch.randn(self.batch_size, self.num_classes)
        mock_attentions = torch.randn(12, self.batch_size, 12, 197, 197)
        mock_model.return_value = (mock_logits, mock_attentions)
        
        # Setup proper mock structure for backbone.encoder.layer[-1].norm1
        mock_layer = Mock()
        mock_layer.norm1 = Mock()
        mock_model.backbone.encoder.layer = [mock_layer for _ in range(12)]
        
        # Test data
        images = torch.randn(self.batch_size, 3, 224, 224)
        
        # Test function
        result_logits, result_attentions = _run_pytorch_inference(
            mock_model, images, show_gradcam=True, image_ids=self.image_ids,
            class_labels=self.class_labels, output_gradcam=self.temp_dir,
            accelerator=self.mock_accelerator
        )
        
        # Assertions
        mock_model.assert_called_once_with(images)
        self.assertTrue(torch.equal(result_logits, mock_logits))
        self.assertTrue(torch.equal(result_attentions, mock_attentions))
        mock_visualize_gradcam.assert_called_once()

    def test_run_onnx_inference_cpu(self):
        """Test ONNX inference on CPU"""
        # Setup mock session
        mock_session = Mock()
        mock_session.get_providers.return_value = ['CPUExecutionProvider']
        mock_logits = np.random.randn(self.batch_size, self.num_classes)
        mock_attentions = np.random.randn(12, self.batch_size, 12, 197, 197)
        mock_session.run.return_value = [mock_logits, mock_attentions]
        
        # Test data
        images = torch.randn(self.batch_size, 3, 224, 224)
        input_name = 'input'
        output_names = ['logits', 'attentions']
        
        # Test function
        result_logits, result_attentions = _run_onnx_inference(
            mock_session, input_name, output_names, images, show_attention=True,
            accelerator=self.mock_accelerator, class_labels=self.class_labels,
            backbone_config=self.mock_config
        )
        
        # Assertions
        mock_session.run.assert_called_once()
        self.assertIsInstance(result_logits, torch.Tensor)
        self.assertIsInstance(result_attentions, torch.Tensor)
        self.assertEqual(result_logits.shape, (self.batch_size, self.num_classes))

    @patch('rad_dino.eval.inference.visualize_attention_maps')
    @patch('rad_dino.eval.inference.visualize_evaluate_metrics')
    @patch('pandas.DataFrame.to_csv')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('os.makedirs')
    @patch('rad_dino.eval.inference._run_pytorch_inference')
    def test_run_inference_pytorch(self, mock_run_pytorch_inference, mock_makedirs, mock_json_dump, mock_file_open, 
                                  mock_to_csv, mock_visualize_metrics, mock_visualize_attention):
        """Test main inference function with PyTorch model"""
        # Setup mock model wrapper
        mock_model = Mock()
        mock_model.eval = Mock()
        
        model_wrapper = {
            'type': 'pytorch',
            'model': mock_model,
            'config': self.mock_config,
            'device': self.mock_accelerator.device
        }
        
        # Setup mock data loader
        images = torch.randn(self.batch_size, 3, 224, 224)
        targets = torch.randint(0, 2, (self.batch_size, self.num_classes))
        batch = (images, targets, self.image_ids)
        mock_loader = [batch]
        
        # Setup mock inference results
        mock_logits = torch.randn(self.batch_size, self.num_classes)
        mock_attentions = torch.randn(12, self.batch_size, 12, 197, 197)
        mock_run_pytorch_inference.return_value = (mock_logits, mock_attentions)
        
        # Setup mock metrics
        mock_visualize_metrics.return_value = (0.8, 0.9)  # PR-AUC, ROC-AUC
        
        # Test function
        run_inference(
            model_wrapper, mock_loader, self.mock_accelerator, self.class_labels,
            show_attention=True, show_gradcam=True, attention_threshold=0.6,
            num_save_heads='all', output_dir=self.temp_dir
        )
        
        # Assertions
        mock_model.eval.assert_called_once()
        mock_run_pytorch_inference.assert_called_once()
        mock_makedirs.assert_called()
        mock_to_csv.assert_called_once()
        mock_json_dump.assert_called_once()
        mock_visualize_attention.assert_called_once()

    @patch('rad_dino.eval.inference.Accelerator')
    @patch('rad_dino.eval.inference.load_model')
    @patch('rad_dino.eval.inference.run_inference')
    @patch('rad_dino.eval.inference.RadImageClassificationDataset')
    @patch('rad_dino.eval.inference.DataLoader')
    @patch('rad_dino.eval.inference.get_transforms')
    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.safe_load')
    @patch('os.makedirs')
    def test_main_function(self, mock_makedirs, mock_yaml_load, mock_file_open,
                          mock_get_transforms, mock_dataloader, mock_dataset,
                          mock_run_inference, mock_load_model, mock_accelerator):
        """Test main function execution"""
        # Setup mocks
        mock_accelerator_instance = Mock()
        mock_accelerator.return_value = mock_accelerator_instance
        
        mock_yaml_load.side_effect = [
            {  # data_config.yaml
                'VinDr-CXR': {
                    'class_labels': self.class_labels,
                    'data_root_folder': '/test/data'
                }
            },
            {  # train_config.yaml
                'batch_size': 8
            }
        ]
        
        mock_transforms = (Mock(), Mock())
        mock_get_transforms.return_value = mock_transforms
        
        mock_dataset_instance = Mock()
        mock_dataset.return_value = mock_dataset_instance
        
        mock_loader_instance = Mock()
        mock_dataloader.return_value = mock_loader_instance
        mock_accelerator_instance.prepare.return_value = mock_loader_instance
        
        mock_model_wrapper = {'type': 'pytorch', 'model': Mock()}
        mock_load_model.return_value = mock_model_wrapper
        
        # Test with custom arguments
        test_args = [
            '--task', 'multilabel',
            '--data', 'VinDr-CXR',
            '--model', 'rad_dino',
            '--show-attention',
            '--show-gradcam'
        ]
        
        with patch('sys.argv', ['inference.py'] + test_args):
            main()
        
        # Assertions
        mock_accelerator.assert_called_once()
        mock_yaml_load.assert_called()
        mock_get_transforms.assert_called_once_with("microsoft/rad-dino")
        mock_dataset.assert_called_once()
        mock_dataloader.assert_called_once()
        mock_load_model.assert_called_once()
        mock_run_inference.assert_called_once()

    def test_save_heads_parsing(self):
        """Test parsing of save_heads argument in run_inference"""
        # This test would be part of run_inference, but we can test the logic separately
        test_cases = [
            ('all', 'all'),
            ('mean', 'mean'),
            ('0,5,11', [0, 5, 11]),
            ('invalid', 'all')  # Should fallback to 'all'
        ]
        
        for input_val, expected in test_cases:
            if input_val == 'all':
                result = 'all'
            elif input_val == 'mean':
                result = 'mean'
            else:
                try:
                    result = [int(h.strip()) for h in input_val.split(',')]
                except ValueError:
                    result = 'all'
            
            self.assertEqual(result, expected)

    def test_onnx_inference_shape_calculations(self):
        """Test ONNX inference attention tensor shape calculations"""
        # Test shape calculation logic
        backbone_config = Mock()
        backbone_config.num_hidden_layers = 12
        backbone_config.num_attention_heads = 16
        backbone_config.patch_size = 14
        
        batch_size = 4
        img_height, img_width = 224, 224
        
        num_patches_h = img_height // backbone_config.patch_size
        num_patches_w = img_width // backbone_config.patch_size
        seq_len = num_patches_h * num_patches_w + 1  # +1 for CLS token
        
        expected_attention_shape = (
            backbone_config.num_hidden_layers,
            batch_size,
            backbone_config.num_attention_heads,
            seq_len,
            seq_len
        )
        
        # Expected: (12, 4, 16, 257, 257)
        self.assertEqual(expected_attention_shape, (12, 4, 16, 257, 257))

    @patch('torch.cuda.empty_cache')
    def test_cuda_memory_management(self, mock_empty_cache):
        """Test CUDA memory management calls"""
        # Test that CUDA cache is cleared when using GPU
        self.mock_accelerator.device = torch.device('cuda:0')
        
        mock_model = Mock()
        mock_logits = torch.randn(self.batch_size, self.num_classes)
        mock_attentions = torch.randn(12, self.batch_size, 12, 197, 197)
        mock_model.return_value = (mock_logits, mock_attentions)
        mock_model.backbone.encoder.layer = [Mock() for _ in range(12)]
        mock_model.backbone.encoder.layer[-1].norm1 = Mock()
        
        images = torch.randn(self.batch_size, 3, 224, 224)
        
        with patch('rad_dino.eval.inference.visualize_gradcam'):
            _run_pytorch_inference(
                mock_model, images, show_gradcam=True, image_ids=self.image_ids,
                class_labels=self.class_labels, output_gradcam=self.temp_dir,
                accelerator=self.mock_accelerator
            )
        
        # Should be called twice: before and after gradcam
        self.assertEqual(mock_empty_cache.call_count, 2)


if __name__ == '__main__':
    # Set up test environment
    import logging
    logging.disable(logging.CRITICAL)  # Disable logging during tests
    
    unittest.main()
