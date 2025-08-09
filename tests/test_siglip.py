import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from transformers import AutoConfig
from rad_dino.models.siglip import MedSigClassifier


class TestMedSigClassifier(unittest.TestCase):
    """Test the MedSigClassifier class."""
    
    def setUp(self):
        self.num_classes = 2
        self.batch_size = 4
        self.feat_dim = 768  # MedSigLIP default feature dimension
        
        # Create a mock backbone with proper config
        self.mock_backbone = Mock()
        self.mock_backbone.config = Mock()
        self.mock_backbone.config.text_config = Mock()
        self.mock_backbone.config.text_config.projection_size = self.feat_dim
        
        # Mock the vision_model to return proper tensor shapes
        self.mock_backbone.vision_model = Mock()
        def mock_vision_model_forward(pixel_values, output_attentions=False, return_dict=False):
            # Return features with proper shape based on input
            batch_size = pixel_values.shape[0]
            # Create a mock output with pooler_output and attentions
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(batch_size, self.feat_dim)
            if output_attentions:
                # Create mock attention maps for 27 layers
                mock_output.attentions = [torch.randn(batch_size, 16, 1024, 1024) for _ in range(27)]
            return mock_output
        self.mock_backbone.vision_model.side_effect = mock_vision_model_forward
        
    def test_medsig_classifier_initialization(self):
        """Test MedSigClassifier initialization."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.feat_dim, self.feat_dim)
        self.assertFalse(model.multi_view)
        
    def test_medsig_classifier_single_view_forward(self):
        """Test MedSigClassifier forward pass for single view."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        x = torch.randn(self.batch_size, 3, 224, 224)
        logits, attention_maps = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attention_maps.shape, (27, self.batch_size, 16, 1024, 1024))
        self.mock_backbone.vision_model.assert_called_once()
        
    def test_medsig_classifier_multi_view_initialization(self):
        """Test MedSigClassifier initialization with multi-view."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertTrue(model.multi_view)
        self.assertEqual(model.num_views, 4)
        self.assertEqual(model.view_fusion_type, "mean")
        
    def test_medsig_classifier_multi_view_forward(self):
        """Test MedSigClassifier forward pass for multi-view."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        x = torch.randn(self.batch_size, 4, 3, 224, 224)  # [batch, views, channels, height, width]
        logits, attention_maps = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attention_maps.shape, (27, self.batch_size, 4, 16, 1024, 1024))
        # Should be called once with reshaped input (batch_size * num_views)
        self.mock_backbone.vision_model.assert_called_once()
        
    def test_medsig_classifier_weighted_mean_fusion(self):
        """Test MedSigClassifier with weighted mean fusion."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="weighted_mean"
        )
        
        x = torch.randn(self.batch_size, 4, 3, 224, 224)
        logits, attention_maps = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attention_maps.shape, (27, self.batch_size, 4, 16, 1024, 1024))
        self.assertEqual(model.view_fusion_type, "weighted_mean")
        self.assertIsNotNone(model.view_scores)
        self.assertIsNotNone(model.fusion_layer)
        
    def test_medsig_classifier_mlp_adapter_fusion(self):
        """Test MedSigClassifier with MLP adapter fusion."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mlp_adapter",
            adapter_dim=256,
            view_fusion_hidden_dim=512
        )
        
        x = torch.randn(self.batch_size, 4, 3, 224, 224)
        logits, attention_maps = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attention_maps.shape, (27, self.batch_size, 4, 16, 1024, 1024))
        self.assertEqual(model.view_fusion_type, "mlp_adapter")
        self.assertIsNotNone(model.view_adapters)
        self.assertIsNotNone(model.fusion_layer)
        
    def test_medsig_classifier_invalid_fusion_type(self):
        """Test MedSigClassifier with invalid fusion type."""
        with self.assertRaises(AssertionError):
            MedSigClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type="invalid_fusion"
            )
        
    def test_medsig_classifier_missing_multi_view_params(self):
        """Test MedSigClassifier with missing multi-view parameters."""
        with self.assertRaises(AssertionError):
            MedSigClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=None,
                view_fusion_type="mean"
            )
        
        with self.assertRaises(AssertionError):
            MedSigClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type=None
            )
        
    def test_medsig_classifier_input_validation(self):
        """Test MedSigClassifier input validation."""
        # Single-view model with multi-view input
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        with self.assertRaises(ValueError):
            x = torch.randn(self.batch_size, 4, 3, 224, 224)  # Multi-view input
            model(x)
        
        # Multi-view model with single-view input
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        with self.assertRaises(ValueError):
            x = torch.randn(self.batch_size, 3, 224, 224)  # Single-view input
            model(x)
        
    def test_medsig_classifier_feature_normalization(self):
        """Test MedSigClassifier feature normalization."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        # Test that features are properly normalized
        features = torch.randn(self.batch_size, self.feat_dim)
        normalized_features = model._single_view_normalization(features)
        
        # Check that normalization doesn't change shape
        self.assertEqual(normalized_features.shape, features.shape)
        
    def test_medsig_classifier_layer_norm(self):
        """Test MedSigClassifier layer normalization."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        # Test that layer norm is properly initialized for multi-view models
        self.assertTrue(hasattr(model, 'layer_norm'))
        self.assertIsInstance(model.layer_norm, nn.LayerNorm)
        
        # Test layer norm forward pass
        features = torch.randn(self.batch_size, self.feat_dim)
        normalized_features = model.layer_norm(features)
        
        # Check that layer norm doesn't change shape
        self.assertEqual(normalized_features.shape, features.shape)
        
    def test_medsig_classifier_head_initialization(self):
        """Test MedSigClassifier classification head initialization."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        # Check that head exists and has correct output dimension
        self.assertIsNotNone(model.head)
        # Check that the head is a Sequential with the correct final layer
        self.assertIsInstance(model.head, nn.Sequential)
        self.assertIsInstance(model.head[-1], nn.Linear)
        self.assertEqual(model.head[-1].out_features, self.num_classes)
        
    def test_medsig_classifier_strategy_dictionaries(self):
        """Test MedSigClassifier strategy dictionary initialization."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        # Check that strategy dictionaries are properly initialized
        self.assertIsNotNone(model.input_reshape_strategies)
        self.assertIsNotNone(model.normalization_strategies)
        self.assertIsNotNone(model.view_fusion_strategies)
        
        # Check that all expected strategies are present
        self.assertIn(True, model.input_reshape_strategies)
        self.assertIn(False, model.input_reshape_strategies)
        self.assertIn(True, model.normalization_strategies)
        self.assertIn(False, model.normalization_strategies)
        self.assertIn("mean", model.view_fusion_strategies)
        self.assertIn("weighted_mean", model.view_fusion_strategies)
        self.assertIn("mlp_adapter", model.view_fusion_strategies)


class TestMedSigClassifierIntegration(unittest.TestCase):
    """Integration tests for MedSigClassifier."""
    
    def setUp(self):
        self.num_classes = 3
        self.batch_size = 2
        self.feat_dim = 768
        
        # Create a more realistic mock backbone
        self.mock_backbone = Mock()
        self.mock_backbone.config = Mock()
        self.mock_backbone.config.text_config = Mock()
        self.mock_backbone.config.text_config.projection_size = self.feat_dim
        
        # Mock the vision_model
        self.mock_backbone.vision_model = Mock()
        def mock_vision_model_forward(pixel_values, output_attentions=False, return_dict=False):
            batch_size = pixel_values.shape[0]
            # Create a mock output with pooler_output and attentions
            mock_output = Mock()
            mock_output.pooler_output = torch.randn(batch_size, self.feat_dim)
            if output_attentions:
                # Create mock attention maps for 27 layers
                mock_output.attentions = [torch.randn(batch_size, 16, 1024, 1024) for _ in range(27)]
            return mock_output
        self.mock_backbone.vision_model.side_effect = mock_vision_model_forward
        
    def test_medsig_classifier_full_pipeline_single_view(self):
        """Test complete MedSigClassifier pipeline for single view."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        x = torch.randn(self.batch_size, 3, 224, 224)
        logits, attention_maps = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attention_maps.shape, (27, self.batch_size, 16, 1024, 1024))
        self.mock_backbone.vision_model.assert_called_once()
        
    def test_medsig_classifier_full_pipeline_multi_view(self):
        """Test complete MedSigClassifier pipeline for multi-view."""
        model = MedSigClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        x = torch.randn(self.batch_size, 4, 3, 224, 224)
        logits, attention_maps = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attention_maps.shape, (27, self.batch_size, 4, 16, 1024, 1024))
        self.mock_backbone.vision_model.assert_called_once()
        
    def test_medsig_classifier_different_fusion_strategies(self):
        """Test MedSigClassifier with different fusion strategies."""
        fusion_strategies = ["mean", "weighted_mean", "mlp_adapter"]
        
        for fusion_type in fusion_strategies:
            with self.subTest(fusion_type=fusion_type):
                model = MedSigClassifier(
                    backbone=self.mock_backbone,
                    num_classes=self.num_classes,
                    multi_view=True,
                    num_views=4,
                    view_fusion_type=fusion_type
                )
                
                x = torch.randn(self.batch_size, 4, 3, 224, 224)
                logits, attention_maps = model(x)
                
                self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
                self.assertEqual(attention_maps.shape, (27, self.batch_size, 4, 16, 1024, 1024))
                self.assertEqual(model.view_fusion_type, fusion_type)


if __name__ == '__main__':
    unittest.main() 