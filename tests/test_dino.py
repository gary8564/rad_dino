import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, Mock
from rad_dino.models.dino import DinoClassifier


class TestDinoClassifier(unittest.TestCase):
    """Test the DinoClassifier class."""
    
    def setUp(self):
        self.num_classes = 2
        self.batch_size = 4
        self.embed_dim = 768  # DINO default feature dimension
        
        # Create a mock backbone
        self.mock_backbone = MagicMock()
        self.mock_backbone.config = MagicMock()
        self.mock_backbone.config.hidden_size = self.embed_dim
        self.mock_backbone.config.num_hidden_layers = 12
        self.mock_backbone.config.num_attention_heads = 12
        
        # Mock the forward method to return proper outputs
        def mock_forward(pixel_values, output_attentions=False, return_dict=True):
            batch_size = pixel_values.shape[0]
            seq_len = 197  # Standard for 224x224 images with patch size 14
            num_layers = 12
            
            # Create last_hidden_state with CLS token
            last_hidden_state = torch.randn(batch_size, seq_len, self.embed_dim)
            
            # Create attention maps for each layer
            attentions = []
            for _ in range(num_layers):
                attention = torch.randn(batch_size, 12, seq_len, seq_len)  # 12 attention heads
                attention = torch.softmax(attention, dim=-1)  # Normalize attention weights
                attentions.append(attention)
            
            # Create a mock object with the required attributes
            mock_output = Mock()
            mock_output.last_hidden_state = last_hidden_state
            mock_output.attentions = attentions
            
            return mock_output
        
        self.mock_backbone.side_effect = mock_forward
        
    def test_initialization(self):
        """Test DinoClassifier initialization."""
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.embed_dim, self.embed_dim)
        self.assertFalse(model.multi_view)
        
    def test_single_view_forward(self):
        """Test DinoClassifier forward pass for single view."""
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
            return_attentions=True
        )
        
        x = torch.randn(self.batch_size, 3, 224, 224)
        logits, attentions = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attentions.shape, (12, self.batch_size, 12, 197, 197))
        self.mock_backbone.assert_called_once()
        
    def test_multi_view_initialization(self):
        """Test DinoClassifier initialization with multi-view."""
        model = DinoClassifier(
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
        
    def test_multi_view_forward(self):
        """Test DinoClassifier forward pass for multi-view."""
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean",
            return_attentions=True
        )
        
        x = torch.randn(self.batch_size, 4, 3, 224, 224)
        logits, attentions = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attentions.shape, (12, self.batch_size, 4, 12, 197, 197))
        
    def test_fusion_strategies(self):
        """Test different fusion strategies."""
        fusion_strategies = ["mean", "weighted_mean", "mlp_adapter"]
        
        for strategy in fusion_strategies:
            with self.subTest(strategy=strategy):
                model = DinoClassifier(
                    backbone=self.mock_backbone,
                    num_classes=self.num_classes,
                    multi_view=True,
                    num_views=4,
                    view_fusion_type=strategy,
                    return_attentions=True
                )
                
                x = torch.randn(self.batch_size, 4, 3, 224, 224)
                logits, attentions = model(x)
                
                self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
                self.assertEqual(attentions.shape, (12, self.batch_size, 4, 12, 197, 197))
                self.assertTrue(torch.all(torch.isfinite(logits)))
        
    def test_invalid_fusion_type(self):
        """Test DinoClassifier with invalid fusion type."""
        with self.assertRaises(AssertionError):
            DinoClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type="invalid_fusion"
            )
        
    def test_missing_multi_view_params(self):
        """Test DinoClassifier with missing multi-view parameters."""
        with self.assertRaises(AssertionError):
            DinoClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=None,
                view_fusion_type="mean"
            )
        
        with self.assertRaises(AssertionError):
            DinoClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type=None
            )
        
    def test_input_validation(self):
        """Test DinoClassifier input validation."""
        # Single-view model with multi-view input
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        with self.assertRaises(ValueError):
            x = torch.randn(self.batch_size, 4, 3, 224, 224)
            model(x)
        
        # Multi-view model with single-view input
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        with self.assertRaises(ValueError):
            x = torch.randn(self.batch_size, 3, 224, 224)
            model(x)
        
    def test_head_initialization(self):
        """Test that classification head is properly initialized."""
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        self.assertIsNotNone(model.classifier)
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.in_features, self.embed_dim)
        self.assertEqual(model.classifier.out_features, self.num_classes)
        
    def test_strategy_dictionaries(self):
        """Test that strategy dictionaries are properly initialized."""
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        # Check that strategy dictionaries exist and are callable
        self.assertIsNotNone(model.input_reshape_strategies)
        self.assertIsNotNone(model.view_fusion_strategies)
        self.assertIsNotNone(model.normalization_strategies)
        
        self.assertTrue(callable(model.input_reshape_strategies[False]))
        self.assertTrue(callable(model.input_reshape_strategies[True]))
        
    def test_full_pipeline_single_view(self):
        """Test complete pipeline with single-view input."""
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
            return_attentions=True
        )
        
        x = torch.randn(self.batch_size, 3, 224, 224)
        logits, attentions = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attentions.shape, (12, self.batch_size, 12, 197, 197))
        self.assertTrue(torch.all(torch.isfinite(logits)))
        self.assertTrue(torch.all(torch.isfinite(attentions)))
        
    def test_full_pipeline_multi_view(self):
        """Test complete pipeline with multi-view input."""
        model = DinoClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean",
            return_attentions=True
        )
        
        x = torch.randn(self.batch_size, 4, 3, 224, 224)
        logits, attentions = model(x)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(attentions.shape, (12, self.batch_size, 4, 12, 197, 197))
        self.assertTrue(torch.all(torch.isfinite(logits)))
        self.assertTrue(torch.all(torch.isfinite(attentions)))


if __name__ == "__main__":
    unittest.main()
