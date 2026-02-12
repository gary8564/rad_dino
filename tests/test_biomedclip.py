import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from rad_dino.models.biomedclip import BiomedCLIPClassifier


class TestBiomedCLIPClassifier(unittest.TestCase):
    """Test the BiomedCLIPClassifier class with a mock open_clip backbone."""

    def setUp(self):
        self.num_classes = 5
        self.batch_size = 2
        self.embed_dim = 512  # BiomedCLIP projection dimension
        self.image_size = 224  # BiomedCLIP input size

        # Create a mock backbone that mimics the open_clip model interface.
        # The classifier relies on:
        #   - backbone.encode_image(x) -> [B, 512] (raw, un-normalised)
        self.mock_backbone = MagicMock(spec=[])

        def _mock_encode_image(x):
            batch_size = x.shape[0]
            # Return un-normalised features; the classifier L2-normalises them
            return torch.randn(batch_size, self.embed_dim)

        self.mock_backbone.encode_image = _mock_encode_image

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_initialization(self):
        """Test BiomedCLIPClassifier initializes correctly."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.embed_dim, self.embed_dim)
        self.assertFalse(model.multi_view)

    def test_multi_view_initialization(self):
        """Test initialization with multi-view enabled."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean",
        )
        self.assertTrue(model.multi_view)
        self.assertEqual(model.num_views, 4)
        self.assertEqual(model.view_fusion_type, "mean")

    # ------------------------------------------------------------------
    # Single-view forward
    # ------------------------------------------------------------------

    def test_single_view_forward(self):
        """Test forward pass with a single-view input tensor."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        logits, attention_maps = model(x)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIsNone(attention_maps)
        self.assertTrue(torch.all(torch.isfinite(logits)))

    def test_single_view_forward_simple_matrix(self):
        """Test forward pass with a simple deterministic input matrix."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.ones(1, 3, self.image_size, self.image_size)
        logits, attention_maps = model(x)

        self.assertEqual(logits.shape, (1, self.num_classes))
        self.assertIsNone(attention_maps)
        self.assertTrue(torch.all(torch.isfinite(logits)))

    # ------------------------------------------------------------------
    # Multi-view forward (all fusion strategies)
    # ------------------------------------------------------------------

    def test_multi_view_forward_mean(self):
        """Test multi-view forward pass with mean fusion."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean",
        )
        x = torch.randn(self.batch_size, 4, 3, self.image_size, self.image_size)
        logits, attention_maps = model(x)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIsNone(attention_maps)
        self.assertTrue(torch.all(torch.isfinite(logits)))

    def test_multi_view_forward_weighted_mean(self):
        """Test multi-view forward pass with weighted-mean fusion."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="weighted_mean",
        )
        x = torch.randn(self.batch_size, 4, 3, self.image_size, self.image_size)
        logits, attention_maps = model(x)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIsNone(attention_maps)
        self.assertTrue(torch.all(torch.isfinite(logits)))

    def test_multi_view_forward_mlp_adapter(self):
        """Test multi-view forward pass with MLP adapter fusion."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mlp_adapter",
            adapter_dim=256,
            view_fusion_hidden_dim=256,
        )
        x = torch.randn(self.batch_size, 4, 3, self.image_size, self.image_size)
        logits, attention_maps = model(x)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertIsNone(attention_maps)
        self.assertTrue(torch.all(torch.isfinite(logits)))

    def test_all_fusion_strategies(self):
        """Test that all fusion strategies produce correct output shapes."""
        fusion_strategies = ["mean", "weighted_mean", "mlp_adapter"]
        for strategy in fusion_strategies:
            with self.subTest(strategy=strategy):
                model = BiomedCLIPClassifier(
                    backbone=self.mock_backbone,
                    num_classes=self.num_classes,
                    multi_view=True,
                    num_views=4,
                    view_fusion_type=strategy,
                )
                x = torch.randn(self.batch_size, 4, 3, self.image_size, self.image_size)
                logits, attention_maps = model(x)
                self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
                self.assertIsNone(attention_maps)
                self.assertTrue(torch.all(torch.isfinite(logits)))

    # ------------------------------------------------------------------
    # Validation / error handling
    # ------------------------------------------------------------------

    def test_invalid_fusion_type(self):
        """Test that an invalid fusion type raises AssertionError."""
        with self.assertRaises(AssertionError):
            BiomedCLIPClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type="invalid_fusion",
            )

    def test_missing_multi_view_params(self):
        """Test that missing multi-view params raise AssertionError."""
        with self.assertRaises(AssertionError):
            BiomedCLIPClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=None,
                view_fusion_type="mean",
            )
        with self.assertRaises(AssertionError):
            BiomedCLIPClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type=None,
            )

    def test_single_view_rejects_multi_view_input(self):
        """Single-view model should reject 5-D input."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 4, 3, self.image_size, self.image_size)
        with self.assertRaises(ValueError):
            model(x)

    def test_multi_view_rejects_single_view_input(self):
        """Multi-view model should reject 4-D input."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean",
        )
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        with self.assertRaises(ValueError):
            model(x)

    # ------------------------------------------------------------------
    # Head / strategy initialization
    # ------------------------------------------------------------------

    def test_head_initialization(self):
        """Test that the classification head is properly initialised."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        self.assertIsNotNone(model.classifier)
        self.assertIsInstance(model.classifier, nn.Linear)
        self.assertEqual(model.classifier.in_features, self.embed_dim)
        self.assertEqual(model.classifier.out_features, self.num_classes)

    def test_strategy_dictionaries(self):
        """Test that strategy dictionaries are properly initialised."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        self.assertIsNotNone(model.input_reshape_strategies)
        self.assertIsNotNone(model.view_fusion_strategies)
        self.assertIsNotNone(model.normalization_strategies)
        self.assertTrue(callable(model.input_reshape_strategies[False]))
        self.assertTrue(callable(model.input_reshape_strategies[True]))

    # ------------------------------------------------------------------
    # Gradient flow
    # ------------------------------------------------------------------

    def test_gradient_flow(self):
        """Verify that gradients flow through the classifier head."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(1, 3, self.image_size, self.image_size)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNotNone(model.classifier.bias.grad)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_binary_classification(self):
        """Test forward pass with binary classification (1 output)."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=1,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        logits, _ = model(x)
        self.assertEqual(logits.shape, (self.batch_size, 1))

    def test_embed_dim_class_constant(self):
        """Test that EMBED_DIM class constant is set correctly."""
        self.assertEqual(BiomedCLIPClassifier.EMBED_DIM, 512)

    def test_feature_normalization(self):
        """Test that extract_features returns L2-normalised features."""
        model = BiomedCLIPClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        features, attn = model.extract_features(x)

        self.assertEqual(features.shape, (self.batch_size, self.embed_dim))
        self.assertIsNone(attn)
        # Check L2 norm is ~1.0 per sample
        norms = features.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
