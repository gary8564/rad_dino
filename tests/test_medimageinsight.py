import os
import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock
from rad_dino.models.medimageinsight import MedImageInsightClassifier


class TestMedImageInsightClassifier(unittest.TestCase):
    """Test the MedImageInsightClassifier class with a mock UniCL backbone."""

    def setUp(self):
        self.num_classes = 5
        self.batch_size = 2
        self.embed_dim = 1024  # MedImageInsight projection dim
        self.image_size = 480  # MedImageInsight input size

        # Create a mock backbone that mimics the UniCLModel interface.
        # The key attributes/methods the classifier relies on:
        #   - backbone.image_projection  (nn.Parameter with shape [2048, 1024])
        #   - backbone.encode_image(x, norm=True)  -> [B, 1024]
        self.mock_backbone = MagicMock(spec=[])
        # image_projection is accessed for its .shape to infer embed_dim
        self.mock_backbone.image_projection = torch.randn(2048, self.embed_dim)

        # encode_image must return a real tensor so the classifier head can
        # compute logits via matrix multiplication.
        def _mock_encode_image(x, norm=True):
            batch_size = x.shape[0]
            features = torch.randn(batch_size, self.embed_dim)
            if norm:
                features = features / features.norm(dim=-1, keepdim=True)
            return features

        self.mock_backbone.encode_image = _mock_encode_image

    def test_initialization(self):
        """Test MedImageInsightClassifier initializes correctly."""
        model = MedImageInsightClassifier(
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
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean",
        )
        self.assertTrue(model.multi_view)
        self.assertEqual(model.num_views, 4)
        self.assertEqual(model.view_fusion_type, "mean")

    def test_single_view_forward(self):
        """Test forward pass with a single-view input tensor."""
        model = MedImageInsightClassifier(
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
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        # Use a simple all-ones tensor
        x = torch.ones(1, 3, self.image_size, self.image_size)
        logits, attention_maps = model(x)

        self.assertEqual(logits.shape, (1, self.num_classes))
        self.assertIsNone(attention_maps)
        self.assertTrue(torch.all(torch.isfinite(logits)))

    def test_multi_view_forward_mean(self):
        """Test multi-view forward pass with mean fusion."""
        model = MedImageInsightClassifier(
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
        model = MedImageInsightClassifier(
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
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mlp_adapter",
            adapter_dim=512,
            view_fusion_hidden_dim=512,
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
                model = MedImageInsightClassifier(
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

    def test_invalid_fusion_type(self):
        """Test that an invalid fusion type raises AssertionError."""
        with self.assertRaises(AssertionError):
            MedImageInsightClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type="invalid_fusion",
            )

    def test_missing_multi_view_params(self):
        """Test that missing multi-view params raise AssertionError."""
        with self.assertRaises(AssertionError):
            MedImageInsightClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=None,
                view_fusion_type="mean",
            )
        with self.assertRaises(AssertionError):
            MedImageInsightClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=True,
                num_views=4,
                view_fusion_type=None,
            )

    def test_single_view_rejects_multi_view_input(self):
        """Single-view model should reject 5-D input."""
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 4, 3, self.image_size, self.image_size)
        with self.assertRaises(ValueError):
            model(x)

    def test_multi_view_rejects_single_view_input(self):
        """Multi-view model should reject 4-D input."""
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean",
        )
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        with self.assertRaises(ValueError):
            model(x)

    def test_head_initialization(self):
        """Test that the classification head is properly initialised."""
        model = MedImageInsightClassifier(
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
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        self.assertIsNotNone(model.input_reshape_strategies)
        self.assertIsNotNone(model.view_fusion_strategies)
        self.assertIsNotNone(model.normalization_strategies)
        self.assertTrue(callable(model.input_reshape_strategies[False]))
        self.assertTrue(callable(model.input_reshape_strategies[True]))

    def test_gradient_flow(self):
        """Verify that gradients flow through the classifier head."""
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(1, 3, self.image_size, self.image_size)
        logits, _ = model(x)
        loss = logits.sum()
        loss.backward()

        # The classifier head should have gradients
        self.assertIsNotNone(model.classifier.weight.grad)
        self.assertIsNotNone(model.classifier.bias.grad)

    def test_binary_classification(self):
        """Test forward pass with binary classification (1 output)."""
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=1,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size)
        logits, _ = model(x)
        self.assertEqual(logits.shape, (self.batch_size, 1))

    def test_return_attentions_warning(self):
        """return_attentions=True should log a warning but not error."""
        import logging
        with self.assertLogs("rad_dino.models.medimageinsight", level=logging.WARNING) as cm:
            model = MedImageInsightClassifier(
                backbone=self.mock_backbone,
                num_classes=self.num_classes,
                multi_view=False,
                return_attentions=True,
            )
        self.assertTrue(any("does not support attention" in msg for msg in cm.output))
        x = torch.randn(1, 3, self.image_size, self.image_size)
        _, attn = model(x)
        self.assertIsNone(attn)

    def test_extract_stage_feature_maps_no_image_encoder(self):
        """extract_stage_feature_maps returns None when backbone lacks image_encoder."""
        model = MedImageInsightClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(1, 3, self.image_size, self.image_size)
        result = model.extract_stage_feature_maps(x)
        self.assertIsNone(result)

    def test_extract_stage_feature_maps_with_mock_blocks(self):
        """extract_stage_feature_maps captures features from mock blocks."""
        # Build a backbone with image_encoder.blocks that returns (features, size)
        mock_block = torch.nn.Linear(10, 10)
        _original_forward = mock_block.forward

        spatial_h, spatial_w = 4, 4
        embed_dim = 10

        def _block_forward(x):
            B = x.shape[0]
            feat = torch.randn(B, spatial_h * spatial_w, embed_dim)
            return feat, (spatial_h, spatial_w)

        mock_block.forward = _block_forward

        backbone = MagicMock(spec=[])
        backbone.image_projection = torch.randn(2048, self.embed_dim)

        image_encoder = MagicMock()
        image_encoder.blocks = [mock_block]
        backbone.image_encoder = image_encoder

        def _mock_encode(x, norm=True):
            for block in backbone.image_encoder.blocks:
                block(x)
            return torch.randn(x.shape[0], self.embed_dim)

        backbone.encode_image = _mock_encode

        model = MedImageInsightClassifier(
            backbone=backbone,
            num_classes=self.num_classes,
            multi_view=False,
        )
        x = torch.randn(1, 3, self.image_size, self.image_size)
        result = model.extract_stage_feature_maps(x)

        self.assertIsNotNone(result)
        self.assertIn(0, result)
        self.assertEqual(result[0]["features"].shape[-1], embed_dim)
        self.assertEqual(result[0]["spatial_size"], (spatial_h, spatial_w))


class TestMedImageInsight(unittest.TestCase):
    """
    Test on real MedImageInsight UniCL backbone
    """

    DEFAULT_MODEL_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "rad_dino", "models", "MedImageInsights")
    )

    @classmethod
    def setUpClass(cls):
        cls.model_dir = os.environ.get("MEDIMAGEINSIGHT_PATH", cls.DEFAULT_MODEL_DIR)
        if not os.path.isdir(cls.model_dir):
            raise unittest.SkipTest(
                f"MedImageInsight repo not found at {cls.model_dir}. "
                "Set MEDIMAGEINSIGHT_PATH or clone the repo."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available; skipping integration test.")

        from rad_dino.models.medimageinsight import load_medimageinsight_model
        cls.device = "cuda"
        cls.backbone = load_medimageinsight_model(cls.model_dir, device=cls.device)

    def test_real_backbone_single_view_forward(self):
        """Forward pass through backbone with a small random input."""
        num_classes = 1  # binary, like RSNA-Pneumonia
        model = MedImageInsightClassifier(
            backbone=self.backbone,
            num_classes=num_classes,
            multi_view=False,
        ).to(self.device)

        # Minimal batch: 1 image, 480x480
        x = torch.randn(1, 3, 480, 480, device=self.device)
        with torch.no_grad():
            logits, attn = model(x)

        self.assertEqual(logits.shape, (1, num_classes))
        self.assertIsNone(attn)
        self.assertTrue(torch.all(torch.isfinite(logits)))


if __name__ == "__main__":
    unittest.main()
