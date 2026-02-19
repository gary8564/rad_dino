import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from rad_dino.models.ark import SwinTransformer, ArkClassifier, load_prtrained_ark_model


class TestSwinTransformer(unittest.TestCase):
    """Test the SwinTransformer class."""
    
    def setUp(self):
        self.num_classes_list = [14, 14, 14, 3, 6, 1]  # List of class counts for different tasks
        self.img_size = 768
        self.batch_size = 4
        
    def test_swin_transformer_initialization(self):
        """Test SwinTransformer initialization."""
        model = SwinTransformer(
            num_classes_list=self.num_classes_list,
            img_size=self.img_size,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48)
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes_list, self.num_classes_list)
        self.assertEqual(model.num_features, 1376)  # Swin-Large with projector
        
    def test_swin_transformer_forward(self):
        """Test SwinTransformer forward pass."""
        model = SwinTransformer(
            num_classes_list=self.num_classes_list,
            img_size=self.img_size,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48)
        )
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        output, attention_maps = model(x)
        
        # Should return a list of outputs for each task
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), len(self.num_classes_list))
        for i, out in enumerate(output):
            self.assertEqual(out.shape, (self.batch_size, self.num_classes_list[i]))
        # return_attention=False by default, so attention_maps should be None
        self.assertIsNone(attention_maps)
        
    def test_swin_transformer_with_projector(self):
        """Test SwinTransformer with projector."""
        model = SwinTransformer(
            num_classes_list=self.num_classes_list,
            img_size=self.img_size,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            projector_features=1376,
            use_mlp=False
        )
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        output, attention_maps = model(x)
        
        self.assertIsInstance(output, list)
        self.assertEqual(len(output), len(self.num_classes_list))
        self.assertEqual(model.num_features, 1376)
        self.assertIsNone(attention_maps)
        
    def test_swin_transformer_single_head_forward(self):
        """Test SwinTransformer forward pass with specific head."""
        model = SwinTransformer(
            num_classes_list=self.num_classes_list,
            img_size=self.img_size,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48)
        )
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        # forward(x, head_n=0) returns (logits_for_head_0, attention_maps)
        logits, attention_maps = model(x, head_n=0)
        
        self.assertEqual(logits.shape, (self.batch_size, self.num_classes_list[0]))
        self.assertIsNone(attention_maps)
        
    def test_swin_transformer_generate_embeddings(self):
        """Test SwinTransformer generate_embeddings method."""
        model = SwinTransformer(
            num_classes_list=self.num_classes_list,
            img_size=self.img_size,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            projector_features=1376,
            use_mlp=False
        )
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        
        # Test embeddings after projection
        embeddings, attention_maps = model.generate_embeddings(x, after_proj=True)
        self.assertEqual(embeddings.shape, (self.batch_size, model.num_features))
        
        # Test embeddings before projection
        embeddings, attention_maps = model.generate_embeddings(x, after_proj=False)
        self.assertEqual(embeddings.shape, (self.batch_size, model.encoder_features))


class TestArkClassifier(unittest.TestCase):
    """Test the ArkClassifier class."""
    
    def setUp(self):
        self.num_classes = 2
        self.img_size = 768
        self.batch_size = 4
        self.embed_dim = 1376  # Ark projector features
        
        # Create a proper mock backbone
        self.mock_backbone = Mock()
        self.mock_backbone.num_features = self.embed_dim
        self.mock_backbone.encoder_features = 1536  # Swin-Large default
        self.mock_backbone.projector = nn.Linear(1536, 1376)
        
        # Mock the forward_features method
        def mock_forward_features(x):
            return torch.randn(x.shape[0], 1536)
        self.mock_backbone.forward_features = mock_forward_features
        
        # Mock the generate_embeddings method
        def mock_generate_embeddings(x, after_proj=True):
            if after_proj:
                return torch.randn(x.shape[0], 1376), []
            else:
                return torch.randn(x.shape[0], 1536), []
        self.mock_backbone.generate_embeddings = mock_generate_embeddings
        
    def test_ark_classifier_initialization(self):
        """Test ArkClassifier initialization."""
        model = ArkClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.num_classes, self.num_classes)
        self.assertEqual(model.embed_dim, 1536)  # Should use encoder_features when use_backbone_projector=False
        self.assertFalse(model.multi_view)
        
    def test_ark_classifier_single_view_forward(self):
        """Test ArkClassifier forward pass for single view."""
        model = ArkClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False
        )
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        output, attention_maps = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_ark_classifier_multi_view_initialization(self):
        """Test ArkClassifier initialization with multi-view."""
        model = ArkClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        self.assertIsNotNone(model)
        self.assertTrue(model.multi_view)
        self.assertEqual(model.num_views, 4)
        self.assertEqual(model.view_fusion_type, "mean")
        
    def test_ark_classifier_multi_view_forward(self):
        """Test ArkClassifier forward pass for multi-view."""
        # Mock backbone to return features for multiple views
        def mock_generate_embeddings_multi(x, after_proj=True):
            if after_proj:
                return torch.randn(x.shape[0], 1376), []
            else:
                return torch.randn(x.shape[0], 1536), []
        self.mock_backbone.generate_embeddings = mock_generate_embeddings_multi
        
        model = ArkClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mean"
        )
        
        x = torch.randn(self.batch_size, 4, 3, self.img_size, self.img_size)
        output, attention_maps = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_ark_classifier_weighted_mean_fusion(self):
        """Test ArkClassifier with weighted mean fusion."""
        def mock_generate_embeddings_multi(x, after_proj=True):
            if after_proj:
                return torch.randn(x.shape[0], 1376), []
            else:
                return torch.randn(x.shape[0], 1536), []
        self.mock_backbone.generate_embeddings = mock_generate_embeddings_multi
        
        model = ArkClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="weighted_mean"
        )
        
        x = torch.randn(self.batch_size, 4, 3, self.img_size, self.img_size)
        output, attention_maps = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_ark_classifier_mlp_adapter_fusion(self):
        """Test ArkClassifier with MLP adapter fusion."""
        def mock_generate_embeddings_multi(x, after_proj=True):
            if after_proj:
                return torch.randn(x.shape[0], 1376), []
            else:
                return torch.randn(x.shape[0], 1536), []
        self.mock_backbone.generate_embeddings = mock_generate_embeddings_multi
        
        model = ArkClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=True,
            num_views=4,
            view_fusion_type="mlp_adapter",
            adapter_dim=512,
            view_fusion_hidden_dim=512
        )
        
        x = torch.randn(self.batch_size, 4, 3, self.img_size, self.img_size)
        output, attention_maps = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        
    def test_ark_classifier_without_backbone_projector(self):
        """Test ArkClassifier without using backbone projector."""
        model = ArkClassifier(
            backbone=self.mock_backbone,
            num_classes=self.num_classes,
            multi_view=False,
            use_backbone_projector=False
        )
        
        self.assertEqual(model.embed_dim, 1536)  # Should use encoder_features
        
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        output, attention_maps = model(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))


class TestArkFeatureMapCapture(unittest.TestCase):
    """Test the stage-wise feature map capture for Swin/Ark."""

    def setUp(self):
        self.num_classes_list = [2]
        self.img_size = 224
        self.batch_size = 1

    def test_extract_stage_feature_maps_returns_dict(self):
        """Feature map capture should return a dict with one entry per stage."""
        backbone = SwinTransformer(
            num_classes_list=self.num_classes_list,
            img_size=self.img_size,
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
        )
        model = ArkClassifier(
            backbone=backbone,
            num_classes=2,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        result = model.extract_stage_feature_maps(x)

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 4)
        for stage_idx in range(4):
            self.assertIn(stage_idx, result)
            info = result[stage_idx]
            H, W = info["spatial_size"]
            N = info["features"].shape[1]
            self.assertEqual(N, H * W,
                             f"Stage {stage_idx}: feature count {N} != spatial {H}x{W}")

    def test_hooks_are_cleaned_up(self):
        """Hooks should be removed after extract_stage_feature_maps."""
        backbone = SwinTransformer(
            num_classes_list=self.num_classes_list,
            img_size=self.img_size,
            patch_size=4,
            window_size=7,
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
        )
        model = ArkClassifier(
            backbone=backbone,
            num_classes=2,
            multi_view=False,
        )
        x = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        model.extract_stage_feature_maps(x)
        self.assertEqual(len(backbone._feature_hook_handles), 0)


class TestLoadArkModel(unittest.TestCase):
    """Test the load_prtrained_ark_model function."""
    
    @patch('torch.load')
    def test_load_ark_model(self, mock_load):
        """Test loading Ark model from checkpoint."""
        # Mock checkpoint data
        mock_checkpoint = {
            "teacher": {
                "backbone.patch_embed.proj.weight": torch.randn(192, 3, 4, 4),
                "backbone.layers.0.blocks.0.attn.qkv.weight": torch.randn(576, 192),
                "classifier.weight": torch.randn(2, 1536)
            }
        }
        mock_load.return_value = mock_checkpoint
        
        with patch('rad_dino.models.ark.SwinTransformer') as mock_swin:
            mock_model = Mock()
            mock_model.load_state_dict.return_value = Mock()
            mock_model.patch_embed.proj.weight.shape = (192, 3, 4, 4)  # Mock the shape attribute
            mock_swin.return_value = mock_model
            
            model = load_prtrained_ark_model(
                checkpoint_path="/fake/path/checkpoint.pth.tar",
                num_classes_list=[2, 3],
                img_size=768,
                patch_size=4,
                window_size=12,
                embed_dim=192,
                depths=(2, 2, 18, 2),
                num_heads=(6, 12, 24, 48),
                projector_features=1376,
                use_mlp=False,
                device="cpu"
            )
            
            self.assertIsNotNone(model)
            mock_model.load_state_dict.assert_called_once()


if __name__ == "__main__":
    unittest.main() 