import unittest
from unittest.mock import Mock, MagicMock, patch
import torch
import torch.nn as nn
from rad_dino.train.model_registry import (
    ModelRegistry, UnfreezeModelHandler, UnfreezeArkHandler, 
    UnfreezeMedSigLIPHandler, UnfreezeViTHandler,
    get_model_registry, register_unfreeze_handler, 
    get_model_info, get_layer_term
)


class TestUnfreezeModelHandler(unittest.TestCase):
    """Test the abstract base class for unfreeze handlers."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Create a concrete implementation for testing
        class TestHandler(UnfreezeModelHandler):
            def get_model_info(self, model):
                return {'model_type': 'Test', 'total_layers': 4, 'layer_pattern': 'layer.{}'}
            
            def get_layer_term(self):
                return 'test layers'
        
        handler = TestHandler()
        model = Mock()
        
        info = handler.get_model_info(model)
        self.assertEqual(info['model_type'], 'Test')
        self.assertEqual(info['total_layers'], 4)
        self.assertEqual(info['layer_pattern'], 'layer.{}')
        
        term = handler.get_layer_term()
        self.assertEqual(term, 'test layers')


class TestUnfreezeArkHandler(unittest.TestCase):
    """Test the Ark unfreeze handler."""
    
    def setUp(self):
        self.handler = UnfreezeArkHandler()
        self.model = Mock()
        
        # Mock backbone with layers
        self.model.backbone = Mock()
        self.model.backbone.layers = [Mock() for _ in range(4)]
    
    def test_get_model_info(self):
        """Test getting model info for Ark models."""
        info = self.handler.get_model_info(self.model)
        
        self.assertEqual(info['model_type'], 'SwinT')
        self.assertEqual(info['total_layers'], 4)
        self.assertEqual(info['layer_pattern'], 'layers.{}')
    
    def test_get_layer_term(self):
        """Test getting layer term for Ark models."""
        term = self.handler.get_layer_term()
        self.assertEqual(term, 'stages')


class TestUnfreezeMedSigLIPHandler(unittest.TestCase):
    """Test the MedSigLIP unfreeze handler."""
    
    def setUp(self):
        self.handler = UnfreezeMedSigLIPHandler()
        self.model = Mock()
        
        # Mock backbone config
        self.model.backbone = Mock()
        self.model.backbone.config = Mock()
        self.model.backbone.config.vision_config = Mock()
        self.model.backbone.config.vision_config.num_hidden_layers = 12
    
    def test_get_model_info(self):
        """Test getting model info for MedSigLIP models."""
        info = self.handler.get_model_info(self.model)
        
        self.assertEqual(info['model_type'], 'MedSigLIP')
        self.assertEqual(info['total_layers'], 12)
        self.assertEqual(info['layer_pattern'], 'vision_model.encoder.layers.{}')
    
    def test_get_layer_term(self):
        """Test getting layer term for MedSigLIP models."""
        term = self.handler.get_layer_term()
        self.assertEqual(term, 'vision encoder layers')


class TestUnfreezeViTHandler(unittest.TestCase):
    """Test the ViT unfreeze handler."""
    
    def setUp(self):
        self.handler = UnfreezeViTHandler()
        self.model = Mock()
        
        # Mock backbone config
        self.model.backbone = Mock()
        self.model.backbone.config = Mock()
        self.model.backbone.config.num_hidden_layers = 12
    
    def test_get_model_info(self):
        """Test getting model info for ViT models."""
        info = self.handler.get_model_info(self.model)
        
        self.assertEqual(info['model_type'], 'ViT')
        self.assertEqual(info['total_layers'], 12)
        self.assertEqual(info['layer_pattern'], 'layer.{}')
    
    def test_get_layer_term(self):
        """Test getting layer term for ViT models."""
        term = self.handler.get_layer_term()
        self.assertEqual(term, 'layers')


class TestModelRegistry(unittest.TestCase):
    """Test the model registry functionality."""
    
    def setUp(self):
        self.registry = ModelRegistry()
        self.model = Mock()
    
    def test_default_registration(self):
        """Test that default handlers are registered."""
        # Check that default handlers are registered
        ark_handler = self.registry.get_unfreeze_handler('ark')
        medsig_handler = self.registry.get_unfreeze_handler('medsiglip')
        dinov2_handler = self.registry.get_unfreeze_handler('dinov2-base')
        
        self.assertIsInstance(ark_handler, UnfreezeArkHandler)
        self.assertIsInstance(medsig_handler, UnfreezeMedSigLIPHandler)
        self.assertIsInstance(dinov2_handler, UnfreezeViTHandler)
    
    def test_custom_handler_registration(self):
        """Test registering custom handlers."""
        class CustomHandler(UnfreezeModelHandler):
            def get_model_info(self, model):
                return {'model_type': 'Custom', 'total_layers': 8, 'layer_pattern': 'custom.{}'}
            
            def get_layer_term(self):
                return 'custom layers'
        
        handler = CustomHandler()
        self.registry.register_unfreeze_handler('custom_model', handler)
        
        retrieved_handler = self.registry.get_unfreeze_handler('custom_model')
        self.assertIsInstance(retrieved_handler, CustomHandler)
        
        info = self.registry.get_model_info(self.model, 'custom_model')
        self.assertEqual(info['model_type'], 'Custom')
        self.assertEqual(info['total_layers'], 8)
        
        term = self.registry.get_layer_term(self.model, 'custom_model')
        self.assertEqual(term, 'custom layers')
    
    def test_get_model_info_with_registered_handler(self):
        """Test getting model info with registered handler."""
        # Mock the model for Ark handler
        self.model.backbone = Mock()
        self.model.backbone.layers = [Mock() for _ in range(4)]
        
        info = self.registry.get_model_info(self.model, 'ark')
        self.assertEqual(info['model_type'], 'SwinT')
        self.assertEqual(info['total_layers'], 4)
        self.assertEqual(info['layer_pattern'], 'layers.{}')
    
    def test_get_model_info_with_unregistered_handler(self):
        """Test getting model info with unregistered handler."""
        with self.assertRaises(ValueError):
            self.registry.get_model_info(self.model, 'unknown_model')
    
    def test_get_layer_term_with_unregistered_handler(self):
        """Test getting layer term with unregistered handler."""
        term = self.registry.get_layer_term(self.model, 'unknown_model')
        self.assertEqual(term, 'layers')  # Default fallback


class TestGlobalRegistryFunctions(unittest.TestCase):
    """Test the global registry functions."""
    
    def setUp(self):
        self.model = Mock()
        self.model.backbone = Mock()
        self.model.backbone.layers = [Mock() for _ in range(4)]
    
    def test_get_model_registry(self):
        """Test getting the global registry."""
        registry = get_model_registry()
        self.assertIsInstance(registry, ModelRegistry)
    
    def test_register_unfreeze_handler(self):
        """Test registering handlers with global registry."""
        class TestHandler(UnfreezeModelHandler):
            def get_model_info(self, model):
                return {'model_type': 'Test', 'total_layers': 6, 'layer_pattern': 'test.{}'}
            
            def get_layer_term(self):
                return 'test layers'
        
        handler = TestHandler()
        register_unfreeze_handler('test_model', handler)
        
        # Verify registration
        registry = get_model_registry()
        retrieved_handler = registry.get_unfreeze_handler('test_model')
        self.assertIsInstance(retrieved_handler, TestHandler)
    
    def test_get_model_info_global(self):
        """Test getting model info using global function."""
        info = get_model_info(self.model, 'ark')
        self.assertEqual(info['model_type'], 'SwinT')
        self.assertEqual(info['total_layers'], 4)
    
    def test_get_layer_term_global(self):
        """Test getting layer term using global function."""
        term = get_layer_term(self.model, 'ark')
        self.assertEqual(term, 'stages')


class TestModelRegistryIntegration(unittest.TestCase):
    """Test integration with actual model structures."""
    
    def test_ark_model_integration(self):
        """Test Ark model integration with registry."""
        # Create a mock Ark model structure
        class MockArkModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                # Mock layers for Ark handler
                self.backbone.layers = [nn.Linear(64, 32) for _ in range(4)]
                self.classifier = nn.Linear(32, 2)
        
        model = MockArkModel()
        info = get_model_info(model, 'ark')
        
        self.assertEqual(info['model_type'], 'SwinT')
        self.assertEqual(info['total_layers'], 4)
        self.assertEqual(info['layer_pattern'], 'layers.{}')
        
        term = get_layer_term(model, 'ark')
        self.assertEqual(term, 'stages')
    
    def test_vit_model_integration(self):
        """Test ViT model integration with registry."""
        # Create a mock ViT model structure
        class MockViTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                # Mock config for ViT handler
                self.backbone.config = Mock()
                self.backbone.config.num_hidden_layers = 12
                self.classifier = nn.Linear(64, 2)
        
        model = MockViTModel()
        info = get_model_info(model, 'dinov2-base')
        
        self.assertEqual(info['model_type'], 'ViT')
        self.assertEqual(info['total_layers'], 12)
        self.assertEqual(info['layer_pattern'], 'layer.{}')
        
        term = get_layer_term(model, 'dinov2-base')
        self.assertEqual(term, 'layers')
    
    def test_medsiglip_model_integration(self):
        """Test MedSigLIP model integration with registry."""
        # Create a mock MedSigLIP model structure
        class MockMedSigModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                # Mock config for MedSigLIP handler
                self.backbone.config = Mock()
                self.backbone.config.vision_config = Mock()
                self.backbone.config.vision_config.num_hidden_layers = 12
                self.classifier = nn.Linear(64, 2)
        
        model = MockMedSigModel()
        info = get_model_info(model, 'medsiglip')
        
        self.assertEqual(info['model_type'], 'MedSigLIP')
        self.assertEqual(info['total_layers'], 12)
        self.assertEqual(info['layer_pattern'], 'vision_model.encoder.layers.{}')
        
        term = get_layer_term(model, 'medsiglip')
        self.assertEqual(term, 'vision encoder layers')


if __name__ == "__main__":
    unittest.main() 