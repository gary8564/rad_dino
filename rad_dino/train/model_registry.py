from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class UnfreezeModelHandler(ABC):
    """Abstract base class for handling the logic of unfreezing layers in a model."""
    
    @abstractmethod
    def get_model_info(self, model) -> Dict[str, Any]:
        """Get model information for unfreezing.
        
        Returns:
            dict: Contains 'model_type', 'total_layers', 'layer_pattern'
        """
        pass
    
    @abstractmethod
    def get_layer_term(self) -> str:
        """Get model-specific terminology for logging.
        
        Returns:
            str: e.g., 'stages', 'layers', 'vision encoder layers'
        """
        pass

class UnfreezeArkHandler(UnfreezeModelHandler):
    """Handler for Ark (Swin Transformer) models."""
    
    def get_model_info(self, model) -> Dict[str, Any]:
        return {
            'model_type': 'SwinT',
            'total_layers': len(model.backbone.layers),
            'layer_pattern': 'layers.{}'
        }
    
    def get_layer_term(self) -> str:
        return 'stages'

class UnfreezeMedSigLIPHandler(UnfreezeModelHandler):
    """Handler for MedSigLIP models."""
    
    def get_model_info(self, model) -> Dict[str, Any]:
        return {
            'model_type': 'MedSigLIP',
            'total_layers': model.backbone.config.vision_config.num_hidden_layers,
            'layer_pattern': 'vision_model.encoder.layers.{}'
        }
    
    def get_layer_term(self) -> str:
        return 'vision encoder layers'

class UnfreezeMedImageInsightHandler(UnfreezeModelHandler):
    """
    Handler for MedImageInsight DaViT image encoder model.

    The backbone is a UniCLModel whose image encoder is a DaViT with 4 stages accessible at model.backbone.image_encoder.blocks[0..3].
    """

    def get_model_info(self, model) -> Dict[str, Any]:
        return {
            'model_type': 'DaViT-UniCL',
            'total_layers': len(model.backbone.image_encoder.blocks),  # 4 stages
            'layer_pattern': 'image_encoder.blocks.{}'
        }

    def get_layer_term(self) -> str:
        return 'stages'
    

class UnfreezeBiomedCLIPHandler(UnfreezeModelHandler):
    """Handler for BiomedCLIP (open_clip ViT-B/16) models.

    The backbone is an open_clip CLIP model whose visual encoder is a timm
    VisionTransformer with 12 blocks at model.backbone.visual.trunk.blocks.
    """

    def get_model_info(self, model) -> Dict[str, Any]:
        return {
            'model_type': 'BiomedCLIP-ViT',
            'total_layers': len(model.backbone.visual.trunk.blocks),
            'layer_pattern': 'visual.trunk.blocks.{}'
        }

    def get_layer_term(self) -> str:
        return 'vision_encoder_layers'


class UnfreezeViTHandler(UnfreezeModelHandler):
    """Handler for ViT models (DINO, RadDINO, etc.)."""
    
    def get_model_info(self, model) -> Dict[str, Any]:
        return {
            'model_type': 'ViT',
            'total_layers': model.backbone.config.num_hidden_layers,
            'layer_pattern': 'layer.{}'
        }
    
    def get_layer_term(self) -> str:
        return 'layers'

class ModelRegistry:
    """Registry for model unfreezers."""
    
    def __init__(self):
        self._unfreeze_handlers = {}
        self._register_default_unfreezers()
    
    def _register_default_unfreezers(self):
        """Register the default unfreezers."""
        self.register_unfreeze_handler('ark', UnfreezeArkHandler())
        self.register_unfreeze_handler('medimageinsight', UnfreezeMedImageInsightHandler())
        self.register_unfreeze_handler('medsiglip', UnfreezeMedSigLIPHandler())
        self.register_unfreeze_handler('dinov2-base', UnfreezeViTHandler())
        self.register_unfreeze_handler('dinov2-small', UnfreezeViTHandler())
        self.register_unfreeze_handler('dinov2-large', UnfreezeViTHandler())
        self.register_unfreeze_handler('dinov3-small-plus', UnfreezeViTHandler())
        self.register_unfreeze_handler('dinov3-base', UnfreezeViTHandler())
        self.register_unfreeze_handler('dinov3-large', UnfreezeViTHandler())
        self.register_unfreeze_handler('rad-dino', UnfreezeViTHandler())
    
    def register_unfreeze_handler(self, model_type: str, unfreeze_handler: UnfreezeModelHandler):
        """Register a new unfreeze handler for a specific model type.
        
        Args:
            model_type: The model type string (e.g., 'ark', 'medsiglip', 'rad-dino', 'dinov2-base', and other dinov2/dinov3 models)
            unfreeze_handler: The unfreeze handler to register
        """
        self._unfreeze_handlers[model_type] = unfreeze_handler
        logger.info(f"Registered unfreeze handler for {model_type}: {unfreeze_handler.__class__.__name__}")
    
    def get_unfreeze_handler(self, model_type: str) -> Optional[UnfreezeModelHandler]:
        """Get the appropriate unfreeze handler for a model type.
        
        Args:
            model_type: The model type string
            
        Returns:
            UnfreezeModelHandler: The appropriate unfreeze handler, or None if none found
        """
        return self._unfreeze_handlers.get(model_type)
    
    def get_model_info(self, model, model_type: str) -> Dict[str, Any]:
        """Get model information using the appropriate unfreezer.
        
        Args:
            model: The model to get information for
            model_type: The model type string
            
        Returns:
            dict: Model information with type, total_layers, and layer_pattern
        """
        unfreeze_handler = self.get_unfreeze_handler(model_type)
        if unfreeze_handler is None:
            raise ValueError(f"No unfreeze handler found for model type: {model_type}")
        return unfreeze_handler.get_model_info(model)
    
    def get_layer_term(self, model, model_type: str) -> str:
        """Get model-specific layer term.
        
        Args:
            model: The model to get terminology for
            model_type: The model type string
            
        Returns:
            str: Model-specific terminology
        """
        unfreeze_handler = self.get_unfreeze_handler(model_type)
        if unfreeze_handler is None:
            return 'layers'  # Default fallback
        return unfreeze_handler.get_layer_term()

# Global registry instance
_model_registry = ModelRegistry()

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return _model_registry

def register_unfreeze_handler(model_type: str, unfreeze_handler: UnfreezeModelHandler):
    """Register a new unfreeze handler with the global registry.
    
    Args:
        model_type: The model type string (e.g., 'ark', 'medsiglip', 'rad-dino', 'dinov2-base', and other dinov2/dinov3 models)
        unfreeze_handler: The unfreeze handler to register
    """
    _model_registry.register_unfreeze_handler(model_type, unfreeze_handler)

def get_model_info(model, model_type: str) -> Dict[str, Any]:
    """Get model information using the global registry.
    
    Args:
        model: The model to get information for
        model_type: The model type string
        
    Returns:
        dict: Model information with model_type, total_layers, and layer_pattern
    """
    return _model_registry.get_model_info(model, model_type)

def get_layer_term(model, model_type: str) -> str:
    """Get model-specific layer term using the global registry.
    
    Args:
        model: The model to get layer term for
        model_type: The model type string
        
    Returns:
        str: Model-specific layer term
    """
    return _model_registry.get_layer_term(model, model_type) 