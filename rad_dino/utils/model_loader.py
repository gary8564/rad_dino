import os
import onnxruntime
import torch
from transformers import AutoModel
from typing import Tuple, Any
from rad_dino.models.dino import DinoClassifier
from rad_dino.models.siglip import MedSigClassifier
from rad_dino.models.ark import ArkClassifier, load_prtrained_ark_model
from rad_dino.configs.config import ModelWrapper
from accelerate import Accelerator
import logging

logger = logging.getLogger(__name__)

def _load_best_dino_model(checkpoint_dir: str, 
                     backbone: AutoModel, 
                     num_classes: int, 
                     accelerator: Accelerator, 
                     multi_view: bool = False, 
                     view_fusion_type: str = "mean") -> DinoClassifier:
    """Load PyTorch model from checkpoint"""
    # Load multi-view configuration from checkpoint
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    
    # Get multi-view parameters from checkpoint or use defaults
    num_views = ckpt.get("num_views", 4) if multi_view else None
    view_fusion_type = ckpt.get("view_fusion_type", view_fusion_type) if multi_view else None
    adapter_dim = ckpt.get("adapter_dim", None) if multi_view else None
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim", None) if multi_view else None
    
    model = DinoClassifier(backbone, 
                           num_classes=num_classes, 
                           multi_view=multi_view,
                           num_views=num_views,
                           view_fusion_type=view_fusion_type,
                           adapter_dim=adapter_dim,
                           view_fusion_hidden_dim=view_fusion_hidden_dim)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model


def _load_best_ark_model(checkpoint_dir: str, 
                         num_classes: int, 
                         accelerator: Accelerator, 
                         multi_view: bool = False, 
                         view_fusion_type: str = "mean") -> ArkClassifier:
    """Load Ark PyTorch model from checkpoint"""
    # Load the trained classifier weights to get configuration
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    
    # Get multi-view parameters from checkpoint or use defaults
    num_views = ckpt.get("num_views", 4) if multi_view else None
    view_fusion_type = ckpt.get("view_fusion_type", view_fusion_type) if multi_view else None
    adapter_dim = ckpt.get("adapter_dim", None) if multi_view else None
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim", None) if multi_view else None
    use_backbone_projector = ckpt.get("use_backbone_projector", True) 
    
    backbone = load_prtrained_ark_model(
        checkpoint_path=ckpt,
        num_classes_list=[num_classes],  # Ark expects a list of class counts
        img_size=768,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        projector_features=1376,
        use_mlp=False,
        device=accelerator.device
    )
    
    # Create Ark classifier
    model = ArkClassifier(backbone, 
                        num_classes=num_classes, 
                        multi_view=multi_view,
                        num_views=num_views,
                        view_fusion_type=view_fusion_type,
                        adapter_dim=adapter_dim,
                        view_fusion_hidden_dim=view_fusion_hidden_dim,
                        use_backbone_projector=use_backbone_projector)
    
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model


def _load_best_medsig_model(checkpoint_dir: str, 
                            backbone: AutoModel, 
                            num_classes: int, 
                            accelerator: Accelerator, 
                            multi_view: bool = False, 
                            view_fusion_type: str = "mean") -> MedSigClassifier:
    """Load MedSig PyTorch model from checkpoint"""
    # Load multi-view configuration from checkpoint
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    
    # Get multi-view parameters from checkpoint or use defaults
    num_views = ckpt.get("num_views", 4) if multi_view else None
    view_fusion_type = ckpt.get("view_fusion_type", view_fusion_type) if multi_view else None
    adapter_dim = ckpt.get("adapter_dim", None) if multi_view else None
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim", None) if multi_view else None
    
    model = MedSigClassifier(backbone, 
                           num_classes=num_classes, 
                           multi_view=multi_view,
                           num_views=num_views,
                           view_fusion_type=view_fusion_type,
                           adapter_dim=adapter_dim,
                           view_fusion_hidden_dim=view_fusion_hidden_dim)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model

def _validate_onnx_shape(onnx_input_shape: Tuple, multi_view: bool, backbone_config: Any) -> None:
    """Validate ONNX model input shape"""
    if multi_view:
        # For multi-view, expect [batch_size, 4, channels, height, width]
        expected_shape = (None, 4, 3, backbone_config.image_size, backbone_config.image_size)
        if len(onnx_input_shape) != 5 or onnx_input_shape[1] != 4:
            raise ValueError(f"ONNX model input shape {onnx_input_shape} doesn't match expected multi-view shape {expected_shape}")
    else:
        # For single-view, expect [batch_size, channels, height, width]
        expected_shape = (None, 3, backbone_config.image_size, backbone_config.image_size)
        if len(onnx_input_shape) != 4:
            raise ValueError(f"ONNX model input shape {onnx_input_shape} doesn't match expected single-view shape {expected_shape}")

# Load pretrained backbone model
def load_pretrained_model(model_repo):
    return AutoModel.from_pretrained(model_repo)

# Load the classifier model
def load_model(checkpoint_dir: str, 
               model_repo: str, 
               num_classes: int, 
               accelerator: Accelerator, 
               show_gradcam: bool, 
               show_lrp: bool, 
               multi_view: bool = False, 
               view_fusion_type: str = "mean",
               model_type: str = "dino") -> ModelWrapper:
    """Load model (ONNX or PyTorch) based on availability and requirements"""
    
    # Handle different model types
    if model_type == "ark":
        backbone = None  # Ark models don't use HuggingFace backbones
        backbone_config = type('Config', (), {'image_size': 768})()  # Mock config for Ark
    elif model_type == "medsiglip" or model_type == "dino":
        backbone = load_pretrained_model(model_repo)
        backbone_config = backbone.config
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    onnx_model_path = os.path.join(checkpoint_dir, "best.onnx")
    
    # Use ONNX if available and no gradient-based visualization approaches are required
    if os.path.exists(onnx_model_path) and not show_gradcam and not show_lrp:
        # Configure providers for GPU inference
        providers = []
        if accelerator.device.type == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')  # Fallback to CPU
        
        session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        input_name = session.get_inputs()[0].name            
        output_names = [o.name for o in session.get_outputs()]
        
        # Log ONNX model info
        current_providers = session.get_providers()
        logger.info(f"Successfully loaded ONNX model from {onnx_model_path}")
        logger.info(f"ONNX providers: {current_providers}")
        logger.info(f"Input name: {input_name}")
        logger.info(f"Output names: {output_names}")
        
        # Validate ONNX model input shape
        onnx_input_shape = session.get_inputs()[0].shape
        logger.info(f"ONNX model input shape: {onnx_input_shape}")
        _validate_onnx_shape(onnx_input_shape, multi_view, backbone_config)
        
        return ModelWrapper(
            model_type='onnx',
            session=session,
            input_name=input_name,
            output_names=output_names,
            config=backbone_config,
            device=accelerator.device,
            multi_view=multi_view
        )
    else:
        # Use PyTorch model
        if show_gradcam:
            logger.info("GradCAM requested - using PyTorch model for visualization support")
        elif show_lrp:
            logger.info("LRP requested - using PyTorch model for visualization support")
        else:
            logger.warning(f"ONNX model not found at {onnx_model_path}. Fallback to PyTorch state_dict instead...")
        
        # Load appropriate model based on model_type
        if model_type == "ark":
            model = _load_best_ark_model(checkpoint_dir, num_classes, accelerator, multi_view, view_fusion_type)
        elif model_type == "medsiglip":
            model = _load_best_medsig_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, view_fusion_type)
        else: # dino models
            model = _load_best_dino_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, view_fusion_type)
        
        return ModelWrapper(
            model_type='pytorch',
            model=model,
            config=backbone_config,
            device=accelerator.device,
            multi_view=multi_view
        ) 