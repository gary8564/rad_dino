import os
import onnxruntime
import torch
from transformers import AutoModel
from typing import Tuple, Any
from rad_dino.models.dino import DinoClassifier
from rad_dino.models.siglip import MedSigClassifier
from rad_dino.models.ark import ArkClassifier, SwinTransformer
from rad_dino.configs.config import ModelWrapper
from rad_dino.utils.extract_onnx_attentions import augment_onnx_add_attention_outputs
from accelerate import Accelerator
import logging

logger = logging.getLogger(__name__)

def _load_best_dino_model(checkpoint_dir: str, 
                     backbone: AutoModel, 
                     num_classes: int, 
                     accelerator: Accelerator, 
                     multi_view: bool = False, 
                     return_attentions: bool = False) -> DinoClassifier:
    """Load PyTorch model from checkpoint"""
    # Load multi-view configuration from checkpoint
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    
    # Get multi-view parameters from checkpoint or use defaults
    num_views = ckpt.get("num_views")
    view_fusion_type = ckpt.get("view_fusion_type")
    adapter_dim = ckpt.get("adapter_dim")
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim")
    
    model = DinoClassifier(backbone, 
                           num_classes=num_classes, 
                           multi_view=multi_view,
                           num_views=num_views,
                           view_fusion_type=view_fusion_type,
                           adapter_dim=adapter_dim,
                           view_fusion_hidden_dim=view_fusion_hidden_dim,
                           return_attentions=return_attentions)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model


def _load_best_ark_model(checkpoint_dir: str, 
                         num_classes: int, 
                         accelerator: Accelerator, 
                         multi_view: bool = False, 
                         return_attention: bool = False) -> ArkClassifier:
    """Load Ark PyTorch model from checkpoint"""
    # Load the trained classifier weights to get configuration
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    
    # Get multi-view parameters from checkpoint or use defaults
    num_views = ckpt.get("num_views")
    view_fusion_type = ckpt.get("view_fusion_type")
    adapter_dim = ckpt.get("adapter_dim")
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim")
    use_backbone_projector = ckpt.get("use_backbone_projector")
    state_dict = ckpt.get("model_state")
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("Invalid checkpoint: 'model_state' is missing or empty")

    backbone = SwinTransformer(
        num_classes_list=[num_classes],
        img_size=768,
        patch_size=4,
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        projector_features=1376,
        use_mlp=False,
        return_attention=return_attention,
        grad_checkpointing=False,
    )
    
    # If use_backbone_projector flag is accidentally not saved in checkpoint, 
    # infer it from saved classifier weight shape
    if use_backbone_projector is None:
        classifier_in_features = state_dict["classifier.weight"].shape[1]
        # Projector off -> classifier expects backbone.projector.in_features (if projector exists)
        encoder_dim = backbone.projector.in_features
        projector_dim = backbone.num_features
        if classifier_in_features == projector_dim:
            use_backbone_projector = True
        elif classifier_in_features == encoder_dim:
            use_backbone_projector = False
        else:
            raise ValueError(
                "Could not infer 'use_backbone_projector' from checkpoint. "
                f"classifier_in_features={classifier_in_features}, "
                f"encoder_dim={encoder_dim}, projector_dim={projector_dim}."
            )
    
    # Create Ark classifier
    model = ArkClassifier(backbone, 
                        num_classes=num_classes, 
                        multi_view=multi_view,
                        num_views=num_views,
                        adapter_dim=adapter_dim,
                        view_fusion_type=view_fusion_type,
                        view_fusion_hidden_dim=view_fusion_hidden_dim,
                        use_backbone_projector=use_backbone_projector)

    # Identify and delete unnecessary keys to avoid loading Ark omni heads
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("backbone.omni_heads.")]
    if keys_to_drop:
        logger.info(f"Dropping {len(keys_to_drop)} Ark omni heads from checkpoint")
        for k in keys_to_drop:
            state_dict.pop(k, None)

    # Load remaining weights non-strictly to ignore any benign missing/unexpected keys
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Ark checkpoint loaded with msg: {msg}")
    model = model.to(accelerator.device)
    return model


def _load_best_medsig_model(checkpoint_dir: str, 
                            backbone: AutoModel, 
                            num_classes: int, 
                            accelerator: Accelerator, 
                            multi_view: bool = False, 
                            return_attentions: bool = False) -> MedSigClassifier:
    """Load MedSig PyTorch model from checkpoint"""
    # Load multi-view configuration from checkpoint
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    
    # Get multi-view parameters from checkpoint or use defaults
    num_views = ckpt.get("num_views")
    view_fusion_type = ckpt.get("view_fusion_type")
    adapter_dim = ckpt.get("adapter_dim")
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim")
    
    model = MedSigClassifier(backbone, 
                           num_classes=num_classes, 
                           multi_view=multi_view,
                           num_views=num_views,
                           view_fusion_type=view_fusion_type,
                           adapter_dim=adapter_dim,
                           view_fusion_hidden_dim=view_fusion_hidden_dim,
                           return_attentions=return_attentions)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model

def _validate_onnx_shape(onnx_input_shape: Tuple, multi_view: bool, backbone_config: Any) -> None:
    """Validate ONNX model input shape."""
    if hasattr(backbone_config, "image_size"):
        image_size = getattr(backbone_config, "image_size")
    elif hasattr(backbone_config, "vision_config") and hasattr(backbone_config.vision_config, "image_size"):
            image_size = getattr(backbone_config.vision_config, "image_size")
    else:
        raise ValueError(f"image_size not found in backbone config: {backbone_config}")


    if multi_view:
        # For multi-view, expect [batch_size, 4, channels, height, width]
        expected_shape = (None, 4, 3, image_size, image_size)
        if len(onnx_input_shape) != 5 or onnx_input_shape[1] != 4:
            raise ValueError(
                f"ONNX model input shape {onnx_input_shape} doesn't match expected multi-view shape {expected_shape}"
            )
    else:
        # For single-view, expect [batch_size, channels, height, width]
        expected_shape = (None, 3, image_size, image_size)
        if len(onnx_input_shape) != 4:
            raise ValueError(
                f"ONNX model input shape {onnx_input_shape} doesn't match expected single-view shape {expected_shape}"
            )

# Load pretrained backbone model
def load_pretrained_model(model_repo):
    return AutoModel.from_pretrained(model_repo)


def _build_backbone(model_name: str, model_repo: str) -> Tuple[Any, Any]:
    """Return (backbone, backbone_config) for a given architecture name.

    For Ark, returns (None, mock_config) since Ark uses its own loader.
    For others (DINO variants, MedSig), returns HF backbone and its config.
    """
    if model_name == "ark":
        mock_cfg = type('Config', (), {'image_size': 768})()
        return None, mock_cfg
    backbone = load_pretrained_model(model_repo)
    return backbone, backbone.config


def _augment_onnx_for_attention(session: onnxruntime.InferenceSession,
                                           onnx_model_path: str,
                                           providers: list,
                                           model_name: str,
                                           show_attention: bool) -> Tuple[onnxruntime.InferenceSession, str, list[str]]:
    """Augment a ONNX model to expose attention outputs. Returns (session, input_name, output_names)."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    if not show_attention:
        return session, input_name, output_names

    # Only DINO variants are supported for attention augmentation for now
    if "dino" not in model_name:
        raise NotImplementedError(
            f"Attention visualization on ONNX is not supported for architecture '{model_name}'."
        )

    if len(output_names) >= 2:
        # Already has attention outputs
        return session, input_name, output_names

    try:
        augmented_path, new_outputs = augment_onnx_add_attention_outputs(onnx_model_path, None)
        if new_outputs:
            logger.info(
                f"Augmented ONNX model to expose attention outputs. Added {len(new_outputs)} outputs."
            )
            session = onnxruntime.InferenceSession(augmented_path, providers=providers)
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]
    except Exception as e:
        raise Exception(f"Extracting attention from ONNX model failed for {model_name}: {e}.")

    return session, input_name, output_names


def _load_onnx_model(checkpoint_dir: str,
                     model_name: str,
                     accelerator: Accelerator,
                     backbone_config: Any,
                     show_attention: bool,
                     multi_view: bool) -> ModelWrapper:
    """Load ONNX session and return a ModelWrapper. Attempts attention augmentation for DINO only."""
    onnx_model_path = os.path.join(checkpoint_dir, "best.onnx")

    providers = []
    if accelerator.device.type == 'cuda':
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
    session, input_name, output_names = _augment_onnx_for_attention(
        session, onnx_model_path, providers, model_name, show_attention
    )

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


def _load_pt_model(checkpoint_dir: str,
                   model_name: str,
                   backbone: Any,
                   num_classes: int,
                   accelerator: Accelerator,
                   multi_view: bool,
                   backbone_config: Any,
                   show_attention: bool) -> ModelWrapper:
    """Load a PyTorch classifier and return a ModelWrapper."""
    if model_name == "ark":
        model = _load_best_ark_model(checkpoint_dir, num_classes, accelerator, multi_view, return_attention=show_attention)
    elif model_name == "medsiglip":
        model = _load_best_medsig_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, return_attentions=show_attention)
    else:  # dino models
        model = _load_best_dino_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, return_attentions=show_attention)

    return ModelWrapper(
        model_type='pytorch',
        model=model,
        config=backbone_config,
        device=accelerator.device,
        multi_view=multi_view
    )

# Load the classifier model
def load_model(checkpoint_dir: str, 
               model_name: str,
               model_repo: str, 
               num_classes: int, 
               accelerator: Accelerator, 
               show_gradcam: bool, 
               show_attention: bool,
               show_lrp: bool, 
               multi_view: bool = False) -> ModelWrapper:
    """Load model (ONNX or PyTorch) based on availability and requirements."""
    
    # Handle different architectures
    backbone, backbone_config = _build_backbone(model_name, model_repo)
    
    onnx_model_path = os.path.join(checkpoint_dir, "best.onnx")
    
    # Use ONNX if available and no gradient-based visualization approaches are required
    use_onnx_preferred = os.path.exists(onnx_model_path) and not (show_gradcam or show_lrp)
    
    # For MedSigLIP attention visualization we must use PyTorch to capture pooler weights
    if model_name == "medsiglip" and show_attention:
        use_onnx_preferred = False
        
    if use_onnx_preferred:
        return _load_onnx_model(checkpoint_dir, model_name, accelerator, backbone_config, show_attention, multi_view)
    else:
        # Use PyTorch model
        return _load_pt_model(
            checkpoint_dir, model_name, backbone, num_classes, accelerator, multi_view, backbone_config, show_attention
        )