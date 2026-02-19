import os
import torch
from transformers import AutoModel
from typing import Tuple, Any
from rad_dino.models.dino import DinoClassifier
from rad_dino.models.siglip import MedSigClassifier
from rad_dino.models.ark import ArkClassifier, SwinTransformer
from rad_dino.models.medimageinsight import MedImageInsightClassifier, load_medimageinsight_model
from rad_dino.models.biomedclip import BiomedCLIPClassifier, load_biomedclip_model
from rad_dino.configs.config import ModelWrapper
from accelerate import Accelerator
import logging

logger = logging.getLogger(__name__)


def _migrate_state_dict_keys(state_dict: dict) -> dict:
    """
    Remap legacy state_dict keys produced by old classifier classes.

    Old DinoClassifier / MedSigClassifier used:
      - ``head.0.weight`` / ``head.0.bias``  (nn.Sequential wrapping nn.Linear)
    New BaseClassifier-derived classes use:
      - ``classifier.weight`` / ``classifier.bias``  (plain nn.Linear)

    Old MedSigClassifier also used ``fusion_layer.*`` instead of ``view_fusion_layer.*``.

    Additionally strips the ``_orig_mod.`` prefix that ``torch.compile()``
     may add to state-dict keys, ensuring checkpoints saved from compiled models can be loaded.
    """
    migrated = {}
    changed = False
    for key, value in state_dict.items():
        new_key = key
        # Strip _orig_mod. prefix from torch.compile wrapper-style state dicts
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
            changed = True
        # head.0.weight -> classifier.weight
        if new_key.startswith("head.0."):
            new_key = new_key.replace("head.0.", "classifier.", 1)
            changed = True
        # fusion_layer.* -> view_fusion_layer.*
        elif new_key.startswith("fusion_layer."):
            new_key = new_key.replace("fusion_layer.", "view_fusion_layer.", 1)
            changed = True
        migrated[new_key] = value
    if changed:
        logger.info("Migrated legacy state_dict keys (head.0.* -> classifier.*, fusion_layer.* -> view_fusion_layer.*, _orig_mod.* -> *)")
    return migrated


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
    num_views = ckpt.get("num_views", 4)
    view_fusion_type = ckpt.get("view_fusion_type", "mean")
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
    state_dict = _migrate_state_dict_keys(ckpt["model_state"])
    model.load_state_dict(state_dict)
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
    state_dict = _migrate_state_dict_keys(ckpt["model_state"])
    model.load_state_dict(state_dict)
    model = model.to(accelerator.device)
    return model

def _load_best_medimageinsight_model(checkpoint_dir: str,
                                     medimageinsight_path: str,
                                     num_classes: int,
                                     accelerator: Accelerator,
                                     multi_view: bool = False,
                                     return_attentions: bool = False) -> MedImageInsightClassifier:
    """Load MedImageInsight PyTorch model from checkpoint.

    Args:
        checkpoint_dir: Directory containing ``best.pt`` checkpoint.
        medimageinsight_path: Path to the cloned lion-ai/MedImageInsights repo.
        num_classes: Number of output classes.
        accelerator: Accelerator instance.
        multi_view: Whether multi-view was used during training.

    Returns:
        MedImageInsightClassifier instance.
    """
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)

    # Get multi-view parameters from checkpoint
    num_views = ckpt.get("num_views")
    view_fusion_type = ckpt.get("view_fusion_type")
    adapter_dim = ckpt.get("adapter_dim")
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim")

    # Rebuild the UniCL backbone from the cloned repo
    backbone = load_medimageinsight_model(medimageinsight_path, device=str(accelerator.device))

    model = MedImageInsightClassifier(
        backbone,
        num_classes=num_classes,
        multi_view=multi_view,
        num_views=num_views,
        view_fusion_type=view_fusion_type,
        adapter_dim=adapter_dim,
        view_fusion_hidden_dim=view_fusion_hidden_dim,
        return_attentions=return_attentions,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(accelerator.device)
    return model


def _load_best_biomedclip_model(checkpoint_dir: str,
                                num_classes: int,
                                accelerator: Accelerator,
                                multi_view: bool = False,
                                return_attentions: bool = False) -> BiomedCLIPClassifier:
    """Load BiomedCLIP PyTorch model from checkpoint.

    Args:
        checkpoint_dir: Directory containing ``best.pt`` checkpoint.
        num_classes: Number of output classes.
        accelerator: Accelerator instance.
        multi_view: Whether multi-view was used during training.
        return_attentions: Whether to enable attention capture.

    Returns:
        Loaded ``BiomedCLIPClassifier``.
    """
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)

    # Get multi-view parameters from checkpoint
    num_views = ckpt.get("num_views")
    view_fusion_type = ckpt.get("view_fusion_type")
    adapter_dim = ckpt.get("adapter_dim")
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim")

    # Rebuild the open_clip backbone from HF hub
    backbone, _ = load_biomedclip_model(device=str(accelerator.device))

    model = BiomedCLIPClassifier(
        backbone,
        num_classes=num_classes,
        multi_view=multi_view,
        num_views=num_views,
        view_fusion_type=view_fusion_type,
        adapter_dim=adapter_dim,
        view_fusion_hidden_dim=view_fusion_hidden_dim,
        return_attentions=return_attentions,
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model = model.to(accelerator.device)
    return model


# Load pretrained backbone model
def load_pretrained_model(model_repo):
    return AutoModel.from_pretrained(model_repo)


def _build_backbone(model_name: str, model_repo: str) -> Tuple[Any, Any]:
    """Return (backbone, backbone_config) for a given architecture name.

    For Ark/MedImageInsight, returns (None, mock_config) since Ark uses its own loader.
    For others (DINO variants, MedSig), returns HF backbone and its config.
    """
    if model_name == "ark":
        mock_cfg = type('Config', (), {'image_size': 768})()
        return None, mock_cfg
    if model_name == "medimageinsight":
        mock_cfg = type('Config', (), {'image_size': 480})()
        return None, mock_cfg
    if model_name == "biomedclip":
        mock_cfg = type('Config', (), {'image_size': 224, 'patch_size': 16})()
        return None, mock_cfg
    backbone = load_pretrained_model(model_repo)
    return backbone, backbone.config


def _load_pt_model(checkpoint_dir: str,
                   model_name: str,
                   backbone: Any,
                   num_classes: int,
                   accelerator: Accelerator,
                   multi_view: bool,
                   backbone_config: Any,
                   show_attention: bool,
                   medimageinsight_path: str = None) -> ModelWrapper:
    """Load a PyTorch classifier and return a ModelWrapper."""
    if model_name == "ark":
        model = _load_best_ark_model(checkpoint_dir, num_classes, accelerator, multi_view, return_attention=show_attention)
    elif model_name == "medimageinsight":
        if medimageinsight_path is None:
            raise ValueError("medimageinsight_path is required to load MedImageInsight checkpoints.")
        model = _load_best_medimageinsight_model(checkpoint_dir, medimageinsight_path, num_classes, accelerator, multi_view, return_attentions=show_attention)
    elif model_name == "biomedclip":
        model = _load_best_biomedclip_model(checkpoint_dir, num_classes, accelerator, multi_view, return_attentions=show_attention)
    elif model_name == "medsiglip":
        model = _load_best_medsig_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, return_attentions=show_attention)
    else:  # dino models
        model = _load_best_dino_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, return_attentions=show_attention)

    return ModelWrapper(
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
               show_attention: bool,
               multi_view: bool = False,
               medimageinsight_path: str = None) -> ModelWrapper:
    """Load model from checkpoint."""
    
    # Handle different architectures
    backbone, backbone_config = _build_backbone(model_name, model_repo)
    
    return _load_pt_model(
        checkpoint_dir, model_name, backbone, num_classes, accelerator,
        multi_view, backbone_config, show_attention,
        medimageinsight_path=medimageinsight_path
    )