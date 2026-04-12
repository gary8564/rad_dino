import traceback
import torch
from typing import Optional, List, Union
from accelerate import Accelerator
from rad_dino.utils.visualization.visualize_vit_attention import (
    visualize_attention_maps as visualize_vit_attention_maps,
    compute_vit_gradient_rollout,
    compute_biomedclip_gradient_rollout,
)
from rad_dino.utils.visualization.visualize_siglip_attention import (
    visualize_siglip_attention_maps,
    compute_siglip_gradient_rollout,
)
from rad_dino.utils.visualization.visualize_feature_maps import visualize_stage_feature_maps
from rad_dino.utils.visualization.visualize_gradcam import visualize_gradcam
from rad_dino.models.base import BaseClassifier
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PATCH_SIZE = 14
DEFAULT_ATTENTION_HEADS = 'mean'
STANDARD_MAMMO_VIEW_NAMES = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']

def _parse_save_heads(save_heads_arg: Optional[str] = None) -> Union[str, int]:
    """Parse the save_heads argument for attention visualization"""
    if save_heads_arg is None:
        return DEFAULT_ATTENTION_HEADS
    
    if save_heads_arg in ['mean', 'max', 'min']:
        logger.info(f"Using attention head fusion: {save_heads_arg}")
        return save_heads_arg
    if save_heads_arg.isdigit():
        head_fusion = int(save_heads_arg)
        if head_fusion <= 0:
            raise ValueError("Integer save_heads must be > 0")
        logger.info(f"Using per-head visualization for {head_fusion} random heads")
        return head_fusion
    else:
        raise ValueError(f"Invalid save_heads argument: {save_heads_arg}. Must be 'mean', 'max', 'min', or a positive integer.")

def _resolve_gradcam_target(model: BaseClassifier):
    """
    Resolve the GradCAM target layer and CLS token flag based on model architecture.

    Returns (target_layer, has_cls_token) or (None, None) if unsupported.
    """
    # HuggingFace ViT (DINO) (has CLS token)
    if hasattr(model.backbone, 'encoder') and hasattr(model.backbone.encoder, 'layer'):
        return model.backbone.encoder.layer[-1].norm1, True
    # HuggingFace SigLIP vision model (no CLS)
    if hasattr(model.backbone, 'vision_model'):
        vm = model.backbone.vision_model
        if hasattr(vm, 'encoder') and hasattr(vm.encoder, 'layers'):
            return vm.encoder.layers[-1].layer_norm1, False  
    # BiomedCLIP (open_clip timm ViT) (has CLS token)
    if hasattr(model.backbone, 'visual') and hasattr(model.backbone.visual, 'trunk'):
        trunk = model.backbone.visual.trunk
        if hasattr(trunk, 'blocks') and len(trunk.blocks) > 0:
            return trunk.blocks[-1].norm1, True
    # MedImageInsight (DaViT): target the LayerNorm in the last ChannelBlock's FFN
    if hasattr(model.backbone, 'image_encoder') and hasattr(model.backbone.image_encoder, 'blocks'):
        ie = model.backbone.image_encoder
        if len(ie.blocks) > 0:
            last_stage = ie.blocks[-1]
            if (len(last_stage) > 0
                    and hasattr(last_stage[0], 'channel_block')
                    and hasattr(last_stage[0].channel_block, 'ffn')
                    and hasattr(last_stage[0].channel_block.ffn, 'norm')):
                return last_stage[0].channel_block.ffn.norm, False
        return None, None
    # Ark (Swin) — target last stage last block norm1 (no CLS token)
    if hasattr(model.backbone, 'layers'):
        last_stage = model.backbone.layers[-1]
        if hasattr(last_stage, 'blocks') and len(last_stage.blocks) > 0:
            return last_stage.blocks[-1].norm1, False
    return None, None


def _flatten_multiview_batch(images: torch.Tensor, image_ids: List[str]) -> tuple[torch.Tensor, List[str]]:
    """Flatten [B, V, C, H, W] into per-view images and suffixed ids."""
    if images.dim() != 5:
        return images, list(image_ids)

    num_views = images.shape[1]
    view_names = (
        STANDARD_MAMMO_VIEW_NAMES
        if num_views == len(STANDARD_MAMMO_VIEW_NAMES)
        else [f"view_{idx}" for idx in range(num_views)]
    )

    flat_images = images.reshape(images.shape[0] * num_views, *images.shape[2:])
    flat_image_ids = [
        f"{sample_id}_{view_name}"
        for sample_id in image_ids
        for view_name in view_names
    ]
    return flat_images, flat_image_ids


def _run_gradcam_visualization(model: BaseClassifier, 
                               images: torch.Tensor, 
                               image_ids: List[str], 
                               class_labels: List, 
                               output_gradcam: str, 
                               accelerator: Accelerator,
                               image_mean: Optional[torch.Tensor] = None, 
                               image_std: Optional[torch.Tensor] = None) -> None:
    """Run GradCAM visualization for a batch"""
    target_layer, has_cls_token = _resolve_gradcam_target(model)
    if target_layer is None:
        logger.warning(
            "GradCAM target layer could not be resolved for this model architecture. "
            "Skipping GradCAM visualization."
        )
        return

    for sample_idx, image_id in enumerate(image_ids):
        sample_image = images[sample_idx].unsqueeze(0)

        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()

        visualize_gradcam(
            model, sample_image, target_layer,
            image_id, output_gradcam,
            accelerator, image_mean, image_std, class_labels=class_labels,
            has_cls_token=has_cls_token,
        )

        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()

class ExplainableVisualizer:
    """Manages all explainable visualization tasks during inference"""
    
    def __init__(self, accelerator: Accelerator, output_paths, model_wrapper, 
                 image_processor=None, show_attention: bool = False, 
                 show_gradcam: bool = False,
                 show_feature_maps: bool = False,
                 compute_gradient_rollout: bool = False):
        self.accelerator = accelerator
        self.output_paths = output_paths
        self.model_wrapper = model_wrapper
        self.show_attention = show_attention
        self.show_gradcam = show_gradcam
        self.show_feature_maps = show_feature_maps
        self.compute_gradient_rollout = compute_gradient_rollout
        
        # Setup image processor for visualization
        self.image_mean = None
        self.image_std = None
        any_vis = show_attention or show_gradcam or show_feature_maps or compute_gradient_rollout
        if image_processor and any_vis:
            self.image_mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
            self.image_std = torch.tensor(image_processor.image_std).view(3, 1, 1)
        elif any_vis:
            self.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            self.image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def run_gradcam_visualization(self, model: BaseClassifier, images: torch.Tensor, 
                                 image_ids: List[str], class_labels: List) -> None:
        """Run GradCAM visualization if enabled"""
        if not self.show_gradcam:
            return
            
        _run_gradcam_visualization(
            model, images, image_ids, class_labels, 
            self.output_paths.gradcam, self.accelerator, 
            self.image_mean, self.image_std
        )
    
    def run_attention_visualization(self, 
                                    attentions: List[torch.Tensor] | torch.Tensor, 
                                    images: torch.Tensor, 
                                    image_ids: List[str], 
                                    backbone_config, 
                                    attention_threshold: float, 
                                    save_heads: str, 
                                    compute_rollout: bool = False,
                                    pooler_attn_weights: Optional[torch.Tensor] = None) -> None:
        """Run attention visualization if enabled"""
        if not self.show_attention or attentions is None:
            return
            
        save_heads_param = _parse_save_heads(save_heads)

        # Route between DINO-like (ViT with CLS) and SigLIP (ViT w/o CLS, MAP pooling)
        is_siglip = hasattr(backbone_config, "vision_config") and (
            "siglip" in (getattr(backbone_config, "model_type", "") or "").lower()
            or "siglip" in (getattr(getattr(backbone_config, "vision_config", object()), "model_type", "") or "").lower()
        )
        # Determine patch size
        patch_size = None
        if hasattr(backbone_config, 'patch_size'):
            patch_size = getattr(backbone_config, 'patch_size')
        elif hasattr(backbone_config, 'vision_config') and hasattr(backbone_config.vision_config, 'patch_size'):
            patch_size = getattr(backbone_config.vision_config, 'patch_size')
        else:
            patch_size = DEFAULT_PATCH_SIZE

        if is_siglip:
            if compute_rollout:
                if not isinstance(save_heads_param, str) or save_heads_param not in ("mean", "max", "min"):
                    raise ValueError("For SigLIP rollout, save_heads must be one of {'mean','max','min'}.")
                if pooler_attn_weights is None:
                    raise ValueError("For SigLIP rollout, pooler_attn_weights must be available (PyTorch only).")
            visualize_siglip_attention_maps(
                attentions, images, image_ids, self.output_paths.attention,
                self.accelerator, self.image_mean, self.image_std,
                patch_size=patch_size, threshold=attention_threshold,
                head_fusion=save_heads_param,
                compute_rollout=compute_rollout, rollout_discard_ratio=0.9,
                pooler_attn_weights=pooler_attn_weights,
                rollout_last_n_layers=6)
        else:
            visualize_vit_attention_maps(
                attentions, images, image_ids, self.output_paths.attention,
                self.accelerator, self.image_mean, self.image_std,
                patch_size=patch_size, threshold=attention_threshold,
                head_fusion=save_heads_param,
                compute_rollout=compute_rollout, rollout_discard_ratio=0.9)
    
    def run_feature_map_visualization(self, model, images: torch.Tensor,
                                      image_ids: List[str]) -> None:
        """
        Run stage-wise feature map visualization for models that support it
        (DaViT / MedImageInsight and Swin / Ark).
        """
        if not self.show_feature_maps:
            return

        from rad_dino.models.medimageinsight import MedImageInsightClassifier
        from rad_dino.models.ark import ArkClassifier

        stage_features = None
        vis_images = images
        vis_image_ids = image_ids

        if isinstance(model, (MedImageInsightClassifier, ArkClassifier)):
            vis_images, vis_image_ids = _flatten_multiview_batch(images, image_ids)
            stage_features = model.extract_stage_feature_maps(vis_images)
            if stage_features is None:
                logger.warning(
                    "No stage features were captured for %s — skipping feature map visualization.",
                    type(model).__name__,
                )
                return
        else:
            logger.info(
                "Feature map visualization is only supported for DaViT (MedImageInsight) "
                "and Swin (Ark). Skipping for %s.", type(model).__name__
            )
            return

        image_mean = self.image_mean
        image_std = self.image_std
        if image_mean is None:
            image_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        visualize_stage_feature_maps(
            stage_features=stage_features,
            images=vis_images,
            image_ids=vis_image_ids,
            output_dir=self.output_paths.feature_maps,
            image_mean=image_mean,
            image_std=image_std,
        )
    
    def run_gradient_rollout_visualization(self, model: BaseClassifier,
                                            images: torch.Tensor,
                                            image_ids: List[str],
                                            backbone_config) -> None:
        """
        Run Gradient Attention Rollout for class-specific explainability.

        Dispatches to the appropriate architecture-specific implementation:
        - DINO/ViT: CLS-based gradient rollout
        - BiomedCLIP: CLS-based gradient rollout via open_clip hooks
        - SigLIP/MedSigLIP: MAP-pooler + encoder gradient rollout
        - Ark/Swin: Not supported (windowed hierarchical attention is too complex)
        """
        if not self.compute_gradient_rollout:
            return

        from rad_dino.models.dino import DinoClassifier
        from rad_dino.models.biomedclip import BiomedCLIPClassifier
        from rad_dino.models.siglip import MedSigClassifier

        patch_size = DEFAULT_PATCH_SIZE
        num_prefix = 0
        output_dir = self.output_paths.attention

        def _run_dino(img, img_id):
            return compute_vit_gradient_rollout(
                model, img, image_mean=self.image_mean, image_std=self.image_std,
                category_index=None, discard_ratio=0.9,
                num_prefix_tokens=num_prefix, patch_size=patch_size,
                output_dir=output_dir, image_id=img_id,
            )

        def _run_biomedclip(img, img_id):
            return compute_biomedclip_gradient_rollout(
                model, img, image_mean=self.image_mean, image_std=self.image_std,
                category_index=None, discard_ratio=0.9,
                num_prefix_tokens=num_prefix, patch_size=patch_size,
                output_dir=output_dir, image_id=img_id,
            )

        def _run_siglip(img, img_id):
            return compute_siglip_gradient_rollout(
                model, img, image_mean=self.image_mean, image_std=self.image_std,
                category_index=None, discard_ratio=0.9,
                patch_size=patch_size,
                output_dir=output_dir, image_id=img_id,
            )

        if isinstance(model, DinoClassifier):
            patch_size = getattr(backbone_config, 'patch_size', DEFAULT_PATCH_SIZE)
            num_prefix = 1
            if hasattr(backbone_config, 'num_register_tokens'):
                num_prefix += backbone_config.num_register_tokens
            grad_fn = _run_dino
        elif isinstance(model, BiomedCLIPClassifier):
            patch_size = getattr(backbone_config, 'patch_size', 16)
            if hasattr(backbone_config, 'vision_config'):
                patch_size = getattr(backbone_config.vision_config, 'patch_size', patch_size)
            num_prefix = 1
            grad_fn = _run_biomedclip
        elif isinstance(model, MedSigClassifier):
            if hasattr(backbone_config, 'vision_config'):
                patch_size = getattr(backbone_config.vision_config, 'patch_size', DEFAULT_PATCH_SIZE)
            else:
                patch_size = getattr(backbone_config, 'patch_size', DEFAULT_PATCH_SIZE)
            grad_fn = _run_siglip
        else:
            logger.warning(
                "Gradient rollout is not supported for %s. Skipping.",
                type(model).__name__,
            )
            return

        for sample_idx, image_id in enumerate(image_ids):
            sample_image = images[sample_idx:sample_idx + 1]
            try:
                grad_fn(sample_image, image_id)
            except Exception as e:
                logger.error(
                    f"Gradient rollout failed for {image_id}: {e}\n"
                    f"{traceback.format_exc()}"
                )

            if self.accelerator.device.type == 'cuda':
                torch.cuda.empty_cache()

    def log_visualization_outputs(self) -> None:
        """Log information about saved visualizations"""
        if self.show_attention:
            logger.info(f"Attention visualizations saved to {self.output_paths.attention}")
        if self.show_gradcam:
            logger.info(f"GradCAM visualizations saved to {self.output_paths.gradcam}")
        if self.show_feature_maps:
            logger.info(f"Feature map visualizations saved to {self.output_paths.feature_maps}")
        if self.compute_gradient_rollout:
            logger.info(f"Gradient rollout visualizations saved to {self.output_paths.attention}")