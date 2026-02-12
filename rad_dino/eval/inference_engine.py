import torch
from typing import Optional, Tuple
from accelerate import Accelerator
from rad_dino.models.base import BaseClassifier
import logging

logger = logging.getLogger(__name__)


def _validate_input_shape(images: torch.Tensor, multi_view: bool) -> None:
    """Validate input tensor shape for multi-view vs single-view"""
    if multi_view:
        if len(images.shape) != 5 or images.shape[1] != 4:
            raise ValueError(f"Multi-view inference expects images shape [B, 4, C, H, W], got {images.shape}")
        logger.debug(f"Multi-view batch shape: {images.shape}")
    else:
        if len(images.shape) != 4:
            raise ValueError(f"Single-view inference expects images shape [B, C, H, W], got {images.shape}")
        logger.debug(f"Single-view batch shape: {images.shape}")


def _run_pytorch_inference(model: BaseClassifier, 
                           images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    PyTorch inference workflow.
    Returns (logits, attentions, attention_pooling) where attention_pooling is from SigLIP pooler if available.
    """
    with torch.no_grad():
        logits, attentions = model(images)
    # Try to fetch attention pooling weights if present (SigLIP only)
    attention_pooling = getattr(model, 'last_pooler_attn', None)
    return logits, attentions, attention_pooling


class InferenceEngine:
    def __init__(self, model_wrapper, accelerator: Accelerator, show_attention: bool):
        self.model_wrapper = model_wrapper
        self.accelerator = accelerator
        self.multi_view = model_wrapper.multi_view
        self.show_attention = show_attention
        self.model = model_wrapper.model
        self.backbone_config = model_wrapper.config
        self.model.eval()
        logger.info(f"Running inference with PyTorch model (multi_view={self.multi_view})")

    def run_inference(self, images: torch.Tensor, num_classes: int = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Validate input shapes
        _validate_input_shape(images, self.multi_view)

        # Clear CUDA cache before inference
        if self.accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()

        logits, attentions, raw_pooler_attn = _run_pytorch_inference(self.model, images)

        # Reshape pooler attention to [B*V or B, H, 1, N] -> [B, H, N] or [B, V, H, N]
        pooler_attn = None
        if raw_pooler_attn is not None:
            raw_pooler_attn = raw_pooler_attn.squeeze(2)  # squeeze query dim
            batch_size = images.shape[0]
            if self.multi_view and images.dim() == 5:
                num_views = images.shape[1]
                num_heads = raw_pooler_attn.shape[1]
                num_tokens = raw_pooler_attn.shape[-1]
                pooler_attn = raw_pooler_attn.view(batch_size, num_views, num_heads, num_tokens)
            else:
                pooler_attn = raw_pooler_attn  # [B, H, N]

        # Clear CUDA cache after inference
        if self.accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()

        return logits, attentions, pooler_attn