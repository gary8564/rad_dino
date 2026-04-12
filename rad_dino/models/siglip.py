"""MedSigLIP classifier with optional attention map extraction via encoder patching."""

import torch
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple
from transformers import AutoModel

from rad_dino.models.base import BaseClassifier
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


# Monkey-patch the encoder attention forward.
def _make_eager_siglip_attn_forward(attn_module, storage=None, retain_grad=False):
    """
    Create an eager attention forward for a HuggingFace SiglipAttention module.

    Args:
        attn_module: The ``self_attn`` module of a SiglipEncoderLayer.
        storage: Optional list to append attention weight tensors to.
        retain_grad: If True, call ``retain_grad()`` so gradients are
            available after backward (required for gradient rollout).
    """
    def eager_forward(hidden_states, attention_mask=None, output_attentions=False, **kwargs):
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = attn_module.q_proj(hidden_states)
        key_states = attn_module.k_proj(hidden_states)
        value_states = attn_module.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, tgt_len, attn_module.num_heads, attn_module.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, tgt_len, attn_module.num_heads, attn_module.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, tgt_len, attn_module.num_heads, attn_module.head_dim
        ).transpose(1, 2)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(-2, -1)
        ) * attn_module.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = F.dropout(
            attn_weights,
            p=0.0 if not attn_module.training else attn_module.dropout,
            training=attn_module.training,
        )

        if retain_grad:
            attn_weights.retain_grad()
        if storage is not None:
            storage.append(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(
            bsz, tgt_len, embed_dim
        )
        attn_output = attn_module.out_proj(attn_output)

        return attn_output, attn_weights

    return eager_forward


def patch_siglip_encoder_eager(
    encoder, storage=None, retain_grad=False,
) -> List[Tuple]:
    """
    Monkey-patch all SigLIP encoder attention layers to eager mode.

    Args:
        encoder: The SigLIP encoder module.
        storage: Optional list to append attention weight tensors to.
        retain_grad: If True, call ``retain_grad()`` so gradients are
            available after backward (required for gradient rollout).

    Returns:
        A list of ``(attn_module, original_forward)`` tuples for
        restoration via `unpatch_siglip_encoder`.
    """
    originals: List[Tuple] = []
    for layer in encoder.layers:
        attn = layer.self_attn
        originals.append((attn, attn.forward))
        attn.forward = _make_eager_siglip_attn_forward(
            attn, storage=storage, retain_grad=retain_grad,
        )
    return originals


def unpatch_siglip_encoder(originals: List[Tuple]) -> None:
    """Restore original attention forward methods."""
    for attn, orig_fwd in originals:
        attn.forward = orig_fwd


class MedSigClassifier(BaseClassifier):
    """
    MedSigLIP classifier.

    Features are extracted from the vision model's pooler output (L2-normalised).
    Supports attention map extraction (including pooler attention capture) and
    gradient checkpointing.
    """

    def __init__(
        self,
        backbone: AutoModel,
        num_classes: int,
        multi_view: bool = False,
        num_views: Optional[int] = None,
        view_fusion_type: Optional[str] = None,
        adapter_dim: Optional[int] = None,
        view_fusion_hidden_dim: Optional[int] = None,
        return_attentions: bool = False,
        gradient_checkpointing: bool = False,
    ):
        embed_dim = backbone.config.vision_config.hidden_size
        super().__init__(
            backbone=backbone,
            embed_dim=embed_dim,
            num_classes=num_classes,
            multi_view=multi_view,
            num_views=num_views,
            view_fusion_type=view_fusion_type,
            adapter_dim=adapter_dim,
            view_fusion_hidden_dim=view_fusion_hidden_dim,
        )
        self.return_attentions = return_attentions
        # Pooler attention weights captured during forward
        self.last_pooler_attn = None

        self.feat_dim = self.embed_dim  # alias for old checkpoints

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()


    def enable_gradient_checkpointing(self):
        try:
            self.backbone.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on MedSigLIP backbone")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    def disable_gradient_checkpointing(self):
        try:
            self.backbone.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled for MedSigLIP model")
        except Exception as e:
            logger.warning(f"Failed to disable gradient checkpointing: {e}")

    def extract_features(self, x: torch.Tensor):
        """
        Extract features from the MedSigLIP vision model.

        Optionally captures per-head pooler attention weights by temporarily
        monkey-patching the pooler head's ``forward`` method.

        Args:
            x: Images ``[B(*V), C, H, W]``.

        Returns:
            ``(features, attentions)`` where features are L2-normalised.
        """
        self.last_pooler_attn = None

        # Optionally capture pooler attention weights per-head
        head = getattr(self.backbone.vision_model, "head", None)
        original_head_forward = None
        attn_storage = None
        if head is not None and hasattr(head, "attention") and self.return_attentions:
            original_head_forward = head.forward
            setattr(head, "_last_attn_weights", None)

            def forward_with_pooler_attn_capture(hidden_state: torch.Tensor):
                patch_batch_size = hidden_state.shape[0]
                probe = head.probe.repeat(patch_batch_size, 1, 1)
                attn_output, attn_weights = head.attention(
                    probe, hidden_state, hidden_state, average_attn_weights=False
                )
                setattr(head, "_last_attn_weights", attn_weights.detach())
                residual = attn_output
                hidden_state = head.layernorm(attn_output)
                hidden_state = residual + head.mlp(hidden_state)
                return hidden_state[:, 0]

            head.forward = forward_with_pooler_attn_capture

        # SiglipEncoder does not aggregate or return layer attentions. 
        # We therefore capture them from the patched attention modules directly.
        encoder_patches = None
        vision_outputs = None
        if self.return_attentions:
            attn_storage = []
            encoder_patches = patch_siglip_encoder_eager(
                self.backbone.vision_model.encoder,
                storage=attn_storage,
            )

        try:
            vision_outputs = self.backbone.vision_model(
                pixel_values=x,
                output_attentions=self.return_attentions,
                return_dict=True,
            )
            self.last_pooler_attn = getattr(head, "_last_attn_weights", None)
        finally:
            if encoder_patches is not None:
                unpatch_siglip_encoder(encoder_patches)
            if head is not None and original_head_forward is not None:
                head.forward = original_head_forward

        # L2 normalise pooler output
        features = vision_outputs.pooler_output / vision_outputs.pooler_output.norm(
            dim=-1, keepdim=True
        )

        # Stack attention maps
        if self.return_attentions and attn_storage:
            attentions = torch.stack(
                [a.detach().cpu() for a in attn_storage], dim=0
            )
        else:
            attentions = None

        return features, attentions

    def forward(self, pixel_values: torch.Tensor):
        """Forward pass with optional attention map multi-view reshaping."""
        logits, attentions = super().forward(pixel_values)

        # Reshape attention maps for multi-view if needed
        if attentions is not None and self.multi_view:
            batch_size = pixel_values.shape[0]
            num_views = pixel_values.shape[1]
            attentions = attentions.reshape(
                -1, batch_size, num_views, *attentions.shape[2:]
            )

        return logits, attentions
