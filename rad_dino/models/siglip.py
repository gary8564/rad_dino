import torch
import logging
from typing import Optional
from transformers import AutoModel

from rad_dino.models.base import BaseClassifier
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


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
        embed_dim = backbone.config.text_config.projection_size
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

        # Backward-compatible aliases for old checkpoint state_dict keys
        # Old checkpoints use ``feat_dim`` nowhere in state_dict (it was just
        # an instance attr), but ``fusion_layer`` was used instead of
        # ``view_fusion_layer`` for the mlp_adapter / weighted_mean layer.
        # We'll handle that in the checkpoint loader.
        self.feat_dim = self.embed_dim  # alias

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Feature extraction (with optional pooler attention capture)
    # ------------------------------------------------------------------

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
        # Optionally capture pooler attention weights per-head
        head = getattr(self.backbone.vision_model, "head", None)
        original_head_forward = None
        if head is not None and hasattr(head, "attention") and self.return_attentions:
            original_head_forward = head.forward

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

        # Process through vision model
        vision_outputs = self.backbone.vision_model(
            pixel_values=x,
            output_attentions=self.return_attentions,
            return_dict=True,
        )

        # Restore original forward and capture weights
        if head is not None and original_head_forward is not None:
            self.last_pooler_attn = getattr(head, "_last_attn_weights", None)
            head.forward = original_head_forward

        # L2 normalise pooler output
        features = vision_outputs.pooler_output / vision_outputs.pooler_output.norm(
            dim=-1, keepdim=True
        )

        # Stack attention maps
        if (
            self.return_attentions
            and hasattr(vision_outputs, "attentions")
            and vision_outputs.attentions is not None
        ):
            attentions = torch.stack(vision_outputs.attentions, dim=0)
        else:
            attentions = None

        return features, attentions

    # ------------------------------------------------------------------
    # Forward override — adds attention multi-view reshaping
    # ------------------------------------------------------------------

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


if __name__ == "__main__":
    import os
    from transformers import AutoModel
    from dotenv import load_dotenv, find_dotenv
    from huggingface_hub import login

    load_dotenv(find_dotenv())
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    def unfreeze_layers(model, num_unfreeze_layers):
        num_total_layers = model.backbone.config.vision_config.num_hidden_layers
        assert num_unfreeze_layers <= num_total_layers
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        for i in range(num_total_layers - 1, num_total_layers - num_unfreeze_layers - 1, -1):
            for name, param in model.backbone.named_parameters():
                if f"vision_model.encoder.layers.{i}" in name:
                    param.requires_grad = True

    backbone = AutoModel.from_pretrained("google/medsiglip-448")

    model_single = MedSigClassifier(backbone, num_classes=10, multi_view=False, return_attentions=True)
    print("Single-view model created successfully")

    model_multi = MedSigClassifier(backbone, num_classes=10, multi_view=True, num_views=4, view_fusion_type="mean", return_attentions=True)
    print("Multi-view model created successfully")

    model_weighted = MedSigClassifier(backbone, num_classes=10, multi_view=True, num_views=4, view_fusion_type="weighted_mean", return_attentions=True)
    model_adapter = MedSigClassifier(backbone, num_classes=10, multi_view=True, num_views=4, view_fusion_type="mlp_adapter", return_attentions=True)
    print("All fusion types created successfully")

    model_with_grad_ckpt = MedSigClassifier(backbone, num_classes=10, multi_view=False, gradient_checkpointing=True)
    print("Model with gradient checkpointing created successfully")

    unfreeze_layers(model_multi, 2)
    for name, param in model_multi.named_parameters():
        if "backbone" in name:
            if param.requires_grad:
                print(f"Parameter name: {name}")
        else:
            param.requires_grad = True
            print(f"Parameter name: {name}")

    total_params = sum(p.numel() for p in model_multi.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model_multi.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")

    dummy_input_single = torch.randn(2, 3, 448, 448)
    logits_single, attns_single = model_single(dummy_input_single)
    print(f"Single-view output shapes: logits {logits_single.shape}, attention maps {attns_single.shape}")

    dummy_input_multi = torch.randn(2, 4, 3, 448, 448)
    logits_multi, attns_multi = model_multi(dummy_input_multi)
    print(f"Multi-view output shapes: logits {logits_multi.shape}, attention maps {attns_multi.shape}")
