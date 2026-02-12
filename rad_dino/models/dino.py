import torch
import logging
from typing import Optional

from rad_dino.models.base import BaseClassifier
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


class DinoClassifier(BaseClassifier):
    """
    DINO-family classifier.

    Features are extracted from the CLS token of the last hidden state
    Additional supports attention map extraction and gradient checkpointing.
    """

    def __init__(
        self,
        backbone,
        num_classes: int,
        multi_view: bool = False,
        num_views: Optional[int] = None,
        view_fusion_type: Optional[str] = None,
        adapter_dim: Optional[int] = None,
        view_fusion_hidden_dim: Optional[int] = None,
        return_attentions: bool = False,
        gradient_checkpointing: bool = False,
    ):
        embed_dim = backbone.config.hidden_size
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

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self):
        try:
            self.backbone.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on DINO backbone")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    def disable_gradient_checkpointing(self):
        try:
            self.backbone.gradient_checkpointing_disable()
            logger.info("Gradient checkpointing disabled for DINO model")
        except Exception as e:
            logger.warning(f"Failed to disable gradient checkpointing: {e}")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor):
        """
        Extract CLS token features from the DINO backbone.

        Args:
            x: Images ``[B(*V), C, H, W]``.

        Returns:
            ``(cls_tokens, attentions)``
        """
        outputs = self.backbone(
            x,
            output_attentions=self.return_attentions,
            return_dict=True,
        )
        cls_tokens = outputs.last_hidden_state[:, 0]

        if (
            self.return_attentions
            and hasattr(outputs, "attentions")
            and outputs.attentions is not None
        ):
            attentions = torch.stack(outputs.attentions, dim=0)
        else:
            attentions = None

        return cls_tokens, attentions

    # ------------------------------------------------------------------
    # Forward override — adds attention multi-view reshaping
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """Forward pass with optional attention map multi-view reshaping."""
        # Use shared base forward for input reshape, fusion, classification
        logits, attentions = super().forward(x)

        # Reshape attention maps for multi-view if needed
        if attentions is not None and self.multi_view:
            batch_size = x.shape[0]
            num_views = x.shape[1]
            # attentions: [L, B*V, N_heads, N_seq, N_seq]
            # -> [L, B, V, N_heads, N_seq, N_seq]
            attentions = attentions.reshape(
                -1, batch_size, num_views, *attentions.shape[2:]
            )

        # Use self.head for classification (backward compat)
        # Actually, super().forward() already used self.classifier which
        # is the same linear layer inside self.head. Logits are correct.
        return logits, attentions


if __name__ == "__main__":
    from transformers import AutoModel

    def unfreeze_layers(model, num_unfreeze_layers):
        num_total_layers = model.backbone.config.num_hidden_layers
        assert num_total_layers == 12, "Number of total layers is not 12"
        assert num_unfreeze_layers <= num_total_layers
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
        for i in range(num_total_layers - 1, num_total_layers - num_unfreeze_layers - 1, -1):
            for name, param in model.backbone.named_parameters():
                if f"layer.{i}" in name:
                    param.requires_grad = True

    backbone = AutoModel.from_pretrained('facebook/dinov2-base')
    model = DinoClassifier(backbone, num_classes=10)
    model_with_grad_ckpt = DinoClassifier(backbone, num_classes=10, gradient_checkpointing=True)

    num_layers = model.backbone.config.num_hidden_layers
    print(f"Number of transformer blocks: {num_layers}")

    unfreeze_layers(model, 2)
    for name, param in model.named_parameters():
        if 'backbone' in name:
            if param.requires_grad:
                print(f"Trainable backbone parameter: {name}")
        else:
            param.requires_grad = True
            print(f"Trainable parameter: {name}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
