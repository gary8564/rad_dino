import torch
import logging
from typing import Optional, List
from open_clip import create_model_from_pretrained, get_tokenizer

from rad_dino.models.base import BaseClassifier
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

BIOMEDCLIP_HF_REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_biomedclip_model(device: str = "cuda"):
    """
    Load the BiomedCLIP model from the HuggingFace Hub via open_clip.

    Uses ``open_clip.create_model_from_pretrained`` to download and
    initialise the BiomedCLIP model with ViT-B/16 as image encoder and PubMedBERT as text encoder.

    Args:
        device: Target device, e.g. "cuda" or "cpu".

    Returns:
        model: The loaded open_clip model (nn.Module) moved to *device*.
        preprocess: The image preprocessing transform returned by open_clip.
    """
    model, preprocess = create_model_from_pretrained(
        f"hf-hub:{BIOMEDCLIP_HF_REPO}"
    )
    model.to(device)
    logger.info(
        "Loaded BiomedCLIP model from %s (device=%s)", BIOMEDCLIP_HF_REPO, device
    )
    return model, preprocess


def get_biomedclip_tokenizer():
    """Return the open_clip tokenizer for BiomedCLIP."""
    return get_tokenizer(f"hf-hub:{BIOMEDCLIP_HF_REPO}")


# Attention extraction helper function for open-clip
def _hook_attn(attn_module, storage: list):
    def forward(x):
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, attn_module.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * attn_module.scale
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)

        storage.append(attn.detach())

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_module.proj(x)
        x = attn_module.proj_drop(x)
        return x
    return forward


def _extract_timm_backbone(backbone):
    """Extract the timm VisionTransformer from the BiomedCLIP model."""
    if hasattr(backbone, "visual") and hasattr(backbone.visual, "trunk"):
        return backbone.visual.trunk
    return None


class BiomedCLIPClassifier(BaseClassifier):
    """
    Linear classifier with BiomedCLIP (open_clip ViT-B/16) backbone.

    Features are extracted via ``backbone.encode_image(x)`` which returns
    L2-normalised 512-dimensional image embeddings.

    Supports optional attention map extraction by monkey-patching timm's
    ``Attention.forward()`` to capture the attention weights.
    """

    EMBED_DIM = 512  # BiomedCLIP projection dimension

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
    ):
        super().__init__(
            backbone=backbone,
            embed_dim=self.EMBED_DIM,
            num_classes=num_classes,
            multi_view=multi_view,
            num_views=num_views,
            view_fusion_type=view_fusion_type,
            adapter_dim=adapter_dim,
            view_fusion_hidden_dim=view_fusion_hidden_dim,
        )
        self.return_attentions = return_attentions
        self._original_forwards: list = []
        self._attn_storage: List[torch.Tensor] = []

    # Capture attention hooks
    def _enable_attention_hooks(self):
        """Monkey-patch all timm Attention blocks to capture attention weights."""
        self._attn_storage.clear()
        self._original_forwards.clear()

        trunk = _extract_timm_backbone(self.backbone)
        if trunk is None or not hasattr(trunk, "blocks"):
            logger.warning("Cannot find visual trunk blocks — attention capture disabled")
            return

        for block in trunk.blocks:
            attn_module = block.attn
            original_fwd = attn_module.forward
            self._original_forwards.append((attn_module, original_fwd))
            attn_module.forward = _hook_attn(attn_module, self._attn_storage)

    def _disable_attention_hooks(self):
        """Restore original forward methods."""
        for attn_module, original_fwd in self._original_forwards:
            attn_module.forward = original_fwd
        self._original_forwards.clear()

    def _collect_attentions(self) -> Optional[torch.Tensor]:
        """Stack per-layer attention tensors into [num_layers, B, H, N, N]."""
        if not self._attn_storage:
            return None
        return torch.stack(self._attn_storage, dim=0)

    # Feature extraction
    def extract_features(self, x: torch.Tensor):
        """
        Extract L2-normalised image features via BiomedCLIP's vision encoder.   
        When return_attentions=True, captures per-layer attention weights
        from all transformer blocks.

        Returns:
            (features, attentions_or_None) where attentions has shape
            [num_layers, B, num_heads, seq_len, seq_len].
        """
        if self.return_attentions:
            self._enable_attention_hooks()

        features = self.backbone.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)

        attentions = None
        if self.return_attentions:
            attentions = self._collect_attentions()
            self._disable_attention_hooks()

        return features, attentions

    # Forward override - adds attention multi-view reshaping
    def forward(self, x: torch.Tensor):
        """Forward pass with optional attention map multi-view reshaping."""
        logits, attentions = super().forward(x)

        if attentions is not None and self.multi_view:
            batch_size = x.shape[0]
            num_views = x.shape[1]
            attentions = attentions.reshape(
                -1, batch_size, num_views, *attentions.shape[2:]
            )

        return logits, attentions
