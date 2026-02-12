import torch
import logging
from typing import Optional
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


class BiomedCLIPClassifier(BaseClassifier):
    """
    Linear classifier with BiomedCLIP (open_clip ViT-B/16) backbone.

    Features are extracted via ``backbone.encode_image(x)`` which returns
    L2-normalised 512-dimensional image embeddings.

    ``forward()`` returns ``(logits, None)`` — attention maps are not
    currently supported for the open_clip ViT.
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

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor):
        """
        Extract L2-normalised image features via BiomedCLIP's vision encoder.

        Returns:
            ``(features, None)`` — attention maps not supported.
        """
        features = self.backbone.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features, None
