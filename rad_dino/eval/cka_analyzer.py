"""
Centered Kernel Alignment (CKA) analysis for comparing neural network representations.

1. Layerwise CKA: Compares internal representations between a pretrained and fine-tuned version of the same model.
2. Cross-model CKA: Compares last-layer representations across different foundation models on the same dataset.

References:
    - Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019.
    - Nguyen et al., "Do Wide Neural Networks Really Need to Be Wide?", ICLR 2021.
    - Huix et al., "Are Natural Domain Foundation Models Useful for Medical Image
      Classification?", WACV 2024.
"""

import logging
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rad_dino.models.base import BaseClassifier

logger = logging.getLogger(__name__)


# CKA / HSIC computation helper functions
def _unbiased_hsic(K: torch.Tensor, L: torch.Tensor) -> float:
    """
    Unbiased estimator of the Hilbert-Schmidt Independence Criterion.

    Reference: Song et al., "Feature Selection via Dependence Maximization", JMLR 2012,
    as used in Nguyen et al. (2021) Eq. 3.
    """
    N = K.shape[0]
    ones = torch.ones(N, 1, device=K.device, dtype=K.dtype)

    trace_KL = torch.trace(K @ L)
    sum_K = (ones.t() @ K @ ones).item()
    sum_L = (ones.t() @ L @ ones).item()
    cross = (ones.t() @ K @ L @ ones).item()

    result = trace_KL
    result += (sum_K * sum_L) / ((N - 1) * (N - 2))
    result -= cross * 2 / (N - 2)
    return (1.0 / (N * (N - 3)) * result).item()


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute linear CKA between two feature matrices.

    Args:
        X: Feature matrix of shape ``[N, D1]``.
        Y: Feature matrix of shape ``[N, D2]``.

    Returns:
        CKA similarity score in [0, 1].
    """
    X = X.float()
    Y = Y.float()

    # Gram matrices with zeroed diagonals
    K = X @ X.t()
    K.fill_diagonal_(0.0)
    L = Y @ Y.t()
    L.fill_diagonal_(0.0)

    hsic_KL = _unbiased_hsic(K, L)
    hsic_KK = _unbiased_hsic(K, K)
    hsic_LL = _unbiased_hsic(L, L)

    denom = np.sqrt(hsic_KK * hsic_LL)
    if denom < 1e-12:
        return 0.0
    return hsic_KL / denom


# Layer name resolution helper function
def get_backbone_layer_names(model: BaseClassifier) -> List[str]:
    """
    Return the ordered list of transformer-block-level layer names for
    the backbone inside a ``BaseClassifier``.

    These names correspond to ``nn.Module`` names reachable via
    ``model.named_modules()`` and are suitable for registering forward hooks.
    """
    from rad_dino.models.dino import DinoClassifier
    from rad_dino.models.siglip import MedSigClassifier
    from rad_dino.models.ark import ArkClassifier
    from rad_dino.models.biomedclip import BiomedCLIPClassifier
    from rad_dino.models.medimageinsight import MedImageInsightClassifier

    if isinstance(model, DinoClassifier):
        n = model.backbone.config.num_hidden_layers
        if hasattr(model.backbone, "encoder"):
            return [f"backbone.encoder.layer.{i}" for i in range(n)]
        return [f"backbone.layer.{i}" for i in range(n)]

    if isinstance(model, MedSigClassifier):
        n = model.backbone.config.vision_config.num_hidden_layers
        return [f"backbone.vision_model.encoder.layers.{i}" for i in range(n)]

    if isinstance(model, ArkClassifier):
        layers = []
        for stage_idx, stage in enumerate(model.backbone.layers):
            for block_idx in range(len(stage.blocks)):
                layers.append(f"backbone.layers.{stage_idx}.blocks.{block_idx}")
        return layers

    if isinstance(model, BiomedCLIPClassifier):
        trunk = model.backbone.visual.trunk
        n = len(trunk.blocks)
        return [f"backbone.visual.trunk.blocks.{i}" for i in range(n)]

    if isinstance(model, MedImageInsightClassifier):
        image_encoder = model.backbone.image_encoder
        layers = []
        for stage_idx, stage in enumerate(image_encoder.blocks):
            num_blocks = len(stage)
            for block_idx in range(num_blocks):
                layers.append(
                    f"backbone.image_encoder.blocks.{stage_idx}.{block_idx}"
                )
        return layers

    raise ValueError(f"Unsupported model type for CKA layer resolution: {type(model).__name__}")


# Hook-based feature collection
class _FeatureCollector:
    """Registers forward hooks on named layers and collects their outputs."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self._features: Dict[str, torch.Tensor] = {}
        self._handles: list = []

        name_to_module = dict(model.named_modules())
        for name in layer_names:
            if name not in name_to_module:
                raise KeyError(
                    f"Layer '{name}' not found in model. "
                    f"Available: {list(name_to_module.keys())[:20]}..."
                )
            handle = name_to_module[name].register_forward_hook(
                partial(self._hook_fn, name=name)
            )
            self._handles.append(handle)

    def _hook_fn(self, module, inp, out, *, name: str):
        if isinstance(out, tuple):
            out = out[0]
        self._features[name] = out.detach()

    @property
    def features(self) -> Dict[str, torch.Tensor]:
        return self._features

    def clear(self):
        self._features.clear()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# Layerwise CKA (pretrained vs fine-tuned)
@torch.no_grad()
def compute_layerwise_cka(
    model1: BaseClassifier,
    model2: BaseClassifier,
    dataloader: DataLoader,
    device: torch.device,
    layers: Optional[List[str]] = None,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the layerwise CKA matrix between pretrained and fine-tuned models.

    Args:
        model1: Pretrained model.
        model2: Fine-tuned model.
        dataloader: Test dataloader {"pixel_values": ..., "labels": ...}.
        device: Torch device.
        layers: Layer names to compare. 
                If None, auto-resolved via get_backbone_layer_names(model1).
        max_batches: If set, only use this many batches for CKA (subsampling).

    Returns:
        (cka_matrix, layer_names) where cka_matrix has shape [N_layers, N_layers].
    """
    if layers is None:
        layers = get_backbone_layer_names(model1)
    N = len(layers)

    total_batches = len(dataloader)
    effective_batches = min(total_batches, max_batches) if max_batches else total_batches
    logger.info(
        "Computing layerwise CKA over %d layers, %d/%d batches",
        N, effective_batches, total_batches,
    )

    model1.eval()
    model2.eval()

    # Separate accumulators: self-HSIC per layer (1-D) and cross-HSIC (2-D)
    hsic_kk_accum = torch.zeros(N, device="cpu")
    hsic_ll_accum = torch.zeros(N, device="cpu")
    hsic_kl_accum = torch.zeros(N, N, device="cpu")

    collector1 = _FeatureCollector(model1, layers)
    collector2 = _FeatureCollector(model2, layers)

    try:
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="CKA layerwise", total=effective_batches, leave=False)
        ):
            if max_batches and batch_idx >= max_batches:
                break

            images = batch["pixel_values"].to(device)

            collector1.clear()
            collector2.clear()

            model1(images)
            model2(images)

            feat1_list = [collector1.features[name].flatten(1) for name in layers]
            feat2_list = [collector2.features[name].flatten(1) for name in layers]

            # Precompute gram matrices and self-HSIC for all layers (2*N instead of N^2)
            grams1 = []
            for i in range(N):
                X = feat1_list[i].float()
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                grams1.append(K)
                hsic_kk_accum[i] += _unbiased_hsic(K, K) / effective_batches

            grams2 = []
            for j in range(N):
                Y = feat2_list[j].float()
                L = Y @ Y.t()
                L.fill_diagonal_(0.0)
                grams2.append(L)
                hsic_ll_accum[j] += _unbiased_hsic(L, L) / effective_batches

            # Cross-HSIC
            for i in range(N):
                for j in range(N):
                    hsic_kl_accum[i, j] += _unbiased_hsic(grams1[i], grams2[j]) / effective_batches
    finally:
        collector1.remove_hooks()
        collector2.remove_hooks()

    # CKA[i,j] = HSIC(K_i, L_j) / sqrt(HSIC(K_i, K_i) * HSIC(L_j, L_j))
    # The unbiased HSIC estimator can yield negative values with small batch sizes, 
    # causing sqrt(neg) -> NaN.  To avoid this, we clamp the product to zero first.
    prod = hsic_kk_accum[:, None] * hsic_ll_accum[None, :]
    denom = torch.sqrt(torch.clamp(prod, min=0.0))
    denom = torch.clamp(denom, min=1e-12)
    cka_matrix = torch.clamp(hsic_kl_accum / denom, min=0.0, max=1.0).numpy()

    logger.info("Layerwise CKA matrix computed: shape %s", cka_matrix.shape)
    return cka_matrix, layers


# Cross-model CKA (last-layer feature representation comparison)
def compute_crossmodel_cka(
    features_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise CKA between last-layer features from multiple models.

    Args:
        features_dict: Mapping {model_name: features} where features has
                       shape [N_samples, D]. 

    Returns:
        (cka_matrix, model_names) where cka_matrix has shape [M, M] for M models.
    """
    model_names = list(features_dict.keys())
    M = len(model_names)
    cka_matrix = np.ones((M, M), dtype=np.float64)

    logger.info("Computing cross-model CKA for %d models", M)
    for i in range(M):
        X = torch.from_numpy(features_dict[model_names[i]])
        for j in range(i + 1, M):
            Y = torch.from_numpy(features_dict[model_names[j]])
            score = linear_cka(X, Y)
            cka_matrix[i, j] = score
            cka_matrix[j, i] = score
            logger.info(
                "  CKA(%s, %s) = %.3f", model_names[i], model_names[j], score
            )

    return cka_matrix, model_names