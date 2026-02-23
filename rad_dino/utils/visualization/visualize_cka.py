"""
CKA (Centered Kernel Alignment) visualization.

Provides two plotting functions:
1. ``plot_layerwise_cka`` — heatmap of block-vs-block CKA between pretrained
   and fine-tuned versions of the same model (reproduces Figure 7 of
   Huix et al., WACV 2024).
2. ``plot_crossmodel_cka`` — annotated NxN matrix of last-layer CKA across
   different foundation models (reproduces Figure 8 of Huix et al., WACV 2024).
"""

import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_layerwise_cka(
    cka_matrix: np.ndarray,
    layer_names: List[str],
    model_name: str,
    dataset_name: str,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot a layerwise CKA heatmap (pretrained vs fine-tuned).

    Args:
        cka_matrix: Square CKA matrix of shape ``[N_layers, N_layers]``.
        layer_names: Ordered layer names (used for tick labels).
        model_name: Display name for the model.
        dataset_name: Dataset name (shown in title).
        save_path: File path to save the figure.
        title: Custom title; auto-generated if None.
    """
    N = len(layer_names)
    tick_labels = [str(i) for i in range(N)]

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cka_matrix,
        vmin=0, vmax=1,
        square=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xlabel("Blocks (fine-tuned)", fontsize=12)
    ax.set_ylabel("Blocks (pre-trained)", fontsize=12)
    ax.tick_params(labelsize=7)
    ax.invert_yaxis()

    if title is None:
        title = f"{model_name} — {dataset_name}"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Layerwise CKA heatmap saved to %s", save_path)


def plot_crossmodel_cka(
    cka_matrix: np.ndarray,
    model_names: List[str],
    dataset_name: str,
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Plot an annotated NxN heatmap of last-layer CKA across foundation models.

    Args:
        cka_matrix: Symmetric CKA matrix of shape ``[M, M]``.
        model_names: Model display names.
        dataset_name: Dataset name (shown in title).
        save_path: File path to save the figure.
        title: Custom title; auto-generated if None.
    """
    M = len(model_names)
    fig, ax = plt.subplots(figsize=(max(5, M * 0.9 + 1), max(4, M * 0.8 + 1)))

    sns.heatmap(
        cka_matrix,
        vmin=0, vmax=1,
        annot=True, fmt=".2f",
        square=True,
        xticklabels=model_names,
        yticklabels=model_names,
        annot_kws={"size": 8, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    if title is None:
        title = f"{dataset_name}"
    ax.set_title(title, fontsize=14, pad=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cross-model CKA heatmap saved to %s", save_path)
