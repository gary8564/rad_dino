"""
Embedding visualization for pretrained vision backbones.

Extracts backbone features, reduces dimensionality with UMAP or t-SNE,
and produces scatter plots coloured by class label so that clustering
behaviour can be inspected visually for each model / dataset combination.
"""

import logging
import os
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP

logger = logging.getLogger(__name__)

BINARY_LABEL_NAMES = {
    "NODE21": {0: "Normal", 1: "Nodule"},
    "VinDr-SpineXR": {0: "Normal", 1: "Abnormal"},
    "COVID-CXR": {0: "Normal", 1: "COVID-19"},
    "VinDr-Mammo": {0: "Benign", 1: "Malignant"},
}

# High-contrast, colorblind-safe palette inspired by Okabe-Ito/Tableau.
DISTINCT_CLASS_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#CC79A7",  # magenta
]

def _get_distinct_palette(n_classes: int) -> List:
    """Return a high-contrast palette for categorical class coloring."""
    if n_classes <= len(DISTINCT_CLASS_COLORS):
        return DISTINCT_CLASS_COLORS[:n_classes]
    return sns.color_palette("husl", n_colors=n_classes)


def _resolve_label_names(
    labels: np.ndarray,
    dataset_name: str,
    task: str,
    class_labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Map integer labels to the class label names for the plot legend."""
    unique = sorted(np.unique(labels).astype(int).tolist())

    if class_labels is not None:
        mapping = {i: name for i, name in enumerate(class_labels)}
    elif task == "binary" and dataset_name in BINARY_LABEL_NAMES:
        mapping = BINARY_LABEL_NAMES[dataset_name]
    else:
        mapping = {i: f"Class {i}" for i in unique}

    named = np.array([mapping.get(int(idx), f"Class {int(idx)}") for idx in labels])
    ordered_names = [mapping.get(i, f"Class {i}") for i in unique]
    return named, ordered_names


def compute_umap(
    features: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Run UMAP dimensionality reduction on the feature vectors.

    Args:
        features: L2-normalised feature vectors.
        n_components: Target dimensionality (2 for scatter plots).
        n_neighbors: UMAP locality parameter.
        min_dist: UMAP minimum distance.
        metric: Distance metric.
        random_state: Seed for reproducibility.
        labels: If provided, run supervised UMAP (label-guided projection).

    Returns:
        [N, n_components] embedding array.
    """
    supervised = labels is not None
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(features, y=labels)
    mode = "supervised" if supervised else "unsupervised"
    logger.info(
        f"UMAP ({mode}): {features.shape} -> {embedding.shape} "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric})"
    )
    return embedding


def compute_tsne(
    features: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """
    Run t-SNE dimensionality reduction on the feature vectors.

    Args:
        features: L2-normalised feature vectors.
        n_components: Target dimensionality (2 for scatter plots).
        perplexity: Related to the number of nearest neighbors
                    (typical range 5-50; larger datasets benefit from higher values).
        learning_rate: Step size for gradient descent (typically 10-1000).
        n_iter: Maximum number of optimisation iterations.
        metric: Distance metric.
        random_state: Seed for reproducibility.

    Returns:
        [N, n_components] embedding array.
    """
    reducer = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric=metric,
        random_state=random_state,
        init="pca",
    )
    embedding = reducer.fit_transform(features)
    logger.info(
        f"t-SNE: {features.shape} -> {embedding.shape} "
        f"(perplexity={perplexity}, lr={learning_rate}, n_iter={n_iter}, metric={metric})"
    )
    return embedding


def plot_embedding(
    embedding: np.ndarray,
    label_names: np.ndarray,
    ordered_names: List[str],
    output_path: str,
    method: str = "umap",
    figsize: Tuple[int, int] = (10, 8),
    point_size: int = 10,
    alpha: float = 0.45,
    dpi: int = 300,
) -> None:
    """
    2-D scatter plot of a reduced embedding.

    Args:
        embedding: [N, 2] coordinates.
        label_names: [N] string label for each point.
        ordered_names: Ordered legend entries.
        output_path: Where to save the figure.
        method: "umap", "tsne", or "supervised-umap".
        figsize: Figure size in inches.
        point_size: Marker size.
        alpha: Marker transparency.
        dpi: Output resolution.
    """
    palette = _get_distinct_palette(len(ordered_names))
    color_map = {name: palette[i] for i, name in enumerate(ordered_names)}

    # Plot majority class first so minority (disease) sits on top
    class_counts = {name: (label_names == name).sum() for name in ordered_names}
    plot_order = sorted(ordered_names, key=lambda n: class_counts[n], reverse=True)

    fig, ax = plt.subplots(figsize=figsize)

    for name in plot_order:
        mask = label_names == name
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color_map[name],
            label=name,
            s=point_size,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.3,
        )

    ax.legend(
        markerscale=3,
        fontsize=14,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        loc="best",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(left=True, bottom=True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {method.upper()} plot: {output_path}")


def visualize_feature_embeddings(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    dataset_name: str,
    model_name: str,
    task: str,
    output_dir: str,
    method: Literal["umap", "tsne", "supervised-umap"] = "umap",
    class_labels: Optional[List[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    metric: str = "cosine",
    random_state: int = 42,
) -> str:
    """
    Embedding visualization using UMAP, supervised UMAP, or t-SNE.

    Args:
        train_features: [N_train, D] feature vectors.
        train_labels: [N_train] integer class labels.
        test_features: [N_test, D] feature vectors.
        test_labels: [N_test] integer class labels.
        dataset_name: Dataset name.
        model_name: Model name.
        task: "binary" or "multiclass".
        output_dir: Base output directory.
        method: "umap", "supervised-umap", or "tsne".
        class_labels: Ordered class label names
                      (If multiclass, provided in ``label_mapping.py``).
        n_neighbors: UMAP locality parameter (ignored for t-SNE).
        min_dist: UMAP minimum distance (ignored for t-SNE).
        perplexity: t-SNE perplexity (ignored for UMAP).
        learning_rate: t-SNE learning rate (ignored for UMAP).
        n_iter: t-SNE max iterations (ignored for UMAP).
        metric: Distance metric.
        random_state: Seed for reproducibility.

    Returns:
        Path to the saved figure.
    """
    features = np.concatenate([train_features, test_features], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    label_names, ordered_names = _resolve_label_names(
        labels, dataset_name, task, class_labels
    )

    if method == "tsne":
        embedding = compute_tsne(
            features,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            random_state=random_state,
        )
    elif method == "supervised-umap":
        embedding = compute_umap(
            features,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            labels=labels,
        )
    else:
        embedding = compute_umap(
            features,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

    fname = f"{method}_{model_name}_{dataset_name}.png"
    output_path = os.path.join(output_dir, fname)

    plot_embedding(
        embedding,
        label_names,
        ordered_names,
        output_path=output_path,
        method=method,
    )
    return output_path
