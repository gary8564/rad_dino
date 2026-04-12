"""
Embedding visualization of pretrained vision backbones (UMAP / supervised UMAP / t-SNE).

Extracts features from a frozen backbone for train and test splits,
combines them, and generates a scatter plot coloured by class label.

Usage::
    python rad_dino/run/visualize_embeddings.py \
        --task binary --data NODE21 --model dinov2-large \
        --method umap --output-path ./experiments/umap

    python rad_dino/run/visualize_embeddings.py \
        --task binary --data NODE21 --model dinov2-large \
        --method supervised-umap --output-path ./experiments/supervised_umap

    python rad_dino/run/visualize_embeddings.py \
        --task binary --data NODE21 --model dinov2-large \
        --method tsne --perplexity 30 --output-path ./experiments/tsne
"""

import argparse
import logging
import os
from datetime import datetime

from accelerate import Accelerator

from rad_dino.eval.feature_extractor import (
    build_backbone_model,
    setup_data_and_features,
    MODEL_REPOS,
    DEFAULT_MEDIMAGEINSIGHT_PATH,
)
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.config_utils import validate_dataset
from rad_dino.utils.visualization.visualize_embeddings import (
    visualize_feature_embeddings,
)

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embedding visualization (UMAP / supervised UMAP / t-SNE) for pretrained vision backbones"
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["binary", "multiclass"],
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Dataset name (must match a key in data_config.yaml)",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_REPOS.keys()) + ["biomedclip", "medimageinsight"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--method", type=str, default="umap",
        choices=["umap", "tsne", "supervised-umap"],
        help="Dimensionality reduction method (default: umap).",
    )
    parser.add_argument(
        "--optimize-compute", action="store_true",
        help="Use mixed precision (fp16) for feature extraction.",
    )
    parser.add_argument(
        "--pretrained-ark-path", type=str, default=None,
        help="Path to Ark pre-trained checkpoint.",
    )
    parser.add_argument(
        "--medimageinsight-path", type=str,
        default=DEFAULT_MEDIMAGEINSIGHT_PATH,
    )

    # UMAP hyper-parameters
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP locality parameter (ignored for t-SNE).")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP minimum distance (ignored for t-SNE).")

    # t-SNE hyper-parameters
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity; roughly how many neighbours to consider (ignored for UMAP).")
    parser.add_argument("--learning-rate", type=float, default=200.0,
                        help="t-SNE learning rate (ignored for UMAP).")
    parser.add_argument("--n-iter", type=int, default=1000,
                        help="t-SNE max iterations (ignored for UMAP).")

    # Shared
    parser.add_argument(
        "--metric", type=str, default="cosine",
        choices=["cosine", "euclidean", "correlation"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = get_args_parser().parse_args()
    validate_dataset(args.data)

    if args.model == "ark" and args.pretrained_ark_path is None:
        raise ValueError("Ark requires --pretrained-ark-path.")

    accelerator = Accelerator(
        mixed_precision="fp16" if args.optimize_compute else "no"
    )

    model = build_backbone_model(args, accelerator.device)

    (
        train_features, train_labels,
        test_features, test_labels,
        num_classes, class_labels,
    ) = setup_data_and_features(args, accelerator, model)

    output_dir = os.path.join(
        args.output_path, args.data,
        f"{args.method}_{CURR_TIME}_{args.data}_{args.model}",
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    path = visualize_feature_embeddings(
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
        dataset_name=args.data,
        model_name=args.model,
        task=args.task,
        output_dir=output_dir,
        method=args.method,
        class_labels=class_labels,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        n_iter=args.n_iter,
        metric=args.metric,
        random_state=args.random_state,
    )

    logger.info(f"{args.method.upper()} plot saved: {path}")
    logger.info(f"{args.method.upper()} visualization complete.")


if __name__ == "__main__":
    main()
