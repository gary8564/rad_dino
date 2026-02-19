"""
KNN evaluation of pretrained vision backbones.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
from accelerate import Accelerator

from rad_dino.eval.feature_extractor import (
    add_args,
    build_backbone_model,
    knn_classify,
    save_results,
    setup_data_and_features,
    validate_args,
)
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("DINOv2-style KNN evaluation")
    add_args(parser)
    parser.add_argument(
        "--nb-knn", nargs="+", type=int, default=[20],
        help="Number of nearest neighbours to evaluate. "
             "Default: 20",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07,
        help="Temperature for softmax voting (default: 0.07).",
    )
    return parser


def run_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    nb_knn_list: List[int],
    temperature: float,
    task: str,
    num_classes: int,
    class_labels: Optional[List],
    output_base: str,
    accelerator: Accelerator,
) -> None:
    """Run KNN evaluation for each K value."""
    device = accelerator.device

    train_feat_t = torch.from_numpy(train_features).to(device)
    test_feat_t = torch.from_numpy(test_features).to(device)

    if task == "binary":
        train_labels_t = torch.from_numpy(
            train_labels.squeeze().astype(np.int64)
        ).to(device)
        knn_num_classes = 2
    else:
        train_labels_t = torch.from_numpy(
            train_labels.astype(np.int64)
        ).to(device)
        knn_num_classes = num_classes

    for k in nb_knn_list:
        logger.info(f"Running KNN with K={k}, T={temperature}")
        probas = knn_classify(
            train_feat_t, train_labels_t, test_feat_t,
            k, temperature, knn_num_classes,
        )
        probas_np = probas.cpu().numpy()

        if task == "binary":
            probas_np = probas_np[:, 1:2]

        output_dir = os.path.join(output_base, f"knn_k{k}")
        save_results(
            probas_np, test_labels, task, class_labels,
            output_dir, accelerator,
        )


def main():
    args = get_args_parser().parse_args()
    validate_args(args)

    accelerator = Accelerator(
        mixed_precision="fp16" if args.optimize_compute else "no"
    )

    model = build_backbone_model(args, accelerator.device)

    (
        train_features, train_labels,
        test_features, test_labels,
        num_classes, class_labels,
    ) = setup_data_and_features(args, accelerator, model)

    output_base = os.path.join(
        args.output_path, args.data,
        f"knn_eval_{CURR_TIME}_{args.data}_{args.model}",
    )
    if accelerator.is_main_process:
        os.makedirs(output_base, exist_ok=True)

    run_knn(
        train_features, train_labels,
        test_features, test_labels,
        nb_knn_list=args.nb_knn,
        temperature=args.temperature,
        task=args.task,
        num_classes=num_classes,
        class_labels=class_labels,
        output_base=output_base,
        accelerator=accelerator,
    )

    logger.info("KNN evaluation complete.")


if __name__ == "__main__":
    main()
