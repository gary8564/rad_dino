"""
Linear SVM evaluation of pretrained vision backbones.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from accelerate import Accelerator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from rad_dino.eval.feature_extractor import (
    add_args,
    build_backbone_model,
    save_results,
    setup_data_and_features,
    validate_args,
)
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Linear SVM evaluation")
    add_args(parser)
    parser.add_argument(
        "--max-iter", type=int, default=5000,
        help="Max iterations for LinearSVC (default: 5000).",
    )
    return parser


def run_svm(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    task: str,
    num_classes: int,
    class_labels: Optional[List],
    output_dir: str,
    accelerator: Accelerator,
    max_iter: int = 5000,
) -> None:
    """Run linear SVM evaluation with Platt scaling for probability outputs."""
    logger.info(f"Running Linear SVM (max_iter={max_iter})")

    base_svm = LinearSVC(max_iter=max_iter, class_weight="balanced", dual="auto")
    clf = CalibratedClassifierCV(base_svm, cv=3)

    if task == "binary":
        y_train = train_labels.squeeze().astype(int)
        clf.fit(train_features, y_train)
        probas = clf.predict_proba(test_features)[:, 1:2]
    else:
        y_train = train_labels.astype(int)
        clf.fit(train_features, y_train)
        probas = clf.predict_proba(test_features)

    save_results(probas, test_labels, task, class_labels, output_dir, accelerator)


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

    output_dir = os.path.join(
        args.output_path, args.data,
        f"svm_eval_{CURR_TIME}_{args.data}_{args.model}",
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    run_svm(
        train_features, train_labels,
        test_features, test_labels,
        task=args.task,
        num_classes=num_classes,
        class_labels=class_labels,
        output_dir=output_dir,
        accelerator=accelerator,
        max_iter=args.max_iter,
    )

    logger.info("SVM evaluation complete.")


if __name__ == "__main__":
    main()
