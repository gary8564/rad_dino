"""
CKA (Centered Kernel Alignment) analysis.

Usage examples::
    # Layerwise CKA (pretrained vs fine-tuned)
    python rad_dino/run/cka.py \\
        --mode layerwise \\
        --model dinov2-large \\
        --checkpoint-dir /path/to/finetuned/checkpoint \\
        --data TBX11 --task binary \\
        --output-path /path/to/output

    # Cross-model CKA
    python rad_dino/run/cka.py \\
        --mode crossmodel \\
        --models dinov2-large rad-dino medsiglip \\
        --checkpoint-dirs /path/to/ckpt1 /path/to/ckpt2 /path/to/ckpt3 \\
        --data TBX11 --task binary \\
        --output-path /path/to/output
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import torch
from accelerate import Accelerator

from rad_dino.data.data_loader import create_test_loader
from rad_dino.data.dataset import RadImageClassificationDataset
from rad_dino.eval.cka_analyzer import (
    compute_crossmodel_cka,
    compute_layerwise_cka,
)
from rad_dino.eval.feature_extractor import (
    MODEL_REPOS,
    build_backbone_model,
    extract_features,
)
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.config_utils import setup_configs
from rad_dino.utils.model_loader import load_model
from rad_dino.utils.transforms import get_transforms
from rad_dino.utils.visualization.visualize_cka import (
    plot_crossmodel_cka,
    plot_layerwise_cka,
)

init_logging()
logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")
DEFAULT_MEDIMAGEINSIGHT_PATH = os.path.normpath(
    os.path.join(CURR_DIR, "..", "models", "MedImageInsights")
)

ALL_MODELS = list(MODEL_REPOS.keys()) + ["medimageinsight", "biomedclip"]


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("CKA analysis for foundation models")

    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["layerwise", "crossmodel"],
        help="Analysis mode: 'layerwise' (pretrained vs fine-tuned) or "
             "'crossmodel' (last-layer CKA across FMs).",
    )
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["binary", "multiclass"],
    )
    parser.add_argument(
        "--data", type=str, required=True,
        choices=["TBX11", "SIIM-ACR", "VinDr-Mammo", "NODE21"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--optimize-compute", action="store_true",
        help="Use mixed precision (fp16).",
    )

    # --- layerwise mode ---
    parser.add_argument(
        "--model", type=str, default=None, choices=ALL_MODELS,
        help="Model name (layerwise mode).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Path to the fine-tuned checkpoint directory (layerwise mode).",
    )

    # --- crossmodel mode ---
    parser.add_argument(
        "--models", nargs="+", type=str, default=None,
        help="List of model names (crossmodel mode).",
    )
    parser.add_argument(
        "--checkpoint-dirs", nargs="+", type=str, default=None,
        help="List of fine-tuned checkpoint directories, one per model "
             "(crossmodel mode).",
    )

    # --- optional model-specific paths ---
    parser.add_argument(
        "--pretrained-ark-path", type=str, default=None,
        help="Path to Ark pre-trained checkpoint.",
    )
    parser.add_argument(
        "--medimageinsight-path", type=str,
        default=DEFAULT_MEDIMAGEINSIGHT_PATH,
        help="Path to cloned MedImageInsights repository.",
    )

    return parser

def _validate_layerwise_args(args):
    if args.model is None:
        raise ValueError("--model is required for layerwise mode.")
    if args.checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required for layerwise mode.")


def _validate_crossmodel_args(args):
    if args.models is None or args.checkpoint_dirs is None:
        raise ValueError(
            "--models and --checkpoint-dirs are required for crossmodel mode."
        )
    if len(args.models) != len(args.checkpoint_dirs):
        raise ValueError(
            f"Number of --models ({len(args.models)}) must match "
            f"--checkpoint-dirs ({len(args.checkpoint_dirs)})."
        )
    if len(args.models) < 2:
        raise ValueError("Cross-model CKA requires at least 2 models.")


def _get_model_repo(model_name: str) -> str:
    if model_name in MODEL_REPOS:
        return MODEL_REPOS[model_name]
    if model_name in ("biomedclip", "medimageinsight"):
        return model_name
    raise ValueError(f"Unknown model: {model_name}")


def _determine_num_classes(task: str, dataset: RadImageClassificationDataset) -> int:
    if task == "binary":
        return 1
    return len(set(dataset.labels))


def _create_test_dataloader(args, accelerator: Accelerator, model_name: str):
    """Create a test dataloader for CKA evaluation."""
    data_config, _ = setup_configs(args.data, args.task)
    data_root_folder = data_config.get_data_root_folder(use_multi_view=False)
    _, eval_transforms = get_transforms(model_name)

    test_loader = create_test_loader(
        data_root_folder=data_root_folder,
        task=args.task,
        batch_size=args.batch_size,
        test_transforms=eval_transforms,
        multi_view=False,
    )
    return test_loader


def run_layerwise(args, accelerator: Accelerator) -> None:
    """Run layerwise CKA: pretrained backbone vs fine-tuned model."""
    _validate_layerwise_args(args)
    device = accelerator.device

    # Build pretrained backbone (frozen, dummy classifier head)
    pretrained_model = build_backbone_model(args, device)

    # Load the fine-tuned model from checkpoint
    test_loader = _create_test_dataloader(args, accelerator, args.model)
    num_classes = _determine_num_classes(args.task, test_loader.dataset)
    model_repo = _get_model_repo(args.model)

    finetuned_wrapper = load_model(
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model,
        model_repo=model_repo,
        num_classes=num_classes,
        accelerator=accelerator,
        show_attention=False,
        multi_view=False,
        medimageinsight_path=getattr(args, "medimageinsight_path", None),
    )
    finetuned_model = finetuned_wrapper.model
    finetuned_model.eval()

    # Compute layerwise CKA
    cka_matrix, layer_names = compute_layerwise_cka(
        pretrained_model, finetuned_model, test_loader, device,
    )

    # Save results
    output_dir = os.path.join(
        args.output_path, args.data,
        f"cka_layerwise_{CURR_TIME}_{args.data}_{args.model}",
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "cka_matrix.npy"), cka_matrix)

        plot_layerwise_cka(
            cka_matrix, layer_names,
            model_name=args.model,
            dataset_name=args.data,
            save_path=os.path.join(output_dir, "cka_layerwise.png"),
        )

    logger.info("Layerwise CKA complete. Results in %s", output_dir)


def run_crossmodel(args, accelerator: Accelerator) -> None:
    """Run cross-model CKA: last-layer features across multiple FMs."""
    _validate_crossmodel_args(args)
    device = accelerator.device

    features_dict = {}

    for model_name, ckpt_dir in zip(args.models, args.checkpoint_dirs):
        logger.info("Extracting features for %s from %s", model_name, ckpt_dir)

        test_loader = _create_test_dataloader(args, accelerator, model_name)
        num_classes = _determine_num_classes(args.task, test_loader.dataset)
        model_repo = _get_model_repo(model_name)

        model_wrapper = load_model(
            checkpoint_dir=ckpt_dir,
            model_name=model_name,
            model_repo=model_repo,
            num_classes=num_classes,
            accelerator=accelerator,
            show_attention=False,
            multi_view=False,
            medimageinsight_path=getattr(args, "medimageinsight_path", None),
        )
        model = model_wrapper.model
        model.eval()

        feats, _ = extract_features(model, test_loader, device, normalize=True)
        features_dict[model_name] = feats
        logger.info("  %s features: %s", model_name, feats.shape)

        # Free GPU memory before loading next model
        del model, model_wrapper
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Compute pairwise CKA
    cka_matrix, model_names = compute_crossmodel_cka(features_dict)

    # Save results
    output_dir = os.path.join(
        args.output_path, args.data,
        f"cka_crossmodel_{CURR_TIME}_{args.data}",
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "cka_matrix.npy"), cka_matrix)

        plot_crossmodel_cka(
            cka_matrix, model_names,
            dataset_name=args.data,
            save_path=os.path.join(output_dir, "cka_crossmodel.png"),
        )

    logger.info("Cross-model CKA complete. Results in %s", output_dir)


def main():
    args = get_args_parser().parse_args()

    accelerator = Accelerator(
        mixed_precision="fp16" if args.optimize_compute else "no"
    )

    if args.mode == "layerwise":
        run_layerwise(args, accelerator)
    elif args.mode == "crossmodel":
        run_crossmodel(args, accelerator)


if __name__ == "__main__":
    main()
