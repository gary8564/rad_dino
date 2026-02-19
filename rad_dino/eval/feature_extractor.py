"""
Feature extraction evaluation (KNN, Linear SVM).
"""

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm

from rad_dino.configs.config import OutputPaths
from rad_dino.data.data_loader import create_test_loader
from rad_dino.data.dataset import RadImageClassificationDataset
from rad_dino.data.label_mapping import class_labels_mapping
from rad_dino.eval.evaluation_processor import EvaluationProcessor
from rad_dino.models.ark import ArkClassifier, load_prtrained_ark_model
from rad_dino.models.base import BaseClassifier
from rad_dino.models.biomedclip import BiomedCLIPClassifier, load_biomedclip_model
from rad_dino.models.dino import DinoClassifier
from rad_dino.models.medimageinsight import (
    MedImageInsightClassifier,
    load_medimageinsight_model,
)
from rad_dino.models.siglip import MedSigClassifier
from rad_dino.utils.config_utils import setup_configs
from rad_dino.utils.data_utils import collate_fn
from rad_dino.utils.model_loader import load_pretrained_model
from rad_dino.utils.transforms import get_transforms

logger = logging.getLogger(__name__)

MODEL_REPOS = {
    "rad-dino": "microsoft/rad-dino",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-large": "facebook/dinov2-large",
    "dinov3-small-plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "medsiglip": "google/medsiglip-448",
    "ark": "microsoft/swin-large-patch4-window12-384-in22k",
}

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MEDIMAGEINSIGHT_PATH = os.path.normpath(
    os.path.join(CURR_DIR, "..", "models", "MedImageInsights")
)


# Feature extraction
@torch.no_grad()
def extract_features(
    model: BaseClassifier,
    loader: DataLoader,
    device: torch.device,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract backbone features and labels from a DataLoader.

    For multi-view models the per-view features are averaged (mean fusion).
    Only mean fusion is supported because weighted-mean and MLP-adapter
    fusions have learned parameters that require training to optimize.

    Args:
        model: A ``BaseClassifier`` subclass.
        loader: DataLoader.
        device: Torch device.
        normalize: If True, L2-normalise feature vectors.

    Returns:
        Tuple of (features, labels) with shapes [N, embed_dim] and [N, ...] respectively.

    Raises:
        ValueError: If the model uses a learned multi-view fusion strategy (``weighted_mean`` or ``mlp_adapter``).
    """
    if model.multi_view:
        fusion_type = getattr(model, "view_fusion_type", None)
        if fusion_type not in (None, "mean"):
            raise ValueError(
                f"Multi-view feature extraction only supports 'mean' fusion "
                f"for KNN/SVM (no learned parameters), but got '{fusion_type}'. "
                f"Weighted-mean and MLP-adapter fusions require training and "
                f"are not meaningful with frozen feature evaluation."
            )

    model.eval()
    all_features = []
    all_labels = []

    for batch in tqdm(loader, desc="Extracting features", leave=False):
        images = batch["pixel_values"].to(device)
        labels = batch["labels"]

        batch_size = images.shape[0]

        # Input reshape (handles multi-view [B, V, C, H, W] -> [B*V, C, H, W])
        x, _ = model.input_reshape_strategies[model.multi_view](images)
        features, _ = model.extract_features(x)

        # Multi-view mean fusion
        if model.multi_view:
            features = features.view(
                batch_size, model.num_views, -1
            ).mean(dim=1)  # [B, D]

        if normalize:
            features = F.normalize(features, p=2, dim=1)

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_features, axis=0), np.concatenate(all_labels, axis=0)


# KNN classification
def knn_classify(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    nb_knn: int,
    temperature: float,
    num_classes: int,
) -> torch.Tensor:
    """
    DINOv2-style temperature-weighted KNN classification.

    1. L2-normalise features.
    2. Compute cosine similarity.
    3. Select top-K neighbours.
    4. Apply temperature-scaled softmax over similarities.
    5. Weighted one-hot voting to obtain class probabilities.

    Args:
        train_features: [N_train, D] training features.
        train_labels: [N_train] integer class labels.
        test_features: [N_test, D] test features.
        nb_knn: Number of nearest neighbours (K).
        temperature: Softmax temperature (DINOv2 default: 0.07).
        num_classes: Total number of classes.

    Returns:
        Class probability tensor with shape [N_test, num_classes].

    References:
        Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/knn.py
    """
    # L2-normalize features
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    # Compute cosine similarity
    similarities = test_features @ train_features.T # [N_test, N_train]
    
    # Select top-K nearest neighbours
    topk_sims, topk_indices = similarities.topk(nb_knn, dim=1) # [N_test, K]
    topk_labels = train_labels[topk_indices] # [N_test, K]
    
    # Apply temperature-scaled softmax over similarities
    weights = F.softmax(topk_sims / temperature, dim=1) # [N_test, K]
    one_hot = F.one_hot(topk_labels, num_classes).float() # [N_test, K, num_classes]
    probas = (one_hot * weights.unsqueeze(-1)).sum(dim=1) # [N_test, num_classes]

    return probas


# Build backbone model
def build_backbone_model(args, device: torch.device) -> BaseClassifier:
    """
    Build a ``BaseClassifier`` with a frozen pretrained backbone.

    Only ``extract_features()`` is used — the classification head is a dummy
    (``num_classes=1``) and is never called.  For multi-view, only mean
    fusion is used (no learned parameters).
    """
    multi_view_config = None
    if args.multi_view:
        data_config, _ = setup_configs(args.data, args.task)
        multi_view_config = data_config.get_multi_view_config(args.multi_view)

    mv_kwargs = dict(
        multi_view=args.multi_view,
        num_views=multi_view_config.num_views if multi_view_config else None,
        view_fusion_type="mean" if args.multi_view else None,
    )

    if args.model == "biomedclip":
        backbone, _ = load_biomedclip_model(device=str(device))
        model = BiomedCLIPClassifier(backbone, num_classes=1, **mv_kwargs)
    elif args.model == "medimageinsight":
        mii_path = getattr(args, "medimageinsight_path", DEFAULT_MEDIMAGEINSIGHT_PATH)
        backbone = load_medimageinsight_model(mii_path, device=str(device))
        model = MedImageInsightClassifier(backbone, num_classes=1, **mv_kwargs)
    elif args.model == "ark":
        if getattr(args, "pretrained_ark_path", None) is None:
            raise ValueError(
                "Ark requires --pretrained-ark-path for the pre-trained checkpoint."
            )
        backbone = load_prtrained_ark_model(
            checkpoint_path=args.pretrained_ark_path,
            num_classes_list=[14, 14, 14, 3, 6, 1],
            img_size=768, patch_size=4, window_size=12,
            embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
            projector_features=1376, use_mlp=False,
            return_attention=False, grad_checkpointing=False,
            device=device,
        )
        model = ArkClassifier(
            backbone, num_classes=1, use_backbone_projector=True, **mv_kwargs
        )
    elif args.model == "medsiglip":
        backbone = load_pretrained_model(MODEL_REPOS[args.model])
        model = MedSigClassifier(backbone, num_classes=1, **mv_kwargs)
    else:
        backbone = load_pretrained_model(MODEL_REPOS[args.model])
        model = DinoClassifier(backbone, num_classes=1, **mv_kwargs)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    logger.info(f"Built backbone: {args.model} (embed_dim={model.embed_dim})")
    return model


# Arg parser utilities
def add_args(parser) -> None:
    """Add CLI arguments."""
    parser.add_argument(
        "--task", type=str, required=True,
        choices=["multiclass", "binary"],
        help="Classification task. Only binary and multiclass are supported "
             "for KNN/SVM evaluation (multi-label is not supported).",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        choices=[
            "TBX11", "SIIM-ACR",
            "VinDr-Mammo", "NODE21"
        ],
        help="Dataset to evaluate on. Only binary and multiclass are supported "
             "for KNN/SVM evaluation (multi-label is not supported).",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=[
            "rad-dino", "dinov2-large", "dinov3-large",
            "medsiglip", "ark", "medimageinsight", "biomedclip",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument(
        "--multi-view", action="store_true",
        help="Enable multi-view processing for mammography data.",
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
        help="Path to cloned MedImageInsights repository.",
    )


def validate_args(args) -> None:
    """Validate CLI arguments."""
    if args.task == "multilabel":
        raise ValueError(
            "KNN and SVM evaluation only support binary and multiclass tasks. "
            "Multi-label classification is not supported because KNN uses "
            "single-label one-hot voting and LinearSVC is not designed for "
            "multi-label outputs. Use linear probing or fine-tuning instead."
        )
    if args.multi_view and args.data != "VinDr-Mammo":
        raise ValueError("Multi-view is only supported for VinDr-Mammo.")
    if args.model == "ark" and getattr(args, "pretrained_ark_path", None) is None:
        raise ValueError("Ark requires --pretrained-ark-path.")
    if args.model == "medimageinsight":
        mii_path = getattr(args, "medimageinsight_path", DEFAULT_MEDIMAGEINSIGHT_PATH)
        if not os.path.isdir(mii_path):
            raise ValueError(
                f"MedImageInsight repo not found at '{mii_path}'."
            )

# Data loading and feature extraction
def setup_data_and_features(args, accelerator: Accelerator, model: BaseClassifier):
    """
    Load train/test data, extract features, and determine class info.

    Returns:
        ``(train_features, train_labels, test_features, test_labels,
          num_classes, class_labels)``
    """
    device = accelerator.device
    data_config, _ = setup_configs(args.data, args.task)
    data_root_folder = data_config.get_data_root_folder(args.multi_view)
    _, eval_transforms = get_transforms(args.model)

    # Train loader (eval transforms - no augmentation for feature extraction)
    train_ds = RadImageClassificationDataset(
        data_root_folder, "train", args.task,
        transform=eval_transforms, multi_view=args.multi_view,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=data_config.num_workers, pin_memory=True,
        drop_last=False, collate_fn=collate_fn,
    )

    # Test loader
    test_loader = create_test_loader(
        data_root_folder=data_root_folder,
        task=args.task,
        batch_size=args.batch_size,
        test_transforms=eval_transforms,
        multi_view=args.multi_view,
    )
    test_ds = test_loader.dataset

    # Class info (only binary and multiclass supported)
    if args.task == "binary":
        num_classes = 2
        class_labels = None
    else:
        raw_class_labels = list(set(test_ds.labels))
        class_labels = class_labels_mapping(args.data, raw_class_labels)
        num_classes = len(class_labels)

    # Extract features
    logger.info("Extracting train features...")
    train_features, train_labels = extract_features(model, train_loader, device)
    logger.info(f"Train features: {train_features.shape} (labels: {train_labels.shape})")

    logger.info("Extracting test features...")
    test_features, test_labels = extract_features(model, test_loader, device)
    logger.info(f"Test features: {test_features.shape} (labels: {test_labels.shape})")

    return train_features, train_labels, test_features, test_labels, num_classes, class_labels


# Save results
def save_results(
    probas: np.ndarray,
    test_labels: np.ndarray,
    task: str,
    class_labels: Optional[List],
    output_dir: str,
    accelerator: Accelerator,
) -> None:
    """Save predictions and metrics using ``EvaluationProcessor``."""
    os.makedirs(os.path.join(output_dir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "table"), exist_ok=True)

    output_paths = OutputPaths(
        base=output_dir,
        figs=os.path.join(output_dir, "figs"),
        table=os.path.join(output_dir, "table"),
    )

    eval_proc = EvaluationProcessor(accelerator, output_paths, task, class_labels)

    n = probas.shape[0]
    image_ids = [str(i) for i in range(n)]
    probs_tensor = torch.from_numpy(probas).float()
    labels_tensor = torch.from_numpy(test_labels).float()
    dummy_logits = torch.zeros_like(probs_tensor)

    eval_proc.add_batch_results(image_ids, labels_tensor, dummy_logits, probs=probs_tensor)
    metrics = eval_proc.process_and_save_results()

    logger.info(f"Results saved to {output_dir}")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")
