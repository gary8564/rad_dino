"""Inference entry point: evaluate a fine-tuned checkpoint and generate explainability visualizations."""

import gc
import os
import argparse
import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import logging
from typing import List, Any, Optional
from dotenv import load_dotenv, find_dotenv

from rad_dino.data.dataset import RadImageClassificationDataset
from rad_dino.data.data_loader import create_test_loader
from rad_dino.utils.transforms import get_transforms
from rad_dino.configs.config import InferenceConfig, OutputPaths
from rad_dino.utils.config_utils import setup_configs, MODEL_REPOS
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.model_loader import load_model
from rad_dino.eval.inference_engine import InferenceEngine
from rad_dino.eval.explainable_visualizer import ExplainableVisualizer
from rad_dino.eval.evaluation_processor import EvaluationProcessor
from rad_dino.data.label_mapping import class_labels_mapping

load_dotenv(find_dotenv())

init_logging()
logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MAX_VISUALIZATION_SAMPLES = 24
DEFAULT_MIN_POSITIVE_VISUALIZATION_LABELS = 20
DEFAULT_MEDIMAGEINSIGHT_PATH = os.path.normpath(os.path.join(CURR_DIR, "..", "models", "MedImageInsights"))
NEGATIVE_LABEL_HINTS = {
    "negative", "neg", "normal", "no finding",
    "no findings", "none", "healthy", "background",
}

def get_args_parser() -> argparse.ArgumentParser:
    """Create argument parser for inference script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, 
                       choices=['multilabel', 'multiclass', 'binary'])
    parser.add_argument('--data', type=str, required=True, 
                       choices=['VinDr-CXR', 'TAIX-Ray', 'RSNA-Pneumonia', 'VinDr-Mammo', 'NODE21', 'COVID-CXR', 'VinDr-PCXR', 'VinDr-SpineXR', 'TBX11K', 'SIIM-ACR'])
    parser.add_argument('--model', type=str, required=True, 
                       choices=['rad-dino', 'dinov2-small', 'dinov2-base', 'dinov2-large', 'dinov2-large-reg', 'dinov3-small-plus', 'dinov3-base', 'dinov3-large', 'medsiglip', 'ark', 'medimageinsight', 'biomedclip']) 
    parser.add_argument('--model-path', required=True, type=str)
    parser.add_argument('--medimageinsight-path', type=str, default=DEFAULT_MEDIMAGEINSIGHT_PATH,
                       help="Path to the cloned lion-ai/MedImageInsights repository (default: rad_dino/models/MedImageInsights/).")
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--multi-view', action='store_true', 
                       help="Enable multi-view processing for mammography data")
    parser.add_argument("--optimize-compute", action="store_true",
                       help="Whether to use advanced tricks to lessen the heavy computational resource.")
    parser.add_argument('--show-attention', action='store_true')
    parser.add_argument('--show-gradcam', action='store_true')
    parser.add_argument('--attention-threshold', type=float, default=None, 
                       help="Threshold for attention masking")
    parser.add_argument('--save-heads', type=str, default=None,
                       help="Which attention heads to save: 'mean', 'max', 'min' (default: 'mean')")
    parser.add_argument('--compute-rollout', action='store_true', 
                       help="Enable attention rollout computation in addition to raw attention maps")
    parser.add_argument('--compute-gradient-rollout', action='store_true',
                       help="Enable gradient attention rollout for class-specific explainability. "
                            "Supported for DINO/ViT, BiomedCLIP, and MedSigLIP models.")
    parser.add_argument('--show-feature-maps', action='store_true',
                       help="Visualize stage-wise feature maps (DaViT paper Fig. 5 style). "
                            "Supported for Ark (Swin) and MedImageInsight (DaViT).")
    parser.add_argument('--max-visualization-samples', type=int, default=DEFAULT_MAX_VISUALIZATION_SAMPLES,
                       help="Maximum number of samples to run explainability visualizations on. "
                            "These samples are selected with a positive-first policy.")
    parser.add_argument('--min-positive-visualization-labels', type=int, default=DEFAULT_MIN_POSITIVE_VISUALIZATION_LABELS,
                       help="Minimum number of positive targets to cover when selecting visualization samples. "
                            "For binary/multiclass this is usually the number of positive samples.")
    parser.add_argument('--visualization-sample-ids', type=str, default=None,
                       help="Path to a text file with one sample ID per line. "
                            "When provided, only these samples are visualized — "
                            "this guarantees the same images across different model runs. "
                            "Tip: use the auto-generated visualization_selection.txt from "
                            "a previous run.")
    parser.add_argument('--compile', action='store_true',
                       help="Compile the model with torch.compile for faster inference. "
                            "Checkpoints are compatible whether this flag is on or off.")
    return parser

def validate_args(config: InferenceConfig) -> None:
    """Validate command line arguments"""
    if config.multi_view and config.data != 'VinDr-Mammo':
        raise ValueError("Multi-view processing is only supported for VinDr-Mammo dataset")
    
    # MedImageInsight (DaViT) and Ark (Swin) do not support attention/rollout
    # explainability.  Swin's hierarchical windowed attention cannot be stitched
    # into a meaningful global attention map -- the original Swin paper uses
    # GradCAM instead.  Only GradCAM and feature maps are allowed for these models.
    if config.model in ("medimageinsight", "ark"):
        skipped = []
        if config.show_attention:
            skipped.append("--show-attention")
            config.show_attention = False
        if config.compute_rollout:
            skipped.append("--compute-rollout")
            config.compute_rollout = False
        if config.compute_gradient_rollout:
            skipped.append("--compute-gradient-rollout")
            config.compute_gradient_rollout = False
        if skipped:
            arch = "DaViT (dual attention)" if config.model == "medimageinsight" else "Swin (hierarchical windowed attention)"
            logger.warning(
                "%s does not support %s. These flags will be ignored. "
                "Use --show-gradcam or --show-feature-maps instead.",
                arch, ", ".join(skipped),
            )

    if (config.save_heads is None or config.attention_threshold is None) and config.show_attention:
        raise ValueError("Attention visualization requires specifying save_heads and attention_threshold")
    
    if (config.save_heads is not None and config.attention_threshold is not None) and not config.show_attention:
        logger.warning("Attention visualization is not enabled, but save_heads and attention_threshold are specified.")

    if config.max_visualization_samples <= 0:
        raise ValueError("--max-visualization-samples must be > 0")
    if config.min_positive_visualization_labels < 0:
        raise ValueError("--min-positive-visualization-labels must be >= 0")


def _normalize_label_name(label: object) -> str:
    return str(label).strip().lower().replace("-", " ").replace("_", " ")


def _is_negative_label_name(label: object) -> bool:
    normalized = _normalize_label_name(label)
    return normalized in NEGATIVE_LABEL_HINTS or normalized.startswith("birads 1") or normalized.startswith("bi rads 1")


def _compute_positive_target_counts(
    dataset: RadImageClassificationDataset,
    config: InferenceConfig,
    class_labels: Optional[List],
) -> np.ndarray:
    """Compute how many explainability-relevant positive targets each sample contains."""
    if config.task == "binary":
        return dataset.df["label"].to_numpy(dtype=np.float32).astype(np.int32)

    if config.task == "multilabel":
        positive_columns = [col for col in dataset.df.columns if not _is_negative_label_name(col)]
        if not positive_columns:
            positive_columns = list(dataset.df.columns)
        return dataset.df[positive_columns].to_numpy(dtype=np.float32).sum(axis=1).astype(np.int32)

    raw_labels = sorted(set(dataset.df["label"].tolist()))
    negative_raw_labels = set()
    if class_labels is not None:
        for idx, label_name in enumerate(class_labels):
            if idx < len(raw_labels) and _is_negative_label_name(label_name):
                negative_raw_labels.add(raw_labels[idx])
    if not negative_raw_labels and raw_labels:
        negative_raw_labels.add(raw_labels[0])
    return (~dataset.df["label"].isin(negative_raw_labels)).to_numpy(dtype=np.int32)


def _load_visualization_sample_ids_from_file(
    filepath: str,
    dataset: RadImageClassificationDataset,
) -> List[str]:
    """
    Load explicit sample IDs from a text file (one ID per line).

    Unknown IDs that are not present in the dataset are silently dropped
    so that the file can be shared across datasets of different sizes.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_ids = [line.strip() for line in f if line.strip()]

    dataset_id_set = set(dataset.sample_ids)
    valid_ids = [sid for sid in raw_ids if sid in dataset_id_set]
    skipped = len(raw_ids) - len(valid_ids)
    if skipped:
        logger.warning(
            "Dropped %d sample IDs from %s that are not in the current dataset.",
            skipped, filepath,
        )
    if not valid_ids:
        raise ValueError(
            f"None of the {len(raw_ids)} sample IDs in {filepath} exist in the dataset. "
            "Check that the file matches the current dataset split."
        )
    logger.info(
        "Loaded %d visualization sample IDs from %s", len(valid_ids), filepath,
    )
    return valid_ids


def _select_visualization_sample_ids(
    dataset: RadImageClassificationDataset,
    config: InferenceConfig,
    class_labels: Optional[List],
) -> List[str]:
    """Pick a small positive-first subset for explainability to reduce memory load.

    When ``config.visualization_sample_ids`` is specified as a path to a text file, its contents
    take priority so that the same images can be reused across model runs.
    """
    if config.visualization_sample_ids is not None:
        return _load_visualization_sample_ids_from_file(
            config.visualization_sample_ids, dataset,
        )

    positive_counts = _compute_positive_target_counts(dataset, config, class_labels)
    sample_ids = dataset.sample_ids
    effective_max_samples = max(
        config.max_visualization_samples,
        config.min_positive_visualization_labels,
    )

    ranked_indices = sorted(
        range(len(sample_ids)),
        key=lambda idx: (-int(positive_counts[idx]), idx),
    )

    selected_sample_ids: List[str] = []
    accumulated_positive_targets = 0
    for idx in ranked_indices:
        positive_count = int(positive_counts[idx])
        if positive_count <= 0:
            break
        selected_sample_ids.append(sample_ids[idx])
        accumulated_positive_targets += positive_count
        if (
            accumulated_positive_targets >= config.min_positive_visualization_labels
            or len(selected_sample_ids) >= effective_max_samples
        ):
            break

    if not selected_sample_ids:
        fallback_count = min(config.max_visualization_samples, len(sample_ids))
        selected_sample_ids = sample_ids[:fallback_count]
        logger.warning(
            "No positive samples were found for explainability selection. "
            "Falling back to the first %d samples.",
            len(selected_sample_ids),
        )
    else:
        logger.info(
            "Selected %d samples for explainability, covering %d positive targets.",
            len(selected_sample_ids),
            accumulated_positive_targets,
        )
        if accumulated_positive_targets < config.min_positive_visualization_labels:
            logger.warning(
                "Could only cover %d positive targets, below the requested minimum of %d.",
                accumulated_positive_targets,
                config.min_positive_visualization_labels,
            )

    return selected_sample_ids


def _log_visualization_selection(
    output_paths: OutputPaths,
    accelerator: Accelerator,
    selected_sample_ids: List[str],
) -> None:
    """Persist the selected explainability subset for reproducibility."""
    if not accelerator.is_main_process or not selected_sample_ids:
        return

    selection_path = os.path.join(output_paths.table, "visualization_selection.txt")
    with open(selection_path, "w", encoding="utf-8") as file:
        for sample_id in selected_sample_ids:
            file.write(f"{sample_id}\n")

def create_output_directories(output_dir: str, accelerator: Accelerator, config: InferenceConfig) -> OutputPaths:
    """Create output directories and return paths"""
    if accelerator.is_main_process:
        os.makedirs(f"{output_dir}/figs", exist_ok=True)
        os.makedirs(f"{output_dir}/table", exist_ok=True)
        
        # Only create visualization directories if needed
        gradcam_path = None
        attention_path = None
        feature_maps_path = None
        
        if config.show_gradcam:
            os.makedirs(f"{output_dir}/gradcam", exist_ok=True)
            gradcam_path = f"{output_dir}/gradcam"
            
        if config.show_attention:
            os.makedirs(f"{output_dir}/attention", exist_ok=True)
            attention_path = f"{output_dir}/attention"

        if config.show_feature_maps:
            os.makedirs(f"{output_dir}/feature_maps", exist_ok=True)
            feature_maps_path = f"{output_dir}/feature_maps"
    
    return OutputPaths(
        base=output_dir,
        figs=f"{output_dir}/figs",
        table=f"{output_dir}/table",
        gradcam=gradcam_path,
        attention=attention_path,
        feature_maps=feature_maps_path,
    )

def determine_class_info(config: InferenceConfig, dataset: RadImageClassificationDataset) -> tuple[List, int]:
    """Determine class labels and number of classes based on different classification tasks.
   
    Args:
        config: Inference configuration
        dataset: Dataset containing class information
        
    Returns:
        tuple: (class_labels, num_classes) - Class labels and number of classes
    """
    if config.task == "binary":
        class_labels = None
        num_classes = 1
    elif config.task == "multiclass":
        # Keep deterministic class-index order aligned with numeric dataset labels.
        raw_class_labels = sorted(set(dataset.labels))
        # Process class labels based on dataset-specific mappings
        class_labels = class_labels_mapping(config.data, raw_class_labels)
        num_classes = len(class_labels)
    else:  # multilabel
        class_labels = dataset.labels
        num_classes = len(class_labels)
    
    return class_labels, num_classes

def setup_data_loader(config: InferenceConfig, accelerator: Accelerator) -> tuple[RadImageClassificationDataset, DataLoader]:
    """Setup dataset and data loader
    Args:
        config: Inference configuration
        accelerator: Accelerator for distributed computing
        
    Returns:
        tuple: (Dataset, DataLoader)
    """
    # Setup data configs 
    data_config, _ = setup_configs(config.data, config.task)
    
    # Get data root folder from config
    data_root_folder = data_config.get_data_root_folder(config.multi_view)

    # Setup transforms
    _, test_transforms = get_transforms(config.model)

    # Create test dataset and data loader
    test_loader = create_test_loader(
        data_root_folder=data_root_folder,
        task=config.task,
        batch_size=config.batch_size,
        test_transforms=test_transforms,
        multi_view=config.multi_view
    )
    test_ds = test_loader.dataset
    test_loader = accelerator.prepare(test_loader)
    return test_ds, test_loader

def setup_model(config: InferenceConfig, repo: str, num_classes: int, 
                              accelerator: Accelerator) -> Any:
    """Setup model"""
    # Load model
    model_wrapper = load_model(config.model_path, config.model, repo, num_classes, accelerator, 
                              config.show_attention, config.multi_view,
                              medimageinsight_path=config.medimageinsight_path)

    # In-place torch.compile: compiles the forward pass without
    # changing the module structure or state_dict keys.
    if config.compile:
        logger.info("Compiling model with torch.compile (in-place, backend='inductor')")
        model_wrapper.model.compile(backend="inductor")
    
    return model_wrapper

def run_inference(model_wrapper, 
                  dataset: RadImageClassificationDataset,
                  loader: DataLoader,
                  accelerator: Accelerator, 
                  config: InferenceConfig, 
                  output_paths: OutputPaths,
                  model_repo: str, 
                  class_labels: List, 
                  num_classes: int) -> None:
    """Main inference function
    
    Args:
        model_wrapper: Model wrapper containing the trained model
        loader: DataLoader for test data
        accelerator: Accelerator for distributed training
        config: Inference configuration
        output_paths: Paths for saving outputs
        model_repo: Model repository name for image processor
        class_labels: List of class labels
        num_classes: Number of classes
    """
    
    # Initialize InferenceEngine and EvaluationProcessor for prediction and saving evaluation metrics
    inference_engine = InferenceEngine(model_wrapper, accelerator, config.show_attention)
    evaluation_processor = EvaluationProcessor(
        accelerator, output_paths, config.task, class_labels
    )
    
    # Initialize ExplainableVisualizer for visualization (only if any visualization flag is enabled)
    explainable_visualizer = None
    image_processor = None
    any_vis_enabled = (config.show_attention or config.show_gradcam
                       or config.show_feature_maps or config.compute_gradient_rollout)
    if any_vis_enabled:
        needs_image_proc = config.show_attention or config.show_gradcam or config.compute_gradient_rollout
        if model_repo is not None:
            image_processor = AutoImageProcessor.from_pretrained(model_repo)
        elif needs_image_proc:
            image_processor = None
        explainable_visualizer = ExplainableVisualizer(
            accelerator, output_paths, model_wrapper, image_processor,
            config.show_attention, config.show_gradcam,
            config.show_feature_maps, config.compute_gradient_rollout,
        )
    
    # Validate rollout computation
    if config.compute_rollout and not config.show_attention:
        raise ValueError("Attention rollout computation is only supported when attention visualization is enabled.")

    selected_sample_ids = _select_visualization_sample_ids(dataset, config, class_labels) if any_vis_enabled else []
    pending_visualizations = set(selected_sample_ids)
    _log_visualization_selection(output_paths, accelerator, selected_sample_ids)
    
    for batch in tqdm(loader, desc="Inference", disable=not accelerator.is_main_process):
        images = batch["pixel_values"]
        targets = batch["labels"]
        image_ids = batch["sample_ids"]
        images = images.to(accelerator.device)

        need_attention = False
        if hasattr(model_wrapper.model, 'return_attentions'):
            model_wrapper.model.return_attentions = need_attention
        
        # Run inference
        logits, attentions, pooler_attn = inference_engine.run_inference(images, num_classes)
        
        # Run visualizations (only if enabled and visualizer is initialized)
        # Process results
        evaluation_processor.add_batch_results(image_ids, targets, logits)

        selected_batch_indices = [
            idx for idx, sample_id in enumerate(image_ids)
            if sample_id in pending_visualizations
        ]

        if explainable_visualizer is not None and selected_batch_indices:
            for sample_idx in selected_batch_indices:
                sample_id = image_ids[sample_idx]
                sample_images = images[sample_idx:sample_idx + 1]
                sample_ids = [sample_id]

                if config.show_gradcam:
                    explainable_visualizer.run_gradcam_visualization(
                        model_wrapper.model, sample_images, sample_ids, class_labels
                    )

                if config.show_attention:
                    if hasattr(model_wrapper.model, 'return_attentions'):
                        model_wrapper.model.return_attentions = True
                    _, sample_attentions, sample_pooler_attn = inference_engine.run_inference(
                        sample_images, num_classes
                    )
                    if hasattr(model_wrapper.model, 'return_attentions'):
                        model_wrapper.model.return_attentions = False

                    if sample_attentions is not None:
                        explainable_visualizer.run_attention_visualization(
                            sample_attentions, sample_images, sample_ids, model_wrapper.config,
                            config.attention_threshold, config.save_heads, config.compute_rollout,
                            pooler_attn_weights=sample_pooler_attn,
                        )
                    del sample_attentions, sample_pooler_attn

                if config.show_feature_maps:
                    explainable_visualizer.run_feature_map_visualization(
                        model_wrapper.model, sample_images, sample_ids
                    )

                if config.compute_gradient_rollout:
                    explainable_visualizer.run_gradient_rollout_visualization(
                        model_wrapper.model, sample_images, sample_ids, model_wrapper.config,
                    )

                pending_visualizations.discard(sample_id)
                gc.collect()
                if accelerator.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Free attention tensors before the next batch to avoid OOM
        # (attention maps for ViT at 518x518 can exceed 29 GB per batch)
        del attentions, pooler_attn
        gc.collect()
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    if explainable_visualizer is not None:
        explainable_visualizer.log_visualization_outputs()

    # Save results and compute metrics
    evaluation_processor.process_and_save_results()

def main():
    """Main function"""
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Validate medimageinsight-specific args
    if args.model == "medimageinsight" and not os.path.isdir(args.medimageinsight_path):
        raise ValueError(
            f"MedImageInsight repo not found at '{args.medimageinsight_path}'. "
            "Clone it first: git lfs install && git clone https://huggingface.co/lion-ai/MedImageInsights "
            f"{args.medimageinsight_path}"
        )

    # Create configuration object
    config = InferenceConfig(
        task=args.task,
        data=args.data,
        model=args.model,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        multi_view=args.multi_view,
        optimize_compute=args.optimize_compute,
        compile=args.compile,
        show_attention=args.show_attention,
        show_gradcam=args.show_gradcam,
        attention_threshold=args.attention_threshold,
        save_heads=args.save_heads,
        compute_rollout=args.compute_rollout,
        compute_gradient_rollout=args.compute_gradient_rollout,
        show_feature_maps=args.show_feature_maps,
        max_visualization_samples=args.max_visualization_samples,
        min_positive_visualization_labels=args.min_positive_visualization_labels,
        visualization_sample_ids=args.visualization_sample_ids,
        medimageinsight_path=args.medimageinsight_path
    )
    
    # Validate arguments
    validate_args(config)
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision="fp16" if config.optimize_compute else "no")
    
    # Get model repository (medimageinsight / biomedclip does not have corresponding HF repo for AutoModel/AutoImageProcessor)
    if config.model in ("medimageinsight", "biomedclip"):
        repo = None 
    elif config.model not in MODEL_REPOS:
        raise ValueError(f"Model {config.model} is not supported. Please choose from {list(MODEL_REPOS.keys())}.")
    else:
        repo = MODEL_REPOS[config.model]
    logger.info(f"Running inference with multi_view={config.multi_view}")
    
    # Setup data loader and dataset
    test_dataset, test_loader = setup_data_loader(config, accelerator)
    
    # Determine class information for model setup
    class_labels, num_classes = determine_class_info(config, test_dataset)
    
    # Setup model and validation
    model_wrapper = setup_model(config, repo, num_classes, accelerator)

    # Setup output directories
    modelname = config.model_path.rsplit('/', 1)[-1]
    output_path = os.path.join(config.output_path, config.data, modelname)
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)
    
    output_paths = create_output_directories(output_path, accelerator, config)
    
    # Run inference
    run_inference(
        model_wrapper,
        test_dataset,
        test_loader,
        accelerator,
        config,
        output_paths,
        repo,
        class_labels,
        num_classes
    )

if __name__ == "__main__":
    main() 