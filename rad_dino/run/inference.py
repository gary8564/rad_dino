import os
import argparse
import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import logging
from typing import List, Any
from dotenv import load_dotenv, find_dotenv

from rad_dino.data.dataset import RadImageClassificationDataset
from rad_dino.data.data_loader import create_test_loader
from rad_dino.utils.transforms import get_transforms
from rad_dino.configs.config import InferenceConfig, OutputPaths
from rad_dino.utils.config_utils import setup_configs
from rad_dino.loggings.setup import init_logging
from rad_dino.utils.model_loader import load_model
from rad_dino.eval.inference_engine import InferenceEngine
from rad_dino.eval.explainable_visualizer import ExplainableVisualizer
from rad_dino.eval.evaluation_processor import EvaluationProcessor
from rad_dino.data.label_mapping import class_labels_mapping

load_dotenv(find_dotenv())

init_logging()
logger = logging.getLogger(__name__)

# Constants
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
MAX_GRADCAM_IMAGES = 10
DEFAULT_MEDIMAGEINSIGHT_PATH = os.path.normpath(os.path.join(CURR_DIR, "..", "models", "MedImageInsights"))

# Model repository mapping
MODEL_REPOS = {
    "rad-dino": "microsoft/rad-dino",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-base": "facebook/dinov2-with-registers-base", #"facebook/dinov2-base", 
    "dinov2-small": "facebook/dinov2-small",
    "dinov3-large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3-base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3-small-plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "medsiglip": "google/medsiglip-448",
    "ark": "microsoft/swin-large-patch4-window12-384-in22k"
}

def get_args_parser() -> argparse.ArgumentParser:
    """Create argument parser for inference script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, 
                       choices=['multilabel', 'multiclass', 'binary'])
    parser.add_argument('--data', type=str, required=True, 
                       choices=['VinDr-CXR', 'TAIX-Ray', 'RSNA-Pneumonia', 'VinDr-Mammo', 'NODE21'])
    parser.add_argument('--model', type=str, required=True, 
                       choices=['rad-dino', 'dinov2-small', 'dinov2-base', 'dinov2-large', 'dinov3-small-plus', 'dinov3-base', 'dinov3-large', 'medsiglip', 'ark', 'medimageinsight', 'biomedclip']) 
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
    parser.add_argument('--show-lrp', action='store_true')
    parser.add_argument('--show-gradcam', action='store_true')
    parser.add_argument('--attention-threshold', type=float, default=None, 
                       help="Threshold for attention masking")
    parser.add_argument('--save-heads', type=str, default=None,
                       help="Which attention heads to save: 'mean', 'max', 'min' (default: 'mean')")
    parser.add_argument('--compute-rollout', action='store_true', 
                       help="Enable attention rollout computation in addition to raw attention maps")
    parser.add_argument('--compile', action='store_true',
                       help="Compile the model with torch.compile for faster inference. "
                            "Checkpoints are compatible whether this flag is on or off.")
    return parser

def validate_args(config: InferenceConfig) -> None:
    """Validate command line arguments"""
    if config.multi_view and config.data != 'VinDr-Mammo':
        raise ValueError("Multi-view processing is only supported for VinDr-Mammo dataset")
    
    if (config.save_heads is None or config.attention_threshold is None) and config.show_attention:
        raise ValueError("Attention visualization requires specifying save_heads and attention_threshold")
    
    if (config.save_heads is not None and config.attention_threshold is not None) and not config.show_attention:
        logger.warning("Attention visualization is not enabled, but save_heads and attention_threshold are specified.")

def create_output_directories(output_dir: str, accelerator: Accelerator, config: InferenceConfig) -> OutputPaths:
    """Create output directories and return paths"""
    if accelerator.is_main_process:
        os.makedirs(f"{output_dir}/figs", exist_ok=True)
        os.makedirs(f"{output_dir}/table", exist_ok=True)
        
        # Only create visualization directories if needed
        gradcam_path = None
        attention_path = None
        lrp_path = None
        
        if config.show_gradcam:
            os.makedirs(f"{output_dir}/gradcam", exist_ok=True)
            gradcam_path = f"{output_dir}/gradcam"
            
        if config.show_attention:
            os.makedirs(f"{output_dir}/attention", exist_ok=True)
            attention_path = f"{output_dir}/attention"
            
        if config.show_lrp:
            os.makedirs(f"{output_dir}/lrp", exist_ok=True)
            lrp_path = f"{output_dir}/lrp"
    
    return OutputPaths(
        base=output_dir,
        figs=f"{output_dir}/figs",
        table=f"{output_dir}/table",
        gradcam=gradcam_path,
        attention=attention_path,
        lrp=lrp_path
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
        raw_class_labels = list(set(dataset.labels))
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
    if config.show_attention or config.show_gradcam or config.show_lrp:
        if model_repo is None:
            raise ValueError(
                f"Explainability visualizations (attention/GradCAM/LRP) are not supported for model '{config.model}'."
            )
        image_processor = AutoImageProcessor.from_pretrained(model_repo)
        explainable_visualizer = ExplainableVisualizer(
            accelerator, output_paths, model_wrapper, image_processor,
            config.show_attention, config.show_gradcam, config.show_lrp
        )
    
    # Validate rollout computation
    if config.compute_rollout and not config.show_attention:
        raise ValueError("Attention rollout computation is only supported when attention visualization is enabled.")
    
    # GradCAM visualization counter
    gradcam_count = 0
    
    for batch in tqdm(loader, desc="Inference", disable=not accelerator.is_main_process):
        images = batch["pixel_values"]
        targets = batch["labels"]
        image_ids = batch["sample_ids"]
        images = images.to(accelerator.device)
        
        # Clear CUDA cache before inference
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Run inference
        logits, attentions, pooler_attn = inference_engine.run_inference(images, num_classes)
        
        # Run visualizations (only if enabled and visualizer is initialized)
        if explainable_visualizer is not None:
            # GradCAM visualization (limited to MAX_GRADCAM_IMAGES)
            compute_gradcam = gradcam_count < MAX_GRADCAM_IMAGES and config.show_gradcam
            if compute_gradcam:
                explainable_visualizer.run_gradcam_visualization(
                    model_wrapper.model, images, image_ids, class_labels
                )
            gradcam_count += 1
            
            # Attention visualization
            if config.show_attention and attentions is not None:
                explainable_visualizer.run_attention_visualization(
                    attentions, images, image_ids, model_wrapper.config,
                    config.attention_threshold, config.save_heads, config.compute_rollout,
                    pooler_attn_weights=pooler_attn
                )
            
            # LRP visualization
            if config.show_lrp:
                explainable_visualizer.run_lrp_visualization(
                    model_wrapper.model, images, image_ids, model_wrapper.multi_view
                )
            
            # Clear CUDA cache after visualizations
            if accelerator.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Log visualization outputs 
            explainable_visualizer.log_visualization_outputs()
        
        # Process results
        evaluation_processor.add_batch_results(image_ids, targets, logits)
    
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
        show_lrp=args.show_lrp,
        show_gradcam=args.show_gradcam,
        attention_threshold=args.attention_threshold,
        save_heads=args.save_heads,
        compute_rollout=args.compute_rollout,
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