import os
import onnxruntime
import argparse
import torch
import yaml
import json
import pandas as pd
import numpy as np
from accelerate import Accelerator
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoImageProcessor
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from rad_dino.data.dataset import RadImageClassificationDataset
from rad_dino.utils.data_utils import get_transforms, collate_fn
from rad_dino.utils.visualization.visualize_gradcam import visualize_gradcam
from rad_dino.utils.metrics.compute_metrics import compute_evaluation_metrics
from rad_dino.utils.visualization.visualize_attention import visualize_attention_maps
from rad_dino.utils.visualization.visualize_lrp import visualize_lrp_maps
from rad_dino.utils.model_loader import _load_best_ark_model, _load_best_medsig_model, _load_best_dino_model
from rad_dino.models.dino import DinoClassifier
from rad_dino.models.siglip import MedSigClassifier
from rad_dino.models.ark import ArkClassifier
from rad_dino.loggings.setup import init_logging
from rad_dino.configs.config import InferenceConfig, ModelWrapper, OutputPaths

init_logging()
logger = logging.getLogger(__name__)

# Constants
CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")
MAX_GRADCAM_IMAGES = 10
DEFAULT_PATCH_SIZE = 14
DEFAULT_ATTENTION_HEADS = 'mean'

# Model repository mapping
MODEL_REPOS = {
    "rad-dino": "microsoft/rad-dino",
    "dinov2-base": "facebook/dinov2-base", 
    "dinov2-small": "facebook/dinov2-small",
    "medsiglip": "google/medsiglip-448",
    "ark": "microsoft/swin-large-patch4-window12-384-in22k"
}

def _calculate_attention_shape(backbone_config: Any, 
                               images: torch.Tensor, 
                               multi_view: bool, 
                               num_layers: int, 
                               num_heads: int) -> Tuple[torch.Size, int]:
    """Calculate attention tensor shape and sequence length"""
    if multi_view and images.shape[1] == 4:
        # Multi-view: images shape is [B, 4, C, H, W]
        img_height, img_width = images.shape[3], images.shape[4]
    else:
        # Single-view: images shape is [B, C, H, W]
        img_height, img_width = images.shape[2], images.shape[3]
    
    patch_size = backbone_config.patch_size
    num_patches_h = img_height // patch_size
    num_patches_w = img_width // patch_size
    seq_len = num_patches_h * num_patches_w + 1
    
    if multi_view and images.shape[1] == 4:
        # Multi-view attention shape: [num_layers, B, 4, num_heads, seq_len, seq_len]
        attention_shape = (num_layers, images.shape[0], 4, num_heads, seq_len, seq_len)
    else:
        # Single-view attention shape: [num_layers, B, num_heads, seq_len, seq_len]
        attention_shape = (num_layers, images.shape[0], num_heads, seq_len, seq_len)
    
    return attention_shape, seq_len

def _load_best_model(checkpoint_dir: str, 
                     backbone: AutoModel, 
                     num_classes: int, 
                     accelerator: Accelerator, 
                     multi_view: bool = False, 
                     fusion_type: str = "mean") -> DinoClassifier:
    """Load PyTorch model from checkpoint"""
    # Load multi-view configuration from checkpoint if available
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    
    # Get multi-view parameters from checkpoint or use defaults
    num_views = ckpt.get("num_views", 4) if multi_view else None
    view_fusion_type = ckpt.get("view_fusion_type", fusion_type) if multi_view else None
    adapter_dim = ckpt.get("adapter_dim", None) if multi_view else None
    view_fusion_hidden_dim = ckpt.get("view_fusion_hidden_dim", None) if multi_view else None
    
    model = DinoClassifier(backbone, 
                          num_classes=num_classes, 
                          multi_view=multi_view,
                          num_views=num_views,
                          view_fusion_type=view_fusion_type,
                          adapter_dim=adapter_dim,
                          view_fusion_hidden_dim=view_fusion_hidden_dim)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model

def _parse_save_heads(save_heads_arg: Optional[str] = None) -> Union[str, int]:
    """Parse the save_heads argument for attention visualization"""
    if save_heads_arg is None:
        return DEFAULT_ATTENTION_HEADS
    
    if save_heads_arg in ['mean', 'max', 'min']:
        logger.info(f"Using attention head fusion: {save_heads_arg}")
        return save_heads_arg
    elif save_heads_arg.isdigit():
        head_fusion = int(save_heads_arg)
        logger.info(f"Using attention head fusion: {head_fusion}")
        return head_fusion
    else:
        raise ValueError(f"Invalid save_heads argument: {save_heads_arg}. Must be one of 'mean', 'max', 'min' or an integer.")

def _run_gradcam_visualization(model: DinoClassifier | MedSigClassifier | ArkClassifier, 
                               images: torch.Tensor, 
                               image_ids: List[str], 
                               class_labels: List, 
                               output_gradcam: str, 
                               accelerator: Accelerator,
                               image_mean: Optional[torch.Tensor] = None, 
                               image_std: Optional[torch.Tensor] = None) -> None:
    """Run GradCAM visualization for a batch"""
    target_layer = model.backbone.encoder.layer[-1].norm1
    sample_idx = 0
    # Multi-view GradCAM: [4, C, H, W] -> [1, 4, C, H, W]
    # Single-view GradCAM: [C, H, W] -> [1, C, H, W]
    sample_image = images[sample_idx].unsqueeze(0)  
    
    if accelerator.device.type == 'cuda':
        torch.cuda.empty_cache()
        
    visualize_gradcam(model, sample_image, target_layer, 
                      image_ids[sample_idx], output_gradcam, 
                      accelerator, image_mean, image_std, class_labels=class_labels)
    
    if accelerator.device.type == 'cuda':
        torch.cuda.empty_cache()

def _run_onnx_inference(session: onnxruntime.InferenceSession, 
                        input_name: str, 
                        output_names: List[str], 
                        images: torch.Tensor, show_attention: bool, 
                        accelerator: Accelerator, 
                        num_classes: int,
                        backbone_config: Any, 
                        multi_view: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ONNX inference workflow"""
    batch_size = images.shape[0]
    attentions = None
    
    # Validate ONNX model input shape compatibility (excluding batch dimension)
    onnx_input_shape = session.get_inputs()[0].shape
    expected_shape = tuple(onnx_input_shape[1:])
    actual_shape = tuple(images.shape[1:])
    
    if expected_shape != actual_shape:
        raise ValueError(
            f"Shape mismatch: ONNX model expects input shape {expected_shape}, "
            f"but got {actual_shape} from data. "
            "Check if you are using the correct model for single-view or multi-view inference."
        )
    
    with torch.no_grad():
        if 'CUDAExecutionProvider' in session.get_providers():
            # GPU inference with IOBinding
            io_binding = session.io_binding()
            device_id = accelerator.device.index if accelerator.device.index is not None else 0
            
            # Bind input
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(images.shape),
                buffer_ptr=images.data_ptr()
            )
            
            # Bind logits output
            logits = torch.empty((batch_size, num_classes), dtype=torch.float32, device=accelerator.device)
            io_binding.bind_output(
                name=output_names[0],
                device_type='cuda', 
                device_id=device_id,
                element_type=np.float32,
                shape=(batch_size, num_classes),
                buffer_ptr=logits.data_ptr()
            )
            
            # Bind attention output if needed and available
            if show_attention and len(output_names) > 1:
                num_layers = backbone_config.num_hidden_layers
                num_heads = backbone_config.num_attention_heads
                attention_shape, _ = _calculate_attention_shape(backbone_config, images, multi_view, num_layers, num_heads)
                
                attentions = torch.empty(attention_shape, dtype=torch.float32, device=accelerator.device)
                io_binding.bind_output(
                    name=output_names[1],
                    device_type='cuda',
                    device_id=device_id,
                    element_type=np.float32,
                    shape=attention_shape,
                    buffer_ptr=attentions.data_ptr()
                )
            
            session.run_with_iobinding(io_binding)
            
        else:
            # CPU inference fallback
            input_data = images.cpu().numpy()
            onnx_outputs = session.run(output_names, {input_name: input_data})
            logits = torch.from_numpy(onnx_outputs[0]).to(accelerator.device)
            if show_attention and len(onnx_outputs) > 1:
                attentions = torch.from_numpy(onnx_outputs[1]).to(accelerator.device)
    
    return logits, attentions

def _run_pytorch_inference(model: DinoClassifier | MedSigClassifier | ArkClassifier, images: torch.Tensor, 
                          show_gradcam: bool, image_ids: List[str], 
                          class_labels: List, output_gradcam: str, 
                          accelerator: Accelerator, image_mean: Optional[torch.Tensor] = None, 
                          image_std: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch inference workflow"""
    with torch.no_grad():
        logits, attentions = model(images)
    
    # GradCAM visualization
    if show_gradcam:
        _run_gradcam_visualization(model, images, image_ids, class_labels, 
                                  output_gradcam, accelerator, image_mean, image_std)
    
    return logits, attentions

def _validate_input_shape(images: torch.Tensor, multi_view: bool) -> None:
    """Validate input tensor shape for multi-view vs single-view"""
    if multi_view:
        if len(images.shape) != 5 or images.shape[1] != 4:
            raise ValueError(f"Multi-view inference expects images shape [B, 4, C, H, W], got {images.shape}")
        logger.debug(f"Multi-view batch shape: {images.shape}")
    else:
        if len(images.shape) != 4:
            raise ValueError(f"Single-view inference expects images shape [B, C, H, W], got {images.shape}")
        logger.debug(f"Single-view batch shape: {images.shape}")

def _validate_onnx_shape(onnx_input_shape: Tuple, multi_view: bool, backbone_config: Any) -> None:
    """Validate ONNX model input shape"""
    if multi_view:
        # For multi-view, expect [batch_size, 4, channels, height, width]
        expected_shape = (None, 4, 3, backbone_config.image_size, backbone_config.image_size)
        if len(onnx_input_shape) != 5 or onnx_input_shape[1] != 4:
            raise ValueError(f"ONNX model input shape {onnx_input_shape} doesn't match expected multi-view shape {expected_shape}")
    else:
        # For single-view, expect [batch_size, channels, height, width]
        expected_shape = (None, 3, backbone_config.image_size, backbone_config.image_size)
        if len(onnx_input_shape) != 4:
            raise ValueError(f"ONNX model input shape {onnx_input_shape} doesn't match expected single-view shape {expected_shape}")

def create_output_directories(output_dir: str, accelerator: Accelerator) -> OutputPaths:
    """Create output directories and return paths"""
    if accelerator.is_main_process:
        os.makedirs(f"{output_dir}/figs", exist_ok=True)
        os.makedirs(f"{output_dir}/table", exist_ok=True)
        os.makedirs(f"{output_dir}/gradcam", exist_ok=True)
        os.makedirs(f"{output_dir}/attention", exist_ok=True)
        os.makedirs(f"{output_dir}/lrp", exist_ok=True)
    
    return OutputPaths(
        base=output_dir,
        figs=f"{output_dir}/figs",
        table=f"{output_dir}/table",
        gradcam=f"{output_dir}/gradcam",
        attention=f"{output_dir}/attention",
        lrp=f"{output_dir}/lrp"
    )

def get_args_parser() -> argparse.ArgumentParser:
    """Create argument parser for inference script"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, 
                       choices=['multilabel', 'multiclass', 'binary'])
    parser.add_argument('--data', type=str, required=True, 
                       choices=['VinDr-CXR', 'CANDID-PTX', 'RSNA-Pneumonia', 'VinDr-Mammo'])
    parser.add_argument('--model', type=str, required=True, 
                       choices=['rad-dino', 'dinov2-small', 'dinov2-base', 'medsiglip', 'ark']) 
    parser.add_argument('--model-path', required=True, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fusion-type', type=str, default='mean', 
                       choices=['mean', 'weighted_mean', 'mlp_adapter']) 
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
    return parser

def load_model(checkpoint_dir: str, 
               model_repo: str, 
               num_classes: int, 
               accelerator: Accelerator, 
               show_gradcam: bool, 
               show_lrp: bool, 
               multi_view: bool = False, 
               fusion_type: str = "mean",
               model_type: str = "dino") -> ModelWrapper:
    """Load model (ONNX or PyTorch) based on availability and requirements"""
    
    # Handle different model types
    if model_type == "ark":
        backbone = None  # Ark models don't use HuggingFace backbones
        backbone_config = type('Config', (), {'image_size': 768})()  # Mock config for Ark
    elif model_type == "medsiglip" or model_type == "dino":
        backbone = AutoModel.from_pretrained(model_repo)
        backbone_config = backbone.config
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    onnx_model_path = os.path.join(checkpoint_dir, "best.onnx")
    
    # Use ONNX if available and no gradient-based visualization approaches are required
    if os.path.exists(onnx_model_path) and not show_gradcam and not show_lrp:
        # Configure providers for GPU inference
        providers = []
        if accelerator.device.type == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')  # Fallback to CPU
        
        session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        input_name = session.get_inputs()[0].name            
        output_names = [o.name for o in session.get_outputs()]
        
        # Log ONNX model info
        current_providers = session.get_providers()
        logger.info(f"Successfully loaded ONNX model from {onnx_model_path}")
        logger.info(f"ONNX providers: {current_providers}")
        logger.info(f"Input name: {input_name}")
        logger.info(f"Output names: {output_names}")
        
        # Validate ONNX model input shape
        onnx_input_shape = session.get_inputs()[0].shape
        logger.info(f"ONNX model input shape: {onnx_input_shape}")
        _validate_onnx_shape(onnx_input_shape, multi_view, backbone_config)
        
        return ModelWrapper(
            model_type='onnx',
            session=session,
            input_name=input_name,
            output_names=output_names,
            config=backbone_config,
            device=accelerator.device,
            multi_view=multi_view
        )
    else:
        # Use PyTorch model
        if show_gradcam:
            logger.info("GradCAM requested - using PyTorch model for visualization support")
        elif show_lrp:
            logger.info("LRP requested - using PyTorch model for visualization support")
        else:
            logger.warning(f"ONNX model not found at {onnx_model_path}. Fallback to PyTorch state_dict instead...")
        
        # Load appropriate model based on model_type
        if model_type == "ark":
            model = _load_best_ark_model(checkpoint_dir, num_classes, accelerator, multi_view, fusion_type)
        elif model_type == "medsiglip":
            model = _load_best_medsig_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, fusion_type)
        else: # dino models
            model = _load_best_dino_model(checkpoint_dir, backbone, num_classes, accelerator, multi_view, fusion_type)
        
        return ModelWrapper(
            model_type='pytorch',
            model=model,
            config=backbone_config,
            device=accelerator.device,
            multi_view=multi_view
        )

def validate_args(config: InferenceConfig) -> None:
    """Validate command line arguments"""
    if config.multi_view and config.data != 'VinDr-Mammo':
        raise ValueError("Multi-view processing is only supported for VinDr-Mammo dataset")
    
    if (config.save_heads is None or config.attention_threshold is None) and config.show_attention:
        raise ValueError("Attention visualization requires specifying save_heads and attention_threshold")
    
    if (config.save_heads is not None and config.attention_threshold is not None) and not config.show_attention:
        logger.warning("Attention visualization is not enabled, but save_heads and attention_threshold are specified.")

def run_inference(model_wrapper: ModelWrapper, loader: DataLoader, 
                 accelerator: Accelerator, config: InferenceConfig, 
                 output_paths: OutputPaths, model_repo: str, 
                 class_labels: List, num_classes: int) -> None:
    """Main inference function"""
    # Load image processor for visualization if needed
    image_processor = None
    image_mean = None
    image_std = None
    if config.show_attention or config.show_gradcam or config.show_lrp:
        image_processor = AutoImageProcessor.from_pretrained(model_repo)
        image_mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
        image_std = torch.tensor(image_processor.image_std).view(3, 1, 1)

    # Initialize result storage
    all_ids = []
    all_trues = []
    all_preds_prob = []
    
    # GradCAM visualization counter
    gradcam_count = 0
    
    # Validate rollout computation
    if config.compute_rollout and not config.show_attention:
        raise ValueError("Attention rollout computation is only supported when attention visualization is enabled.")
    
    # Extract model components
    is_onnx = model_wrapper.model_type == 'onnx'
    multi_view = model_wrapper.multi_view
    
    if is_onnx:
        session = model_wrapper.session
        input_name = model_wrapper.input_name
        output_names = model_wrapper.output_names
        backbone_config = model_wrapper.config
        logger.info(f"Running inference with ONNX model (multi_view={multi_view})")
        
        if config.show_attention and len(output_names) < 2:
            raise ValueError("ONNX model does not have attention outputs, which is required for attention visualization.")
    else:
        model = model_wrapper.model
        backbone_config = model_wrapper.config
        model.eval()
        logger.info(f"Running inference with PyTorch model (multi_view={multi_view})")
    
    for batch in tqdm(loader, desc="Inference", disable=not accelerator.is_main_process):
        images, targets, image_ids = batch
        images = images.to(accelerator.device)
        
        # Validate input shapes
        _validate_input_shape(images, multi_view)
        
        # Clear CUDA cache before inference
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Run inference based on model type
        if is_onnx:
            logits, attentions = _run_onnx_inference(
                session, input_name, output_names, images, config.show_attention,
                accelerator, num_classes, backbone_config, multi_view)
        else:
            compute_gradcam = gradcam_count < MAX_GRADCAM_IMAGES and config.show_gradcam
            logits, attentions = _run_pytorch_inference(
                model, images, compute_gradcam, 
                image_ids, class_labels, output_paths.gradcam, accelerator, image_mean, image_std)
            gradcam_count += 1
        
        # Attention visualization 
        if config.show_attention and attentions is not None:
            save_heads_param = _parse_save_heads(config.save_heads)
            patch_size = getattr(backbone_config, 'patch_size', DEFAULT_PATCH_SIZE)
            
            visualize_attention_maps(
                attentions, images, image_ids, output_paths.attention, accelerator, image_mean, image_std, 
                patch_size=patch_size, threshold=config.attention_threshold, head_fusion=save_heads_param, 
                compute_rollout=config.compute_rollout, rollout_discard_ratio=0.0)
            
        # LRP visualization
        if config.show_lrp and not is_onnx:
            visualize_lrp_maps(
                model, images, image_mean, image_std, image_ids, output_paths.lrp, multi_view)
        
        # Clear CUDA cache after visualizations
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Process predictions
        if config.task == "multiclass":
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        else:
            probs = torch.sigmoid(logits).cpu().numpy()  
        trues = targets.cpu().numpy()
        
        all_ids.extend(image_ids)
        all_trues.append(trues)
        all_preds_prob.append(probs)
    
    # Save results and compute metrics
    Y_true = np.concatenate(all_trues, axis=0)
    Y_pred_prob = np.concatenate(all_preds_prob, axis=0)
    # Temporary check
    assert len(all_ids) == Y_true.shape[0], f"len(all_ids) = {len(all_ids)}, Y_true.shape[0] = {Y_true.shape[0]}"
    assert Y_pred_prob.shape[0] == Y_true.shape[0], f"Y_pred_prob.shape[0] = {Y_pred_prob.shape[0]}, Y_true.shape[0] = {Y_true.shape[0]}"
    
    if accelerator.is_main_process:
        # Prepare true_labels and pred_labels for CSV
        if config.task == "multiclass":
            true_col = Y_true.astype(int).tolist()
            pred_indices = np.argmax(Y_pred_prob, axis=1)
            pred_labels  = [class_labels[i] for i in pred_indices]
            # Convert integer labels to BIRADS strings for visualization
            class_labels = [f"BIRADS_{label+1}" for label in class_labels]
        elif config.task == "binary":
            true_col    = Y_true.squeeze().astype(int).tolist()
            prob_pos    = Y_pred_prob.squeeze()
            pred_labels = (prob_pos >= 0.5).astype(int).tolist()
        else:  # multilabel
            true_col = [list(map(int, row)) for row in Y_true]
            threshold   = 0.5
            pred_labels = [
                [class_labels[i] for i, p in enumerate(row) if p >= threshold]
                for row in Y_pred_prob
            ]

        df = pd.DataFrame({
            "image_id": all_ids,
            "true_labels": true_col ,
            "pred_labels": pred_labels,
            "pred_probs": [list(row) for row in Y_pred_prob],
        })
        
        # Save predictions CSV
        output_csv = os.path.join(output_paths.table, 'predictions.csv')
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved predictions to {output_csv}!")

        # Compute and save metrics
        metrics = compute_evaluation_metrics(Y_true, Y_pred_prob, config.task, class_labels, output_paths.figs, accelerator)
        
        metrics_path = os.path.join(output_paths.table, "metrics.json")
        with open(metrics_path, "w") as jf:
            json.dump(metrics, jf, indent=4)
        logger.info(f"Saved ROC/PR curves to {output_paths.figs} and metrics in JSON to {metrics_path}!")
        
        # Log visualization outputs
        if config.show_attention:
            logger.info(f"Attention visualizations saved to {output_paths.attention}")
        if config.show_gradcam:
            logger.info(f"GradCAM visualizations saved to {output_paths.gradcam}")
        if config.show_lrp:
            logger.info(f"LRP visualizations saved to {output_paths.lrp}")

def main():
    """Main function"""
    # Parse arguments
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create configuration object
    config = InferenceConfig(
        task=args.task,
        data=args.data,
        model=args.model,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        fusion_type=args.fusion_type,
        multi_view=args.multi_view,
        optimize_compute=args.optimize_compute,
        show_attention=args.show_attention,
        show_lrp=args.show_lrp,
        show_gradcam=args.show_gradcam,
        attention_threshold=args.attention_threshold,
        save_heads=args.save_heads,
        compute_rollout=args.compute_rollout
    )
    
    # Validate arguments
    validate_args(config)
    
    # Setup accelerator
    accelerator = Accelerator(mixed_precision="fp16" if config.optimize_compute else "no")
    
    # Get model repository
    if config.model not in MODEL_REPOS:
        raise ValueError(f"Model {config.model} is not supported. Please choose from {list(MODEL_REPOS.keys())}.")
    repo = MODEL_REPOS[config.model]
    logger.info(f"Running inference with multi_view={config.multi_view}")
    
    # Load data configuration
    data_cfg = yaml.safe_load(open(os.path.join(CURR_DIR, "../configs/data_config.yaml")))
    dataset_config = data_cfg.get(config.data, {})
    if not dataset_config:
        raise ValueError(f"Dataset {config.data} not found in data_config.yaml")
    
    # Get data root folder based on multi-view setting
    data_root_folder = dataset_config.get("data_root_folder", None)
    if config.multi_view and dataset_config.get("multi_view", {}).get("data_root_folder_multi_view"):
        data_root_folder = dataset_config["multi_view"]["data_root_folder_multi_view"]
    
    if data_root_folder is None:
        raise ValueError(f"Data root folder is None for {config.data}. Please check data_config.yaml")

    # Setup paths
    checkpoint_dir = os.path.join(CURR_DIR, "../..", config.model_path)
    _, test_transforms = get_transforms(repo)

    # Create data loader
    test_ds = RadImageClassificationDataset(data_root_folder, "test", config.task, 
                                           transform=test_transforms, multi_view=config.multi_view)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid shared memory issues
        pin_memory=False,  # Disable pin_memory since we're not using workers
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=False  # Disable persistent workers
    )
    test_loader = accelerator.prepare(test_loader)
    
    # ------------ model load ------------
    if config.task == "binary":
        class_labels = None
        num_classes = 1
    elif config.task == "multiclass":
        class_labels = list((set(test_ds.labels)))
        num_classes = len(class_labels)
    else:
        class_labels = test_ds.labels
        num_classes = len(class_labels)
    
    # Load model
    model_wrapper = load_model(checkpoint_dir, repo, num_classes, accelerator, 
                              config.show_gradcam, config.show_lrp, config.multi_view, config.fusion_type,
                              model_type=config.model)

    # Validate ONNX multi-view compatibility
    if config.multi_view and model_wrapper.model_type == 'onnx':
        onnx_input_shape = model_wrapper.session.get_inputs()[0].shape
        if len(onnx_input_shape) != 5 or onnx_input_shape[1] != 4:
            raise ValueError("Multi-view inference requested but ONNX model was exported for single-view. Please export the ONNX model with multi-view support.")

    # Setup output directories
    modelname = config.model_path.rsplit('/', 1)[-1]
    output_path = os.path.join(CURR_DIR, config.output_path, config.data, modelname)
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)
    
    output_paths = create_output_directories(output_path, accelerator)
    
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
