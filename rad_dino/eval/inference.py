import os
import onnxruntime
import argparse
import torch
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from accelerate import Accelerator
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from rad_dino.data.dataset import RadImageClassificationDataset
from rad_dino.utils.data_utils import get_transforms, collate_fn
from rad_dino.utils.visualize_gradcam import visualize_gradcam
from rad_dino.utils.plot_utils import visualize_evaluate_metrics
from rad_dino.utils.visualize_attention import visualize_attention_maps
from rad_dino.models.model import DinoClassifier
from transformers import AutoModel
import logging
from rad_dino.loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['multilabel', 'multiclass', 'binary', 'regression', 'ordinal', 'segmentation', 'text_generation'])
    parser.add_argument('--data', type=str, required=True, choices=['VinDr-CXR', 'CANDID-PTX', 'RSNA-Pneumonia', 'VinDr-Mammo'])
    parser.add_argument('--model', type=str, required=True, choices=['rad_dino', 'dinov2-small', 'dinov2-base']) 
    parser.add_argument('--model-path', required=True, type=str)
    parser.add_argument('--output-path', required=True, type=str)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument(
        "--optimize-compute",
        action="store_true",
        help="Whether to use advanced tricks to lessen the heavy computational resource.",
    )
    parser.add_argument('--show-attention', action='store_true')
    parser.add_argument('--show-gradcam', action='store_true')
    parser.add_argument('--attention-threshold', type=float, default=None, help="Threshold for attention masking")
    parser.add_argument('--save-heads', type=str, default=None,
                        help="Which attention heads to save: 'all', 'mean', integer for random N heads, or comma-separated indices (e.g., '0,5,11')")
    return parser

def _load_best_model(checkpoint_dir, backbone, num_classes, accelerator):
    model = DinoClassifier(backbone, num_classes=num_classes)
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model

def load_model(checkpoint_dir, model_repo, num_classes, accelerator, show_gradcam):
    backbone = AutoModel.from_pretrained(model_repo)
    backbone_config = backbone.config
    onnx_model_path = os.path.join(checkpoint_dir, "best.onnx")
    if os.path.exists(onnx_model_path) and not show_gradcam:
        # Configure providers for GPU inference
        providers = []
        if accelerator.device.type == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')  # Fallback to CPU
        session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        input_name = session.get_inputs()[0].name            
        output_names = [o.name for o in session.get_outputs()]
        
        # Check which provider is actually being used
        current_providers = session.get_providers()
        logger.info(f"Successfully loaded ONNX model from {onnx_model_path}")
        logger.info(f"ONNX providers: {current_providers}")
        logger.info(f"Input name: {input_name}")
        logger.info(f"Output names: {output_names}")
        
        # Return ONNX session with metadata
        return {
            'type': 'onnx',
            'session': session,
            'input_name': input_name,
            'output_names': output_names,
            'config': backbone_config,
            'device': accelerator.device,
            }
    else:
        if show_gradcam:
            logger.info("GradCAM requested - using PyTorch model for visualization support")
        else:
            logger.warning(f"ONNX model not found at {onnx_model_path}. Fallback to PyTorch state_dict instead...")
        
        model = _load_best_model(checkpoint_dir, backbone, num_classes, accelerator)
        return {
            'type': 'pytorch',
            'model': model,
            'config': backbone_config,
            'device': accelerator.device
        }

def _run_pytorch_inference(model, images, show_gradcam, image_ids, class_labels, output_gradcam, accelerator, model_repo):
    """PyTorch inference workflow"""
    with torch.no_grad():
        logits, attentions = model(images)
    
    # GradCAM visualization
    if show_gradcam:
        target_layer = model.backbone.encoder.layer[-1].norm1
        sample_idx = 0 
        sample_image = images[sample_idx].unsqueeze(0)
        
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        visualize_gradcam(model, sample_image, target_layer, 
                         image_ids[sample_idx], output_gradcam, accelerator, model_repo=model_repo, class_labels=class_labels)
        
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return logits, attentions

def _run_onnx_inference(session, input_name, output_names, images, show_attention, 
                       accelerator, class_labels, backbone_config):
    """ONNX inference workflow"""
    batch_size = images.shape[0]
    num_classes = len(class_labels) if class_labels is not None else 1
    attentions = None
    
    with torch.no_grad():
        if 'CUDAExecutionProvider' in session.get_providers():
            # GPU inference with IOBinding
            io_binding = session.io_binding()
            
            # Bind input
            io_binding.bind_input(
                name=input_name,
                device_type='cuda',
                device_id=accelerator.device.index if accelerator.device.index is not None else 0,
                element_type=np.float32,
                shape=tuple(images.shape),
                buffer_ptr=images.data_ptr()
            )
            
            # Bind logits output
            logits = torch.empty((batch_size, num_classes), dtype=torch.float32, device=accelerator.device)
            io_binding.bind_output(
                name=output_names[0],
                device_type='cuda', 
                device_id=accelerator.device.index if accelerator.device.index is not None else 0,
                element_type=np.float32,
                shape=(batch_size, num_classes),
                buffer_ptr=logits.data_ptr()
            )
            
            # Bind attention output if needed and available
            if show_attention and len(output_names) > 1:
                # Calculate attention tensor shape dynamically
                num_layers = backbone_config.num_hidden_layers
                num_heads = backbone_config.num_attention_heads
                
                img_height, img_width = images.shape[2], images.shape[3]
                patch_size = backbone_config.patch_size
                num_patches_h = img_height // patch_size
                num_patches_w = img_width // patch_size
                seq_len = num_patches_h * num_patches_w + 1
                
                attentions = torch.empty((num_layers, batch_size, num_heads, seq_len, seq_len), 
                                       dtype=torch.float32, device=accelerator.device)
                io_binding.bind_output(
                    name=output_names[1],
                    device_type='cuda',
                    device_id=accelerator.device.index if accelerator.device.index is not None else 0,
                    element_type=np.float32,
                    shape=(num_layers, batch_size, num_heads, seq_len, seq_len),
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

def _parse_save_heads(save_heads_arg):
    """
    Parse the save_heads argument for attention visualization.
    
    Args:
        save_heads_arg: String or None - can be 'all', 'mean', integer, or comma-separated indices
        
    Returns:
        Parsed save_heads parameter for visualize_attention_maps
    """
    if save_heads_arg is None:
        return 5  # Default fallback
    
    if save_heads_arg == 'all':
        return 'all'
    elif save_heads_arg == 'mean':
        return 'mean'
    else:
        try:
            # Check if it's a single integer (for random N heads)
            if ',' not in save_heads_arg:
                num_heads = int(save_heads_arg)
                logger.info(f"Will randomly select {num_heads} attention heads for visualization")
                return num_heads
            else:
                # Parse comma-separated indices
                head_indices = [int(h.strip()) for h in save_heads_arg.split(',')]
                logger.info(f"Will save specific attention heads: {head_indices}")
                return head_indices
        except (ValueError, TypeError):
            logger.warning(f"Invalid save_heads argument: {save_heads_arg}. Using randomly selected 5 heads instead.")
            return 5

def run_inference(
    model_wrapper, 
    loader, 
    accelerator, 
    show_attention, 
    show_gradcam, 
    attention_threshold, 
    num_save_heads, 
    output_dir, 
    model_repo,
    class_labels
):
    if accelerator.is_main_process:
        os.makedirs(f"{output_dir}/figs", exist_ok=True)
        os.makedirs(f"{output_dir}/table", exist_ok=True)
        os.makedirs(f"{output_dir}/gradcam", exist_ok=True)
        os.makedirs(f"{output_dir}/attention", exist_ok=True)
    
    output_table = f"{output_dir}/table"
    output_figdir = f"{output_dir}/figs"
    output_gradcam = f"{output_dir}/gradcam"
    output_attention = f"{output_dir}/attention"

    all_ids = []
    all_trues = []
    all_pred_raw = []
    all_preds_prob = []
    
    # Limits for visualizations
    max_gradcam_images = 10
    gradcam_count = 0
    
    # Extract model wrapper components
    is_onnx = model_wrapper['type'] == 'onnx'
    
    if is_onnx:
        session = model_wrapper['session']
        input_name = model_wrapper['input_name']
        output_names = model_wrapper['output_names']
        backbone_config = model_wrapper['config']
        logger.info("Running inference with ONNX model")
        
        if show_attention and len(output_names) < 2:
            raise ValueError("ONNX model does not have attention outputs, which is required for attention visualization.")
    else:
        model = model_wrapper['model']
        backbone_config = model_wrapper['config']
        model.eval()
        logger.info("Running inference with PyTorch model")
    
    for batch in tqdm(loader, desc="Inference", disable=not accelerator.is_main_process):
        images, targets, image_ids = batch
        images = images.to(accelerator.device)
        
        # Clear CUDA cache before inference
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Run inference based on model type
        if is_onnx:
            logits, attentions = _run_onnx_inference(
                session, input_name, output_names, images, show_attention,
                accelerator, class_labels, backbone_config
            )
        else:
            compute_gradcam = gradcam_count < max_gradcam_images and show_gradcam
            logits, attentions = _run_pytorch_inference(
                model, images, compute_gradcam, 
                image_ids, class_labels, output_gradcam, accelerator, model_repo)
            gradcam_count += 1
            
        
        # Attention visualization 
        if show_attention and attentions is not None:
            # Parse save_heads only when actually needed
            save_heads_param = _parse_save_heads(num_save_heads)
            patch_size = getattr(backbone_config, 'patch_size', 14)  # Default to 14
            
            visualize_attention_maps(
                attentions, images, image_ids, output_attention, accelerator,
                model_repo, patch_size=patch_size, threshold=attention_threshold, save_heads=save_heads_param)
        
        # Clear CUDA cache after visualizations
        if accelerator.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Process predictions
        raws = logits.cpu().numpy()
        probs = torch.sigmoid(logits).cpu().numpy()  
        trues = targets.cpu().numpy()
        
        all_ids.extend(image_ids)
        all_trues.append(trues)
        all_pred_raw.append(raws)
        all_preds_prob.append(probs)
    
    # Save results and compute metrics
    Y_true = np.vstack(all_trues)
    Y_pred_raw = np.vstack(all_pred_raw)
    Y_pred_prob = np.vstack(all_preds_prob)

    if accelerator.is_main_process:
        # Save predictions CSV
        df = pd.DataFrame({
            "image_id": all_ids,
            "true_labels": [list(map(int, row)) for row in Y_true],
            "pred_raws": [list(row) for row in Y_pred_raw],
            "pred_probs": [list(row) for row in Y_pred_prob],
        })
        output_csv = os.path.join(output_table, 'predictions.csv')
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved predictions to {output_csv}!")

        # Compute and save metrics
        metrics = {}
        
        if Y_pred_prob.shape[1] == 1:  # Binary classification
            # For binary classification, we have 1 output neuron
            y_true_binary = Y_true.squeeze()  # Remove extra dimension if present
            y_pred_binary = Y_pred_prob.squeeze()  # Remove extra dimension if present
            pr_auc, roc_auc = visualize_evaluate_metrics(y_true_binary, y_pred_binary, output_figdir, accelerator, axis=None)
            metrics["AUROC"] = float(roc_auc)
            metrics["AUPRC"] = float(pr_auc)
        else:  # Multilabel classification
            # For multilabel, iterate through each class
            for i, cls in enumerate(class_labels):
                pr_auc, roc_auc = visualize_evaluate_metrics(Y_true[:,i], Y_pred_prob[:,i], output_figdir, accelerator, label=cls, axis=None)
                metrics[cls] = {
                    "AUROC": float(roc_auc),
                    "AUPRC": float(pr_auc)
                }
        
        metrics_path = os.path.join(output_table, "metrics.json")
        with open(metrics_path, "w") as jf:
            json.dump(metrics, jf, indent=4)
        logger.info(f"Saved ROC/PR curves to {output_figdir} and metrics in JSON to {metrics_path}!")
        
        if show_attention:
            logger.info(f"Attention visualizations saved to {output_attention}")
        if show_gradcam:
            logger.info(f"GradCAM visualizations saved to {output_gradcam}")

def main():
    # ------------ config ------------
    parser = get_args_parser()
    args = parser.parse_args()
    
    if (args.save_heads is None or args.attention_threshold is None) and args.show_attention:
        raise ValueError("Attention visualization is required to specify the number of attention heads to be saved and set the attention threshold")
    
    if (args.save_heads is not None and args.attention_threshold is not None) and not args.show_attention:
        logger.warn("Attention visualization is not enabled, but the number of attention heads to be saved and the attention threshold are specified. Please enable attention visualization to save attention heads and set the attention threshold.")
    
    accelerator = Accelerator(mixed_precision="fp16" if args.optimize_compute else "no")
    if args.model == "rad_dino":
        repo = "microsoft/rad-dino"
    elif args.model == "dinov2-base":
        repo = "facebook/dinov2-base"
    elif args.model == "dinov2-small":
        repo = "facebook/dinov2-small"
    else:
        raise ValueError(f"Model {args.model} is not supported. Please choose from 'rad_dino', 'dinov2-base', 'dinov2-small'.")
    data_cfg = yaml.safe_load(open(os.path.join(CURR_DIR, "../configs/data_config.yaml")))
    data_root_folder = data_cfg.get(args.data, {}).get("data_root_folder", None)
    if data_root_folder is None:
        raise ValueError(f"Either class labels or data root folder is NoneType, which is not allowed. Please check if they are specified correctly in data_config.yaml")

    checkpoint_dir = os.path.join(CURR_DIR, "../..", args.model_path)
    _, test_transforms = get_transforms(repo)

    # ------------ data loader ------------
    test_ds = RadImageClassificationDataset(data_root_folder, "test", args.task, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid shared memory issues
        pin_memory=False,  # Disable pin_memory since we're not using workers
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=False  # Disable persistent workers
    )
    test_loader = accelerator.prepare(test_loader)
    class_labels = test_ds.labels if args.task != "binary" else None
    
    # ------------ model load ------------
    num_classes = 1 if args.task == "binary" else len(class_labels)
    model_wrapper = load_model(checkpoint_dir, repo, num_classes, accelerator, args.show_gradcam)

    # ------------ inference & metrics ------------
    modelname = args.model_path.rsplit('/', 1)[-1]
    output_path = os.path.join(CURR_DIR, args.output_path, args.data, modelname)
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)
    run_inference(
        model_wrapper,
        test_loader,
        accelerator,
        args.show_attention,
        args.show_gradcam,
        args.attention_threshold,
        args.save_heads,
        output_path,
        repo,
        class_labels
    )

if __name__ == "__main__":
    main()
