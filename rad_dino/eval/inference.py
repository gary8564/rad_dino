import os
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
from data.VinDrCXR.data import VinDrCXR_Dataset
from utils.utils import get_transforms, collate_fn
from utils.plots import visualize_gradcam, visualize_evaluate_metrics
from models.model import DinoClassifier
from transformers import AutoModel
import logging
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="multilabel", choices=['multilabel', 'multiclass', 'binary', 'regression', 'ordinal', 'segmentation', 'text_generation'])
    parser.add_argument('--data', type=str, default='VinDr-CXR', choices=['VinDr-CXR', 'CANDID-PTX', 'RSNA-Pneumonia', 'VinDr-Mammography'])
    parser.add_argument('--model', type=str, default='rad_dino', choices=['rad_dino', 'dinov2']) 
    parser.add_argument('--model-path', default='runs/checkpoints_rad_dino_2025_05_18_233958', type=str)
    parser.add_argument('--output-path', default='../../experiments', type=str)
    parser.add_argument(
        "--optimize-compute",
        action="store_true",
        help="Whether to use advanced tricks to lessen the heavy computational resource.",
    )
    parser.add_argument('--show-attention', action='store_true')
    return parser

def _load_best_model(checkpoint_dir, model_repo, num_classes, accelerator):
    backbone = AutoModel.from_pretrained(model_repo)
    model = DinoClassifier(backbone, num_classes=num_classes)
    ckpt = torch.load(os.path.join(checkpoint_dir, "best.pt"), map_location=accelerator.device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(accelerator.device)
    return model

def load_model(checkpoint_dir, model_repo, num_classes, accelerator, show_attention):
    if show_attention:
        logger.warning(f"When GRAD-CAM is computed, load unscripted model instead.")
        return _load_best_model(checkpoint_dir, model_repo, num_classes, accelerator)
    try:
        model_path = torch.load(os.path.join(checkpoint_dir, "best_final_scripted.pt"), map_location=accelerator.device)
        model = torch.jit.load(model_path, map_location=accelerator.device)
        model = model.to(accelerator.device)
    except Exception as e:
        logger.warning(f"Failed to load scripted model: {e}. Attempting state dict instead...")
        model = _load_best_model(checkpoint_dir, model_repo, num_classes, accelerator)
    return model

def run_inference(
    model, loader, accelerator, class_labels, show_attention, output_dir
):
    if accelerator.is_main_process:
        os.makedirs(f"{output_dir}/figs", exist_ok=True)
        os.makedirs(f"{output_dir}/table", exist_ok=True)
        os.makedirs(f"{output_dir}/gradcam", exist_ok=True)
    output_table = f"{output_dir}/table"
    output_figdir = f"{output_dir}/figs"
    output_gradcam = f"{output_dir}/gradcam"

    all_ids = []
    all_trues = []
    all_pred_raw = []
    all_preds_prob = []
    
    # Process in smaller chunks to manage memory
    max_gradcam_images = 10  # Limit number of images for GradCAM
    gradcam_count = 0
    
    for batch in tqdm(loader, desc="Inference", disable=not accelerator.is_main_process):
        images, targets, image_ids = batch
        images = images.to(accelerator.device)
        
        # Handle GradCAM visualization separately with memory management
        if show_attention and gradcam_count < max_gradcam_images:
            target_layer = model.backbone.encoder.layer[-1].norm1
            sample_idx = 0 
            sample_image = images[sample_idx].unsqueeze(0)
            
            # Clear CUDA cache before GradCAM
            if accelerator.device.type == 'cuda':
                torch.cuda.empty_cache()
                
            visualize_gradcam(model, sample_image, class_labels, target_layer, image_ids[sample_idx], output_gradcam, accelerator)
            gradcam_count += 1
            
            # Clear CUDA cache after GradCAM
            if accelerator.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Regular inference
        with torch.no_grad():
            logits = model(images)
        raws = logits.cpu().numpy()  
        probs = torch.sigmoid(logits).cpu().numpy()  
        trues = targets.cpu().numpy()
        
        all_ids.extend(image_ids)
        all_trues.append(trues)
        all_pred_raw.append(raws)
        all_preds_prob.append(probs)
        
    Y_true = np.vstack(all_trues)   # (N, C)
    Y_pred_raw = np.vstack(all_pred_raw)   # (N, C)
    Y_pred_prob = np.vstack(all_preds_prob)   # (N, C)

    # Save CSV of predictions
    if accelerator.is_main_process:
        df = pd.DataFrame({
            "image_id": all_ids,
            "true_labels": [list(map(int, row)) for row in Y_true],
            "pred_raws": [list(row) for row in Y_pred_raw],
            "pred_probs": [list(row) for row in Y_pred_prob],
        })
        output_csv = os.path.join(output_table, 'predictions.csv')
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved predictions to {output_csv}!")

        # Collect metrics and plot ROC and PR curves per class
        metrics = {}
        for i, cls in enumerate(class_labels):
            pr_auc, roc_auc = visualize_evaluate_metrics(Y_true[:,i], Y_pred_prob[:,i], cls, output_figdir, accelerator, axis=None)
        
        # store metrics
        metrics[cls] = {
            "AUROC": float(roc_auc),
            "AUPRC": float(pr_auc)
        }
        
        # save metrics JSON
        metrics_path = os.path.join(output_table, "metrics.json")
        with open(metrics_path, "w") as jf:
            json.dump(metrics, jf, indent=4)
        logger.info(f"Saved ROC/PR curves to {output_figdir} and metrics in JSON to {metrics_path}!")

def main():
    # ------------ config ------------
    parser = get_args_parser()
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision="fp16" if args.optimize_compute else "no")
    repo   = "microsoft/rad-dino" if args.model == "rad_dino" else "facebook/dinov2-base"
    data_cfg = yaml.safe_load(open(os.path.join(CURR_DIR, "../configs/data_config.yaml")))
    train_cfg = yaml.safe_load(open(os.path.join(CURR_DIR, "../configs/train_config.yaml")))
    labels = data_cfg.get(args.data, {}).get("class_labels", None)
    data_root_folder = data_cfg.get(args.data, {}).get("data_root_folder", None)
    if labels is None or data_root_folder is None:
        raise ValueError(f"Either class labels or data root folder is NoneType, which is not allowed. Please check if they are specified correctly in data_config.yaml")
    num_classes = len(labels)

    checkpoint_dir = os.path.join(CURR_DIR, "../..", args.model_path)
    _, test_transforms = get_transforms(repo)

    # ------------ data loader ------------
    test_ds = VinDrCXR_Dataset(data_root_folder, "test", class_labels=labels, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,  # Disable multiprocessing to avoid shared memory issues
        pin_memory=False,  # Disable pin_memory since we're not using workers
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=False  # Disable persistent workers
    )
    test_loader = accelerator.prepare(test_loader)

    # ------------ model load ------------
    model = load_model(checkpoint_dir, repo, num_classes, accelerator, args.show_attention)

    # ------------ inference & metrics ------------
    modelname = args.model_path.rsplit('/', 1)[-1]
    output_path = os.path.join(CURR_DIR, args.output_path, args.data, modelname)
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)
    run_inference(
        model,
        test_loader,
        accelerator,
        labels,
        args.show_attention,
        output_path
    )

if __name__ == "__main__":
    main()
