import argparse
import logging
import yaml
import os
import copy
import math
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch 
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
import wandb

from rad_dino.utils.config_utils import setup_configs
from rad_dino.utils.data_utils import get_transforms, load_data, get_class_weights
from rad_dino.train.train_utils import get_criterion, get_eval_metrics
from rad_dino.utils.model_loader import load_pretrained_model
from rad_dino.models.dino import DinoClassifier
from rad_dino.models.siglip import MedSigClassifier
from rad_dino.models.ark import ArkClassifier, load_prtrained_ark_model
from rad_dino.loggings.setup import init_logging
from rad_dino.train.trainer import Trainer

init_logging()
logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 linear probling", add_help=add_help)
    parser.add_argument('--task', type=str, required=True, choices=['multilabel', 'multiclass', 'binary'])
    parser.add_argument('--data', type=str, required=True, choices=['VinDr-CXR', 'CANDID-PTX', 'RSNA-Pneumonia', 'VinDr-Mammo'])
    parser.add_argument('--model', type=str, required=True, choices=['rad-dino', 'dinov2-small', 'dinov2-base', 'medsiglip', 'ark']) 
    parser.add_argument('--kfold', type=int, default=None, help="Number of folds for cross-validation")
    parser.add_argument('--multi-view', action='store_true', help="Enable multi-view processing for mammography data")
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Whether to unfreeze the ViT backbone.")
    parser.add_argument('--unfreeze-num-layers', type=int, default=None, help="Number of transformer blocks to unfreeze from the end of the ViT backbone.")
    parser.add_argument(
        "--progressive-unfreeze",
        action="store_true",
        help="Whether to use progressive unfreezing of the ViT backbone layers. Unfreezes (epoch//2) layers at even epochs (2, 4, 6, ...).",
    )
    parser.add_argument(
        "--optimize-compute",
        action="store_true",
        help="Whether to use advanced tricks to lessen the heavy computational resource. ",
    )
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Whether to use weighted loss.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--resume-checkpoint-path", type=str, default=None, help="Path to the checkpoint to continue training from.",
    )
    parser.add_argument(
        "--pretrained-ark-path", type=str, default=None, help="Path to the Ark pre-trained checkpoint file.",
    )
    parser.add_argument(
        "--output-dir",
        default="../../runs/",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    return parser

def setup(args, accelerator: Accelerator):
    # Setup configs
    data_config, train_config = setup_configs(args.data, args.task) 
    num_workers = data_config.num_workers
    
    # Get multi-view configuration
    multi_view_config = data_config.get_multi_view_config(args.multi_view)
    
    # Get model repo to extract information for data augmentation
    if args.model == "rad-dino":
        model_repo = "microsoft/rad-dino"
    elif args.model == "dinov2-small":
        model_repo = "facebook/dinov2-small"
    elif args.model == "dinov2-base":
        model_repo = "facebook/dinov2-base"
    elif args.model == "medsiglip":
        model_repo = "google/medsiglip-448"
    elif args.model == "ark":
        model_repo = "ark-swin-large-768"  # Custom identifier for Ark models
    else:
        raise ValueError(f"Model {args.model} not supported.")
    train_transforms, val_transforms = get_transforms(model_repo)
   
    # Data setup
    batch_size = train_config.batch_size
    data_root_folder = data_config.get_data_root_folder(args.multi_view)
    
    logger.info(f"Loading data for {args.data} with kfold={args.kfold} and multi_view={args.multi_view}")
    data_loader = load_data(data_root_folder, args.task, batch_size, train_transforms, val_transforms, num_workers, accelerator.gradient_accumulation_steps, kfold=args.kfold, multi_view=args.multi_view)

    # Get actual number of classes from dataset
    if isinstance(data_loader, tuple):
        # Single split case
        dataset = data_loader[0].dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
    else:
        # K-fold case
        dataset = data_loader[0][0].dataset
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
    
    if args.task == "binary":
        num_classes = 1  # Binary classification has 1 output
    else:
        num_classes = len(set(dataset.labels))  # Get number of classes from dataset
        
    # Model setup
    if args.model == "ark":
        ark_checkpoint_path = args.pretrained_ark_path
        # Determine if linear probing or fine-tuning 
        # For linear probing: use projector (use_backbone_projector=True)
        # For fine-tuning: don't use projector (use_backbone_projector=False)
        use_backbone_projector = not args.unfreeze_backbone
        logger.info(f"Ark model setup: {'Linear probing' if use_backbone_projector else 'Fine-tuning'} - "
                   f"use_backbone_projector={use_backbone_projector}")
        
        # Load Ark backbone
        backbone = load_prtrained_ark_model(
            checkpoint_path=ark_checkpoint_path,
            num_classes=num_classes,
            img_size=768,
            patch_size=4,
            window_size=12,
            embed_dim=192,
            depths=(2, 2, 18, 2),
            num_heads=(6, 12, 24, 48),
            projector_features=1376,  # Ark default projector features
            use_mlp=False,
            device=accelerator.device
        )
        model = ArkClassifier(backbone, 
                               num_classes=num_classes, 
                               multi_view=args.multi_view, 
                               num_views=multi_view_config.num_views if multi_view_config else None,
                               view_fusion_type=multi_view_config.view_fusion_type if multi_view_config else None,
                               adapter_dim=multi_view_config.adapter_dim if multi_view_config else None,
                               view_fusion_hidden_dim=multi_view_config.view_fusion_hidden_dim if multi_view_config else None,
                               use_backbone_projector=use_backbone_projector)
    elif args.model == "medsiglip":
        backbone = load_pretrained_model(model_repo)
        model = MedSigClassifier(backbone, 
                               num_classes=num_classes, 
                               multi_view=args.multi_view, 
                               num_views=multi_view_config.num_views if multi_view_config else None,
                               view_fusion_type=multi_view_config.view_fusion_type if multi_view_config else None,
                               adapter_dim=multi_view_config.adapter_dim if multi_view_config else None,
                               view_fusion_hidden_dim=multi_view_config.view_fusion_hidden_dim if multi_view_config else None)
    else:
        backbone = load_pretrained_model(model_repo)
        model = DinoClassifier(backbone, 
                               num_classes=num_classes, 
                               multi_view=args.multi_view, 
                               num_views=multi_view_config.num_views if multi_view_config else None,
                               view_fusion_type=multi_view_config.view_fusion_type if multi_view_config else None,
                               adapter_dim=multi_view_config.adapter_dim if multi_view_config else None,
                               view_fusion_hidden_dim=multi_view_config.view_fusion_hidden_dim if multi_view_config else None)

    # Loss function and evaluation metrics setup
    if args.weighted_loss:
        weights = get_class_weights(args.task, dataset)
        logger.info(f"Using weighted loss with weights: {weights}")
    else:
        weights = None
    criterion = get_criterion(args.task, weights, device=accelerator.device)
    eval_metrics = get_eval_metrics(args.task, num_classes, accelerator.device)
                
    return {
        "data_loader": data_loader, 
        "model": model, 
        "loss_function": criterion, 
        "eval_metrics": {"classification": eval_metrics}, 
        "train_config": train_config
    }

def train_model(args, checkpoint_dir, accelerator: Accelerator):
    """Train model with k-fold or single-split, returning the best model."""
    cfg = setup(args, accelerator)
    data_loader  = cfg["data_loader"]
    model   = cfg["model"]
    criterion    = cfg["loss_function"]
    eval_metrics = cfg["eval_metrics"]
    train_config = cfg["train_config"]
    num_epochs = train_config.epochs
    batch_size = train_config.batch_size
    
    # Prepare fold loaders (single-split as a single "fold")
    is_kfold = isinstance(data_loader, list)
    fold_loaders = data_loader if is_kfold else [data_loader]
    if accelerator.is_main_process:
        logger.info(f"Starting {'k-fold' if is_kfold else 'single-split'} training with {len(fold_loaders)} fold(s)")
    
    # Early stopping callback setup
    early_stopper_config = train_config.early_stopping
    patience = early_stopper_config.patience if early_stopper_config else None

    # Initialize wandb
    if accelerator.is_main_process:
        wandb_name = f"{CURR_TIME}_{args.data}_{args.model}"
        if args.progressive_unfreeze:
            wandb_name += "_progressive_unfreeze"
        elif args.unfreeze_backbone:
            wandb_name += "_unfreeze_backbone"
        if is_kfold:
            wandb_name += f"_kfold"
        if args.weighted_loss:
            wandb_name += "_weighted_loss"
        if args.resume:
            wandb_name += "_resume"
        wandb.init(project="cxr_models_benchmark", name=wandb_name, config={"epochs": num_epochs, "batch_size": batch_size})
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        eval_metrics=eval_metrics,
        train_config=train_config,
        accelerator=accelerator,
        checkpoint_dir=checkpoint_dir,
        args=args
    )
    
    # Train and get best model
    best_model = trainer.train(fold_loaders, is_kfold, patience)
    
    # Export to ONNX if on main process
    if accelerator.is_main_process:
        trainer.export_onnx(best_model, args.model)
    
    return best_model

def main(args):
    if args.task not in ["multilabel", "multiclass", "binary"]:
        raise NotImplementedError(f"Task {args.task} not supported.")
    if args.kfold is not None and args.kfold < 1:
        raise ValueError("`--kfold` must be greater than 0.")
    if args.resume and args.resume_checkpoint_path is None:
        raise ValueError("When `--resume` is specified, `--resume-checkpoint-path` must also be specified.")
    if not args.unfreeze_backbone and args.unfreeze_num_layers is not None:
        raise ValueError("When `--unfreeze-num-layers` is specified, `--unfreeze-backbone` must also be specified.")
    if not args.unfreeze_backbone and args.progressive_unfreeze:
        raise ValueError("When `--progressive-unfreeze` is specified, `--unfreeze-backbone` must also be specified.")
    if args.model == "ark" and args.pretrained_ark_path is None:
        raise ValueError("Ark checkpoint path must be specified for Ark model. Use --pretrained-ark-path argument.")
    accelerator = Accelerator(mixed_precision="fp16" if args.optimize_compute else "no",
                              gradient_accumulation_steps=2)
    
    # Create both output and checkpoint directories
    checkpoint_folder_name = f"checkpoints_{CURR_TIME}_{args.data}_{args.model}"
    if args.progressive_unfreeze:
        checkpoint_folder_name += "_progressive_unfreeze"
    elif args.unfreeze_backbone:
        checkpoint_folder_name += "_unfreeze_backbone"
    if args.multi_view:
        checkpoint_folder_name += "_multi_view"
    output_dir = os.path.join(CURR_DIR, args.output_dir)
    checkpoint_dir = os.path.join(output_dir, checkpoint_folder_name)
    if args.resume_checkpoint_path:
        checkpoint_dir = os.path.join(output_dir, args.resume_checkpoint_path)
    
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # Run the training loop and return the best model
    best_model = train_model(args, checkpoint_dir, accelerator)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

