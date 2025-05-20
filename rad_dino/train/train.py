import argparse
import logging
import yaml
import os
import copy
import math
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from torchmetrics.classification import Accuracy, AUROC, AveragePrecision, F1Score
import torch 
from accelerate import Accelerator
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast, GradScaler
import torchvision.transforms.v2 as transforms
from transformers import AutoModel
from transformers import get_cosine_schedule_with_warmup
import wandb
from pydantic import BaseModel, Field
from data.VinDrCXR.data import VinDrCXR_Dataset
from utils.utils import get_transforms, EarlyStopping, collate_fn
from models.model import DinoClassifier
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

CURR_DIR = os.path.dirname(os.path.realpath(__file__))
CURR_TIME = datetime.now().strftime("%Y_%m_%d_%H%M%S")
# Define model-specific input sizes
MODEL_INPUT_SIZES = {
    "rad_dino": (518, 518),  # RadDINO expects 518x518
    "dinov2": (224, 224)     # DINOv2 expects 224x224
}

class OptimizerConfig(BaseModel):
    base_lr: float = Field(..., description="Base learning rate")
    weight_decay: float = Field(..., description="Weight decay for optimizer")

class LRSchedulerConfig(BaseModel):
    warmup_ratio: float = Field(..., description="Ratio of warmup steps to total steps")

class EarlyStoppingConfig(BaseModel):
    patience: int = Field(..., description="Number of epochs to wait before early stopping")
    min_delta: float = Field(default=0.0, description="Minimum change in metric to qualify as improvement")
    mode: str = Field(default="max", description="Metric optimization mode ('min' or 'max')")

class TrainConfig(BaseModel):
    batch_size: int = Field(..., description="Batch size for training")
    epochs: int = Field(..., description="Number of training epochs")
    optim: OptimizerConfig
    lr_scheduler: Optional[LRSchedulerConfig] = None
    early_stopping: Optional[EarlyStoppingConfig] = None

class DataConfig(BaseModel):
    data_root_folder: str = Field(..., description="Root folder containing the dataset")
    class_labels: List[str] = Field(..., description="List of class labels")
    num_workers: int = Field(..., description="Number of workers for data loading")

@dataclass
class KFold:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    checkpoint_dir: str
    best_checkpoint: str
    best_metric: float
    start_epoch: int
    train_loader: DataLoader
    val_loader: DataLoader

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 linear probling", add_help=add_help)
    parser.add_argument('--task', type=str, default="multilabel", choices=['multilabel', 'multiclass', 'binary', 'regression', 'ordinal', 'segmentation', 'text_generation'])
    parser.add_argument('--data', type=str, default='VinDr-CXR', choices=['VinDr-CXR', 'CANDID-PTX', 'RSNA-Pneumonia', 'VinDr-Mammography'])
    parser.add_argument('--model', type=str, default='rad_dino', choices=['rad_dino', 'dinov2']) 
    parser.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="Whether to unfreeze the last 2 transformer blocks.")
    parser.add_argument(
        "--optimize-compute",
        action="store_true",
        help="Whether to use advanced tricks to lessen the heavy computational resource. ",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--output-dir",
        default="../../runs/",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    return parser
    
def load_data(data_root_folder: str,
              class_labels: list, 
              batch_size: int, 
              train_transforms: transforms.Compose, 
              val_transforms: transforms.Compose, 
              num_workers: int, 
              kfold: int | None = None,
              random_split_ratio: float = 0.9,
              seed: int = 42,):
    base_ds = VinDrCXR_Dataset(data_root_folder, "train", class_labels=class_labels, transform=None)
    if kfold and kfold > 1:
        Y = base_ds.df.to_numpy(dtype=np.int64)
        mskf = MultilabelStratifiedKFold(
            n_splits=kfold, shuffle=True, random_state=seed
        )
        fold_loaders = []
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X=base_ds.image_ids, y=Y), start=1):
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(f"Fold {fold} has empty train or validation set")
            logger.info(f"[KFold] Fold {fold}: {len(train_idx)} train / {len(val_idx)} val")

            # re‐instantiate your dataset with transforms and subset by indices
            train_ds = Subset(
                VinDrCXR_Dataset(data_root_folder, "train", class_labels=class_labels, transform=train_transforms),
                train_idx
            )
            val_ds = Subset(
                VinDrCXR_Dataset(data_root_folder, "train", class_labels=class_labels, transform=val_transforms),
                val_idx
            )

            train_loader = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn
            )
            val_loader = DataLoader(
                val_ds, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn
            )

            fold_loaders.append((train_loader, val_loader))

        return fold_loaders
    
    n_total_samples = len(base_ds)
    n_train = int(random_split_ratio * n_total_samples)
    if n_train == 0 or n_train == n_total_samples:
        raise ValueError("Train or validation set is empty due to small dataset size")
    torch.manual_seed(seed)
    perm = torch.randperm(n_total_samples).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_ds = Subset(
        VinDrCXR_Dataset(data_root_folder, "train", class_labels=class_labels, transform=train_transforms),
        train_idx
    )
    val_ds = Subset(
        VinDrCXR_Dataset(data_root_folder, "train", class_labels=class_labels, transform=val_transforms),
        val_idx
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    return train_loader, val_loader

def load_pretrained_model(model_repo):
    return AutoModel.from_pretrained(model_repo)

def setup(args, accelerator: Accelerator):
    # Configuration settings
    data_config_path = os.path.join(CURR_DIR, "../configs/data_config.yaml")
    train_config_path = os.path.join(CURR_DIR, "../configs/train_config.yaml")
    
    # Load and validate configurations using Pydantic
    with open(data_config_path, 'r') as file:
        data_config_raw = yaml.safe_load(file)
    with open(train_config_path, 'r') as file:
        train_config_raw = yaml.safe_load(file)
    
    # Select the appropriate dataset configuration
    if args.data == "VinDr-CXR":
        data_config_raw = data_config_raw["VinDr-CXR"]
    elif args.data == "CANDID-PTX":
        pass
    elif args.data == "RSNA-Pneumonia":
        pass
    elif args.data == "VinDr-Mammography":
        pass
    else:
        raise NotImplementedError(f"Dataset {args.data} is not supported")
    
    # Validate configurations
    try:
        data_config = DataConfig(**data_config_raw)
        train_config = TrainConfig(**train_config_raw)
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise
    
    class_labels = data_config.class_labels
    if not class_labels:
        raise ValueError("Class labels cannot be empty")
    
    num_workers = data_config.num_workers
    model_repo = "microsoft/rad-dino" if args.model == "rad_dino" else "facebook/dinov2-base"
    train_transforms, val_transforms = get_transforms(model_repo)
   
    # Data setup
    batch_size = train_config.batch_size
    data_root_folder = data_config.data_root_folder
    data_loader = load_data(data_root_folder, class_labels, batch_size, train_transforms, val_transforms, num_workers)

    # Model setup
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_pretrained_model(model_repo)
    model = DinoClassifier(backbone, num_classes=len(class_labels))#.to(device)

    # Loss function and evaluation metrics setup
    if args.task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
        acc = Accuracy(task="multiclass", num_classes=len(class_labels))#.to(device)
        top5_acc = Accuracy(task="multiclass", num_classes=len(class_labels), top_k=5)#.to(device)
        auroc = AUROC(task="multiclass", num_classes=len(class_labels), average="macro", thresholds=None)#.to(device)
        ap = AveragePrecision(task="multiclass", num_classes=len(class_labels), average="macro", thresholds=None)#.to(device)
        f1_score = F1Score(task="multiclass", num_classes=len(class_labels))#.to(device)
    elif args.task == "multilabel":
        criterion = torch.nn.BCEWithLogitsLoss()
        acc = Accuracy(task="multilabel", num_labels=len(class_labels))#.to(device)
        top5_acc = None
        auroc = AUROC(task="multilabel", num_labels=len(class_labels), average="macro", thresholds=None)#.to(device)
        ap = AveragePrecision(task="multilabel", num_labels=len(class_labels), average="macro", thresholds=None)#.to(device)
        f1_score = F1Score(task="multilabel", num_labels=len(class_labels))#.to(device)
    elif args.task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
        acc = Accuracy(task="binary")#.to(device)
        top5_acc = Accuracy(task="binary", top_k=5)#.to(device)
        auroc = AUROC(task="binary")#.to(device)
        ap = AveragePrecision(task="binary")#.to(device)
        f1_score = F1Score(task="binary")#.to(device)
    elif args.task == "regression":
        criterion = torch.nn.MSELoss()
    elif args.task == "ordinal":
        NotImplementedError
    elif args.task == "segmentation":
        NotImplementedError
    elif args.task == "text_generation":
        NotImplementedError
    else:
        raise ValueError("Unknown task: task must be 'multilabel', 'multiclass', 'binary', 'regression', 'ordinal', 'segmentation', or 'text_generation'.")
    
    # Move metrics to the correct device using accelerator
    metrics = {
        "acc": acc,
        "top5_acc": top5_acc,
        "auroc": auroc,
        "ap": ap,
        "f1_score": f1_score
    }
    
    # Move each metric to the correct device
    for key, metric in metrics.items():
        if metric is not None:
            metrics[key] = metric.to(accelerator.device)
    
    eval_metrics = {
        "classification": metrics,
        "regression": None,
        "ordinal": None,
        "text_generation": None
    }
                
    return {
        "data_loader": data_loader, 
        "model": model, 
        "loss_function": criterion, 
        "eval_metrics": eval_metrics, 
        "train_config": train_config
    }

def train_per_epoch(curr_epoch, model, data_loader, optimizer, scheduler, criterion, eval_metrics, accelerator, train_config, log_prefix): #, optimize_compute=False):
    n_steps_per_epoch = math.ceil(len(data_loader.dataset) / train_config.batch_size)
    model.train()
    # running_loss = 0.0
    running_loss = torch.tensor(0.0, device=accelerator.device)
    
    for i, data in enumerate(tqdm(data_loader, desc=f"Epoch {curr_epoch + 1}")):
        images, labels, _ = data 
        # images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with accelerator.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        # if optimize_compute:
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
        # else:
            # loss.backward()
            # optimizer.step()
        accelerator.backward(loss)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Accumulate loss
        running_loss += loss.item()
        
        # Calculate accuracy and AUC-ROC for logging
        acc_metric = eval_metrics["classification"]["acc"]
        acc = acc_metric(outputs, labels)

        auroc_metric = eval_metrics["classification"]["auroc"]
        auroc = acc_metric(outputs, labels)
        
        if i % 10 == 0 and accelerator.is_main_process:
            current_lr = scheduler.get_last_lr()[0] if scheduler else train_config.optim.base_lr
            wandb.log({
                f"train/{log_prefix}loss_step": loss.item(),
                f"trainer/{log_prefix}global_step": (i + 1 + (n_steps_per_epoch * curr_epoch)) / n_steps_per_epoch,
                f"{log_prefix}lr": current_lr
            })
        
        last_batch_acc = acc.item()
        last_batch_auroc = auroc.item()
    
    # compute local average loss
    avg_loss_local = running_loss / len(data_loader)
    # reduce to get global average loss
    loss_tensor = torch.tensor(avg_loss_local, device=accelerator.device)
    avg_loss = accelerator.reduce(loss_tensor, reduction="mean").item()
    
    # reduce the last‐batch acc and auroc across GPUs (mean)
    acc_tensor = torch.tensor(last_batch_acc, device=accelerator.device)
    avg_acc = accelerator.reduce(acc_tensor, reduction="mean").item()
    auroc_tensor = torch.tensor(last_batch_auroc, device=accelerator.device)
    avg_auroc = accelerator.reduce(auroc_tensor, reduction="mean").item()
    
    return avg_loss, avg_acc, avg_auroc

def eval_per_epoch(model, data_loader, criterion, eval_metrics, accelerator: Accelerator, log_prefix):
    model.eval()
    local_val_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for data in data_loader:
            images, labels, _ = data  
            # images = images.to(device)
            # labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            local_val_loss += loss.item()
            preds.append(predictions) 
            trues.append(labels)      
    
    # compute local average loss
    avg_loss_local = local_val_loss / len(data_loader)
    # reduce to get global average loss
    loss_tensor = torch.tensor(avg_loss_local, device=accelerator.device)
    avg_loss = accelerator.reduce(loss_tensor, reduction="mean").item()
    
    # Concatenate predictions and true labels
    # preds = torch.cat(preds)
    # trues = torch.cat(trues).long()
    preds = accelerator.gather_for_metrics(torch.cat(preds))
    trues = accelerator.gather_for_metrics(torch.cat(trues)).long()
    
    # evaluation metric
    acc_metric = eval_metrics["classification"]["acc"]
    f1_score_metric = eval_metrics["classification"]["f1_score"]
    auroc_metric = eval_metrics["classification"]["auroc"]
    ap_metric = eval_metrics["classification"]["ap"]
    
    if accelerator.is_main_process:
        acc = acc_metric(preds, trues)
        f1_score = f1_score_metric(preds, trues)
        ap = ap_metric(preds, trues)
        auroc = auroc_metric(preds, trues)
        
        wandb.log({
            f"val/{log_prefix}loss_step": avg_loss
        })
    else:
        # dummy tensors so every rank has something to reduce
        acc = torch.tensor(0.0, device=accelerator.device)
        f1_score = torch.tensor(0.0, device=accelerator.device)
        ap = torch.tensor(0.0, device=accelerator.device)
        auroc = torch.tensor(0.0, device=accelerator.device)
    
    # broadcast metrics back to all ranks with mean 
    acc = accelerator.reduce(acc, reduction="mean").item()
    f1_score  = accelerator.reduce(f1_score,  reduction="mean").item()
    ap  = accelerator.reduce(ap,  reduction="mean").item()
    auroc  = accelerator.reduce(auroc,  reduction="mean").item()

    return avg_loss, acc, f1_score, ap, auroc # loss_per_epoch, acc.item(), f1_score.item(), ap.item(), auroc.item()

def initialize_fold(
    args,
    base_model: torch.nn.Module,
    train_config: TrainConfig,
    checkpoint_dir: str,
    fold_idx: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    accelerator: Accelerator #device: str
) -> KFold:
    """Initialize model, optimizer, scheduler, and checkpoint paths for a fold."""
    # Create fresh model copy
    model = copy.deepcopy(base_model)#.to(device)
    
    # Apply backbone unfreezing if specified
    if args.unfreeze_backbone:
        for name, param in model.backbone.named_parameters():
            param.requires_grad = 'blocks.10' in name or 'blocks.11' in name
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.optim.base_lr,
        weight_decay=train_config.optim.weight_decay
    )
    
    # Initialize scheduler if configured
    lr_scheduler = None
    if train_config.lr_scheduler:
        num_training_steps = len(train_loader) * train_config.epochs
        num_warmup_steps = int(train_config.lr_scheduler.warmup_ratio * num_training_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
    
    # Prepare model, optimizer, and scheduler for DDP
    train_loader, val_loader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_loader, val_loader, model, optimizer, lr_scheduler
    )
    
    # Setup checkpoint paths
    fold_checkpoint_dir = os.path.join(checkpoint_dir, f"fold_{fold_idx}" if fold_idx > 0 else "")
    if accelerator.is_main_process:
        os.makedirs(fold_checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()  # Synchronize processes
    best_checkpoint = os.path.join(fold_checkpoint_dir, "best.pt")
    
    # Resume if checkpoint exists
    best_metric = -float("inf")
    start_epoch = 0
    if args.resume:
        if os.path.exists(best_checkpoint):
            if accelerator.is_main_process:
                # Load model checkpoint
                ckpt = torch.load(best_checkpoint, map_location=accelerator.device)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if lr_scheduler and ckpt.get("scheduler_state"):
                    lr_scheduler.load_state_dict(ckpt["scheduler_state"])
                best_metric = ckpt.get("best_metric", best_metric)
                start_epoch = ckpt.get("epoch", 0)
                logger.info(f"Fold {fold_idx if fold_idx > 0 else 'single'}: Resumed from epoch {start_epoch}, best_metric={best_metric:.4f}")
            
            # Synchronize values across processes
            accelerator.wait_for_everyone()
            
            # create tensors on each rank
            start_epoch_tensor = torch.tensor(start_epoch, device=accelerator.device)
            best_metric_tensor = torch.tensor(best_metric, device=accelerator.device)
            
            # reduce to get values from rank 0
            start_epoch = accelerator.reduce(start_epoch_tensor, reduction="mean").item()
            best_metric = accelerator.reduce(best_metric_tensor, reduction="mean").item()
        else:
            raise RuntimeError(f"No checkpoint found at {best_checkpoint}. Cannot resume training.")
    
    return KFold(
        model, optimizer, lr_scheduler, fold_checkpoint_dir,
        best_checkpoint, best_metric, start_epoch, train_loader, val_loader
    )

def train_model(args, checkpoint_dir, accelerator: Accelerator):
    """Train model with k-fold or single-split, returning the best model."""
    cfg = setup(args, accelerator)
    data_loader  = cfg["data_loader"]
    model   = cfg["model"]
    criterion    = cfg["loss_function"]
    eval_metrics = cfg["eval_metrics"]
    train_config = cfg["train_config"]
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    optimize_compute = args.optimize_compute    
    num_epochs = train_config.epochs
    batch_size = train_config.batch_size
    
    # Create checkpoint dir only on main process
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
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
        wandb.init(project="dinov2-linear-probe", name=f"{args.model}_{CURR_TIME}_{args.data}", config={"epochs": num_epochs, "batch_size": batch_size})
    
    # Track best model and fold results
    best_metric_global = -float("inf")  
    best_model_global = None
    kfold_results = []

    # Iterate over folds
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders, start=0):
        fold_idx = fold_idx + 1 if is_kfold else 0  # Use 0 for single-split
        log_prefix = f"Fold {fold_idx} " if is_kfold else ""
        if accelerator.is_main_process:
            logger.info(f"{log_prefix} Training size: {len(train_loader)}  Validation size: {len(val_loader)}")
        
        # Initialize KFold 
        kfold = initialize_fold(
            args,
            model,
            train_config,
            checkpoint_dir,
            fold_idx,
            train_loader,
            val_loader,
            accelerator
        )
        
        # Setup early stopping
        early_stopper = None
        if patience and accelerator.is_main_process:
            early_stopper = EarlyStopping(
                patience=patience,
                min_delta=early_stopper_config.min_delta,
                mode=early_stopper_config.mode,
                ckpt_path=kfold.best_checkpoint
            )
        
        # Training loop
        for epoch in range(kfold.start_epoch, num_epochs):
            train_loss, train_acc, train_auroc = train_per_epoch(
                epoch, kfold.model, kfold.train_loader, kfold.optimizer,
                kfold.lr_scheduler, criterion, eval_metrics, accelerator, train_config, log_prefix #, optimize_compute
            )
            if accelerator.is_main_process:
                print(f'{log_prefix}Epoch {epoch+1} \t\t Train loss: {train_loss:.3f} \t\t Top1 Acc: {train_acc:.3f}')
            
            val_loss, val_acc, val_f1, val_ap, val_auroc = eval_per_epoch(
                kfold.model, kfold.val_loader, criterion, eval_metrics, accelerator, log_prefix
            )
            if accelerator.is_main_process:
                print(f'{log_prefix}Epoch {epoch+1} \t\t Val loss {val_loss:.3f} \t\t AUPRC {val_ap:.3f}')
            
            # Log metrics
            if accelerator.is_main_process:
                wandb.log({
                    f"train/{log_prefix}loss_per_epoch": train_loss,
                    f"train/{log_prefix}ACC": train_acc,
                    f"train/{log_prefix}AUROC": train_auroc,
                    f"val/{log_prefix}loss_per_epoch": val_loss,
                    f"val/{log_prefix}ACC": val_acc,
                    f"val/{log_prefix}F1_Score": val_f1,
                    f"val/{log_prefix}AUPRC": val_ap,
                    f"val/{log_prefix}AUROC": val_auroc,  
                })
            
            # Early stopping
            if early_stopper and accelerator.is_main_process:
                if early_stopper.step(val_ap, kfold.model, kfold.optimizer, kfold.lr_scheduler, epoch + 1):
                    logger.info(f"{log_prefix}Stopping early at epoch {epoch + 1}")
                    break
        
        # Load best model for this fold
        if early_stopper and accelerator.is_main_process:
            best_model = early_stopper.load_best_model(kfold.model)
        else:
            best_model = kfold.model
        
        # Track fold results
        kfold_results.append({
            "fold": fold_idx or "single",
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_ap": kfold.best_metric
        })
        
        # Update global best model
        if kfold.best_metric > best_metric_global and accelerator.is_main_process:
            best_metric_global = kfold.best_metric
            best_model_global = copy.deepcopy(best_model)
            logger.info(f"{log_prefix}New global best model with AUPRC={best_metric_global:.4f}")
    
    # Log k-fold summary
    if is_kfold and accelerator.is_main_process:
        avg_ap = np.mean([res["val_ap"] for res in kfold_results])
        std_ap = np.std([res["val_ap"] for res in kfold_results])
        logger.info(f"K-fold results: Mean AUPRC={avg_ap:.4f} ± {std_ap:.4f}")
        wandb.log({"kfold_mean_ap": avg_ap, "kfold_std_ap": std_ap})
    
    # Ensure we have a best model
    if best_model_global is None and accelerator.is_main_process:
        logger.warning("No best model was found during training. Using the last model.")
        best_model_global = best_model
    # return best_model_global
    return accelerator.unwrap_model(best_model_global) if best_model_global is not None else accelerator.unwrap_model(best_model)

def main(args):
    accelerator = Accelerator(mixed_precision="fp16" if args.optimize_compute else "no")
    if accelerator.is_main_process:
        os.makedirs(os.path.join(CURR_DIR, args.output_dir), exist_ok=True)
    accelerator.wait_for_everyone()
    checkpoint_dir = os.path.join(CURR_DIR, args.output_dir, f"checkpoints_{args.model}_{CURR_TIME}")
    
    # Run the training loop (with resume logic baked in) and return the best model
    best_model = train_model(args, checkpoint_dir, accelerator)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)
    
    input_size = MODEL_INPUT_SIZES.get(args.model, (224, 224))  # Default to 224x224
    dummy = torch.randn(1, 3, *input_size, device=device)
    
    # Save the model (TorchScript)
    if accelerator.is_main_process:
        try:
            best_model_scripted = torch.jit.script(best_model)
        except Exception as e:
            logger.warning(f"Scripting failed: {e}. Falling back to tracing with input size {input_size}.")
            with torch.no_grad():
                best_model_scripted = torch.jit.trace(best_model, (dummy,), strict=False)

    # Save the scripted model
    out_path = os.path.join(checkpoint_dir, f"best_final_scripted.pt")
    best_model_scripted.save(out_path)
    logger.info(f"Saved final scripted model to {out_path}")
    
if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

