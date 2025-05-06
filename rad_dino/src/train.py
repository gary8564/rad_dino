import argparse
import logging
import yaml
import os
import copy
import math
import tqdm
from torchmetrics.classification import Accuracy, AUROC, AveragePrecision, F1Score
import torch 
import torch.nn as nn 
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import AutoModel
from transformers import get_cosine_schedule_with_warmup
import wandb
from rad_dino.src.data import VinDrCXR_Dataset
from rad_dino.src.utils import get_transforms, collate_fn
from rad_dino.src.model import DinoClassifier
from logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 linear probling", add_help=add_help)
    parser.add_argument('--task', type=str, default="multilabel", choices=['multilabel', 'multiclass', 'binary', 'regression', 'ordinal', 'segmentation', 'text_generation'])
    parser.add_argument('--model', type=str, default='rad_dino', choices=['rad_dino', 'dinov2']) 
    parser.add_argument(
        "--unfreeze-backbon",
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
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    return parser
    
def load_data(class_labels, batch_size, train_transforms, val_transforms, num_workers):
    full_ds = VinDrCXR_Dataset("train", class_labels=class_labels, transform=None)
    n_total_samples, n_train = len(full_ds), int(0.8*len(full_ds))
    perm = torch.randperm(n_total_samples)
    train_idx, val_idx = perm[:n_train].tolist(), perm[n_train:].tolist()
    train_ds = Subset(
        VinDrCXR_Dataset("train", class_labels=class_labels, transform=train_transforms),
        train_idx
    )
    val_ds   = Subset(
        VinDrCXR_Dataset("train", class_labels=class_labels, transform=val_transforms),
        val_idx
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    return train_loader, val_loader

def load_pretrained_model(model_repo):
    return AutoModel.from_pretrained(model_repo)

def setup(args):
    # Configuration settings
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    train_config_path = os.path.join(curr_dir, "../configs/config.yaml")
    with open(train_config_path, 'r') as file:
        train_config = yaml.safe_load_all(file)
    class_labels = train_config["data"]["labels"]
    model_repo = "microsoft/rad-dino" if args.model == "rad_dino" else "facebook/dinov2-base"
    train_transforms, val_transforms = get_transforms(model_repo)
   
    # Data setup
    batch_size = train_config["data"]["batch_size"]
    num_workers = train_config["data"]["num_workers"]
    train_loader, val_loader = load_data(class_labels, batch_size, train_transforms, val_transforms, num_workers)
    
    # Model setup
    backbone = load_pretrained_model(model_repo)
    model = DinoClassifier(backbone, num_classes=len(class_labels))
    
    # Training setup
    num_epochs = train_config["optim"]["epochs"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["optim"]["base_lr"], weight_decay=train_config["optim"]["weight_decay"])
    lr_scheduler = None
    
    # Loss function and evaluation metrics setup
    if args.task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
        acc = Accuracy(task="multiclass", num_classes=len(class_labels))
        top5_acc = acc = Accuracy(task="multiclass", num_classes=len(class_labels), top_k=5)
        auroc = AUROC(task="multiclass", num_classes=len(class_labels), average="macro", thresholds=None)
        ap = AveragePrecision(task="multiclass", num_classes=len(class_labels), average="macro", thresholds=None)
        f1_score = F1Score(task="multiclass", num_classes=len(class_labels))
    elif args.task == "multilabel":
        criterion = torch.nn.BCEWithLogitsLoss()
        acc = Accuracy(task="multilabel", num_labels=len(class_labels))
        top5_acc = None
        auroc = AUROC(task="multilabel", num_labels=len(class_labels), average="macro", thresholds=None)
        ap = AveragePrecision(task="multilabel", num_labels=len(class_labels), average="macro", thresholds=None)
        f1_score = F1Score(task="multilabel", num_labels=len(class_labels))
    elif args.task == "binary":
        criterion = torch.nn.BCEWithLogitsLoss()
        acc = Accuracy(task="binary")
        top5_acc = Accuracy(task="binary", top_k=5)
        auroc = AUROC(task="binary")
        ap = AveragePrecision(task="binary")
        f1_score = F1Score(task="binary")
        acc = Accuracy(task="binary")
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
    eval_metrics = {"classification": {"acc": acc, 
                                       "top5_acc": top5_acc, 
                                       "auroc": auroc, 
                                       "ap": ap,
                                       "f1_score": f1_score},
                    "regression": None,
                    "ordinal": None,
                    "text_generation": None}
    
    # Optional setup
    # 1) Learning rate scheduling
    if train_config["optim"]["lr_scheduler"]:
        num_training_steps = len(train_loader) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    # 2) Unfreeze parts of the backbone model weights
    if args.unfreeze_backbone:
        for name, param in model.backbone.named_parameters():
            if 'blocks.10' in name or 'blocks.11' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    return train_loader, val_loader, model, optimizer, lr_scheduler, criterion, eval_metrics, train_config

def train_per_epoch(curr_epoch, model, data_loader, optimizer, scheduler, criterion, eval_metrics, device, optimize_compute=False):
    n_steps_per_epoch = math.ceil(len(data_loader.dataset) / wandb.config.batch_size)
    scaler = GradScaler(device=device)
    model.train()
    running_loss = 0.0
    total_loss = 0
    for i, data in enumerate(tqdm(data_loader)):
        images, labels = data 
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device, enabled=optimize_compute):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if optimize_compute:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        acc_metric = eval_metrics["classification"]["acc"]
        acc = acc_metric(labels, labels)
    
        if i % 10 == 0:
            wandb.log({
                "train/train_loss": loss.item(),
                "train/train_acc": acc.item(),
                "train/step": (i + 1 + (n_steps_per_epoch * curr_epoch)) / n_steps_per_epoch,
                "lr": lr_scheduler.get_last_lr()[0]
            })
    return running_loss / len(data_loader), acc.item()

def eval_per_epoch(model, data_loader, criterion, eval_metrics, device):
    model.eval()
    test_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            test_loss += loss.item()
            preds.append(predictions.cpu())
            trues.append(labels.cpu())
    preds = torch.cat(preds)
    trues = torch.cat(trues)
    # evaluation metric
    acc_metric = eval_metrics["classification"]["acc"]
    f1_score_metric = eval_metrics["classification"]["f1_score"]
    ap_metric = eval_metrics["classification"]["ap"]
    loss_per_epoch = test_loss / len(data_loader)
    acc = acc_metric(trues, preds)
    f1_score = f1_score_metric(trues, preds)
    ap = ap_metric(trues, preds)
    wandb.log({"val/val_loss": loss_per_epoch, 
               "val/val_accuracy": acc,
               "val/val_F1_score": f1_score,
               "val/val_AP": ap,})
    return loss_per_epoch, acc.item(), f1_score.item(), ap.item()

def train_model(args):
    train_loader, val_loader, model, optimizer, lr_scheduler, criterion, eval_metrics, train_config = setup(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimize_compute = args.optimize_compute
    model.to(device)
    num_epochs = train_config["optim"]["epochs"]
    batch_size = train_config["data"]["batch_size"]
    wandb.init(project="dinov2-linear-probe", config={"epochs": num_epochs, "batch_size": batch_size})
    best_metric = -float("inf")  # we want to maximize AUPRC
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    
    if args.resume:
        # If resume training from checkpoints
        ckpt_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if lr_scheduler and ckpt.get("scheduler_state") is not None:
                lr_scheduler.load_state_dict(ckpt["scheduler_state"])
            best_metric = ckpt.get("best_metric", best_metric)
            start_epoch = ckpt.get("epoch", 0)
            logger.info(f"Resuming from {ckpt_path} at epoch {start_epoch} with best_metric={best_metric:.4f}")
        else:
            logger.warning(f"`--resume` is set but no checkpoint found at {ckpt_path}, starting from scratch.")
    
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = train_per_epoch(epoch, model, train_loader, optimizer, lr_scheduler, criterion, eval_metrics, device, optimize_compute)
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss:.3f} \t\t Top1 Acc: {train_acc:.3f}')
        val_loss, val_acc, val_f1, val_ap = eval_per_epoch(model, val_loader, criterion, eval_metrics, device)
        print(f"Epoch {epoch+1} \t\t loss {val_loss:.3f} \t\t AUPRC {val_ap:.3f}  \t\t ")
        # Save per‑epoch
        torch.save({
            "epoch": epoch+1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
            "best_metric": best_metric,
        }, os.path.join(checkpoint_dir, f"epoch{epoch + 1: 02d}.pt"))
        logger.info(f"Saved epoch checkpoint to epoch{epoch+1:02d}.pt")
        # Check for best
        if val_ap > best_metric:
            best_metric = val_ap
            torch.save({
                "epoch": epoch+1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": lr_scheduler.state_dict() if lr_scheduler else None,
                "best_metric": best_metric,
            }, os.path.join(checkpoint_dir, "best.pt"))
            logger.info(f"New best AUPRC = {best_metric:.4f} at epoch {epoch+1}, saved best.pt")
            best_model = copy.deepcopy(model).eval()
    return best_model

def main(args):
    # Run the training loop (with resume logic baked in) and return the best model
    best_model = train_model(args)
    
    # Save the model (TorchScript)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_model.to(device)
    try:
        best_model_scripted = torch.jit.script(best_model)
    except Exception:
        dummy = torch.randn(1, 3, 518, 518, device=device)
        with torch.no_grad():
            best_model_scripted = torch.jit.trace(best_model, (dummy,), strict=False)

    # Save the scripted model
    out_path = os.path.join(args.output_dir, f"{args.model}_final_scripted.pt")
    best_model_scripted.save(out_path)
    logger.info(f"Saved final scripted model to {out_path}")
    
if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

