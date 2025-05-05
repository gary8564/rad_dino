import argparse
import logging
import yaml
import os
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
from sklearn.metrics import average_precision_score
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
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
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
    return train_loader, val_loader, model, optimizer, lr_scheduler, num_epochs

def train_per_epoch(model, loader, optimizer, device, optimize_compute=False):
    scaler = GradScaler(device=device)
    model.train()
    running_loss = 0.0
    total_loss = 0
    for i, (images, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device, enabled=optimize_compute):
            logits = model(images)
            loss = criterion(outputs, labels)
            loss = nn.BCEWithLogitsLoss()(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        # Top-1 / Top-5 accuracy
        _, pred = outputs.topk(5, 1, True, True)
        correct = pred.eq(labels.view(-1, 1).expand_as(pred))
        correct1 += correct[:, 0].sum().item()
        correct5 += correct.sum().item()
        total += labels.size(0)
        running_loss += loss.item()

        if i % 10 == 0:
            wandb.log({
                "loss": loss.item(),
                "top1_acc": correct1 / total,
                "top5_acc": correct5 / total,
                "lr": lr_scheduler.get_last_lr()[0]
            })

            
        logits = model(pixel_values=imgs)
        loss = nn.BCEWithLogitsLoss()(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_per_epoch(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(pixel_values=imgs)
            preds.append(logits.softmax(1)[:,1].cpu())
            trues.append(labels.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    # compute AUPRC or any metric you like
    
    return average_precision_score(trues, preds)

def train_model(args):
    train_loader, val_loader, model, optimizer, lr_scheduler, num_epochs = setup(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimize_compute = args.optimize_compute
    if args.task == "multiclass":
        loss_func = torch.nn.CrossEntropyLoss()
    elif args.task == "multilabel":
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif args.task == "binary":
        loss_func = torch.nn.BCEWithLogitsLoss()
    elif args.task == "regression":
        loss_func = torch.nn.MSELoss()
    elif args.task == "ordinal":
        NotImplementedError
    elif args.task == "segmentation":
        NotImplementedError
    elif args.task == "text_generation":
        NotImplementedError
    else:
        raise ValueError("Unknown task: task must be 'multilabel', 'multiclass', 'binary', 'regression', 'ordinal', 'segmentation', or 'text_generation'.")
    model.to(device)
    wandb.init(project="dinov2-linear-probe", config={"epochs": num_epochs})
    for epoch in range(num_epochs):
        train_loss = train_per_epoch(model, train_loader, optimizer, device, optimize_compute)
        val_score = eval_per_epoch(model, val_loader, device)
        print(f"Epoch {epoch}: loss {train_loss:.3f}, AUPRC {val_score:.3f}")
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Top1: {correct1/total:.4f} | Top5: {correct5/total:.4f}")


def main(args):
    pass
    

if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)

