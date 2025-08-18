import os
import copy
import logging
import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from torch.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from rad_dino.train.train_utils import EarlyStopping
from rad_dino.utils.cross_validation import KFold
from rad_dino.loggings.setup import init_logging
from rad_dino.train.model_registry import get_model_info, get_layer_term

init_logging()
logger = logging.getLogger(__name__)

# Define model-specific input sizes
MODEL_INPUT_SIZES = {
    "rad-dino": (518, 518),  # RadDINO expects 518x518
    "dinov2-base": (224, 224),     # DINOv2 expects 224x224
    "dinov2-small": (224, 224),     # DINOv2 expects 224x224
    "medsiglip": (448, 448),    # MedSigLIP expects 448x448
    "ark": (768, 768)    # Ark expects 768x768
}


class Trainer:
    def __init__(self, model, criterion, eval_metrics, train_config, accelerator, checkpoint_dir, args):
        self.model = model
        self.criterion = criterion
        self.eval_metrics = eval_metrics
        self.train_config = train_config
        self.accelerator = accelerator
        self.checkpoint_dir = checkpoint_dir
        self.args = args
        
    def _freeze_backbone(self, model):
        """Freeze the backbone of the model."""
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
            
    def _unfreeze_all_backbone(self, model):
        """Unfreeze all backbone parameters."""
        for name, param in model.backbone.named_parameters():
            param.requires_grad = True
        
    def _unfreeze_last_n_blocks(self, model, n):
        """Unfreeze the last n layers of the backbone.
        
        Args:
            model (torch.nn.Module): The model containing the backbone
            n (int): Number of layers to unfreeze from the end
        """
        # Detect model type and get layer information
        model_info = self._get_model_layer_info(model)
        model_type = model_info['model_type']
        total_layers = model_info['total_layers']
        layer_pattern = model_info['layer_pattern']
        
        if n > total_layers or n < 1:
            raise ValueError(f"Number of unfreeze layers {n} cannot be greater than the total number of layers {total_layers} or less than 1")
        
        # First freeze all backbone parameters
        self._freeze_backbone(model)
        
        # Then unfreeze the specified layers from the end
        for i in range(total_layers - 1, total_layers - n - 1, -1):
            for name, param in model.backbone.named_parameters():
                if layer_pattern.format(i) in name:
                    logger.info(f"Unfreezing {model_type} backbone parameter: {name}")
                    param.requires_grad = True

    def _get_model_layer_info(self, model):
        """Get layer information for different model types using the model registry.
        
        Args:
            model: The model to analyze
            
        Returns:
            dict: Contains model type, total layers, and layer pattern
        """
        return get_model_info(model, self.args.model)

    def _progressive_unfreeze(self, model, current_epoch):
        """
        Progressively unfreeze layers of the backbone based on training progress every two epochs,
        starting from the classification head and moving backwards. 
        """
        # Get model information
        model_info = self._get_model_layer_info(model)
        model_type = model_info['model_type']
        total_layers = model_info['total_layers']
        
        # Ensure backbone is frozen initially
        if current_epoch == 0:
            self._freeze_backbone(model)
            return
        
        # Only unfreeze at even epochs (2, 4, 6, ...)
        if current_epoch % 2 == 0:
            layers_to_unfreeze = current_epoch // 2
            layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
            
            # Get appropriate layer term from the model registry
            layer_term = get_layer_term(model, self.args.model)
            logger.info(f"Progressive unfreezing: Unfreezing {layers_to_unfreeze} {layer_term} at epoch {current_epoch}")
            self._unfreeze_last_n_blocks(model, layers_to_unfreeze)
            
        
    def _apply_unfreezing_strategy(self, model: torch.nn.Module, current_epoch: int):
        """
        Apply the appropriate unfreezing strategy based on args.
        """
        # Get model information
        model_info = self._get_model_layer_info(model)
        model_type = model_info['model_type']
        
        if self.args.unfreeze_backbone:
            if self.args.unfreeze_num_layers is not None:
                # Get appropriate layer term from the model registry
                layer_term = get_layer_term(model, self.args.model)
                logger.info(f"Unfreezing {self.args.unfreeze_num_layers} {layer_term} from the end of the {model_type} backbone.")
                self._unfreeze_last_n_blocks(model, n=self.args.unfreeze_num_layers)
            elif self.args.progressive_unfreeze:
                self._progressive_unfreeze(model, current_epoch)
            else:
                logger.info(f"Unfreezing all layers of the {model_type} backbone.")
                self._unfreeze_all_backbone(model)
        else:
            logger.info(f"{model_type} backbone is frozen.")
            self._freeze_backbone(model)

    def _get_parameter_groups(self, model: torch.nn.Module) -> tuple[list, list]:
        """Separate model parameters into backbone and head parameters.
        
        Args:
            model
            
        Returns:
            tuple (backbone_params, head_params)
        """
        # Debugging: check all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"Trainable parameter: {name}")
            
        # Collect parameters based on their requires_grad status
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'backbone' in name:
                if param.requires_grad:
                    backbone_params.append(param)
            else:
                param.requires_grad = True
                head_params.append(param)
                
        return backbone_params, head_params

    def _update_optimizer_parameter_groups(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module):
        """
        Update optimizer parameter groups for progressive unfreezing.
        """
        backbone_params, head_params = self._get_parameter_groups(model)
        
        # Get existing parameters in optimizer
        existing_params = {p for g in optimizer.param_groups for p in g['params']}
        
        new_backbone_params = [p for p in backbone_params if p not in existing_params]
        
        if new_backbone_params:
            logger.info(f"Adding {len(new_backbone_params)} new backbone parameters to optimizer")
            optimizer.add_param_group({
                'params': new_backbone_params,
                'lr': self.train_config.optim.base_lr * 0.1
            })
        else:
            logger.debug("No new backbone parameters to add to optimizer")

    def train_per_epoch(self, curr_epoch, model, data_loader, optimizer, scheduler, log_prefix):
        n_steps_per_epoch = math.ceil(len(data_loader.dataset) / self.train_config.batch_size)
        model.train()
        running_loss = torch.tensor(0.0, device=self.accelerator.device)
        # Accumulate predictions and labels for epoch-level metrics
        preds, trues = [], []
        # Enable anomaly detection for debugging NaN issues
        # torch.autograd.set_detect_anomaly(True)
        
        for i, data in enumerate(tqdm(data_loader, desc=f"Epoch {curr_epoch + 1}")):
            # Backpropagate the loss and accumulate gradients
            with self.accelerator.accumulate(model):
                images = data["pixel_values"]
                labels = data["labels"]
                
                # Forward pass with mixed precision
                with self.accelerator.autocast():
                    outputs, _ = model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass and gradient clipping
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:  # Only clip when gradients are synced
                    self.accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Accumulate predictions and labels for epoch-level metrics
            preds.append(outputs.detach())
            trues.append(labels.detach())
            
            if i % 10 == 0 and self.accelerator.is_main_process:
                current_lr = scheduler.get_last_lr()[0] if scheduler else self.train_config.optim.base_lr
                wandb.log({
                    f"train/{log_prefix}loss_step": loss.item(),
                    f"trainer/{log_prefix}global_step": (i + 1 + (n_steps_per_epoch * curr_epoch)) / n_steps_per_epoch,
                    f"{log_prefix}lr": current_lr
                })
        
        # compute local average loss
        avg_loss_local = running_loss / len(data_loader)
        # reduce to get global average loss
        loss_tensor = torch.tensor(avg_loss_local, device=self.accelerator.device)
        avg_loss = self.accelerator.reduce(loss_tensor, reduction="mean").item()
        
        # Compute epoch-level metrics on all accumulated predictions and labels
        preds = self.accelerator.gather_for_metrics(torch.cat(preds))
        trues = self.accelerator.gather_for_metrics(torch.cat(trues)).long()
        
        # Calculate metrics on the entire epoch
        acc_metric = self.eval_metrics["classification"]["acc"]
        auroc_metric = self.eval_metrics["classification"]["auroc"]
        
        if self.accelerator.is_main_process:
            avg_acc = acc_metric(preds, trues)
            avg_auroc = auroc_metric(preds, trues)
        else:
            # dummy tensors so every rank has something to reduce
            avg_acc = torch.tensor(0.0, device=self.accelerator.device)
            avg_auroc = torch.tensor(0.0, device=self.accelerator.device)
        
        # Broadcast metrics to all processes
        avg_acc = self.accelerator.reduce(avg_acc, reduction="mean").item()
        avg_auroc = self.accelerator.reduce(avg_auroc, reduction="mean").item()
        
        return avg_loss, avg_acc, avg_auroc

    def eval_per_epoch(self, model, data_loader):
        model.eval()
        local_val_loss = 0
        preds, trues = [], []
        with torch.no_grad():
            for data in data_loader:
                images = data["pixel_values"]
                labels = data["labels"]
                predictions, _ = model(images)
                loss = self.criterion(predictions, labels)
                local_val_loss += loss.item()
                preds.append(predictions.detach())
                trues.append(labels.detach())
        
        # compute local average loss
        avg_loss_local = local_val_loss / len(data_loader)
        # reduce to get global average loss
        loss_tensor = torch.tensor(avg_loss_local, device=self.accelerator.device)
        avg_loss = self.accelerator.reduce(loss_tensor, reduction="mean").item()
        
        # Concatenate predictions and true labels
        preds = self.accelerator.gather_for_metrics(torch.cat(preds))
        trues = self.accelerator.gather_for_metrics(torch.cat(trues)).long()
        
        # evaluation metric
        acc_metric = self.eval_metrics["classification"]["acc"]
        f1_score_metric = self.eval_metrics["classification"]["f1_score"]
        auroc_metric = self.eval_metrics["classification"]["auroc"]
        ap_metric = self.eval_metrics["classification"]["ap"]
        
        if self.accelerator.is_main_process:
            acc = acc_metric(preds, trues)
            f1_score = f1_score_metric(preds, trues)
            ap = ap_metric(preds, trues)
            auroc = auroc_metric(preds, trues)
        else:
            # dummy tensors so every rank has something to reduce
            acc = torch.tensor(0.0, device=self.accelerator.device)
            f1_score = torch.tensor(0.0, device=self.accelerator.device)
            ap = torch.tensor(0.0, device=self.accelerator.device)
            auroc = torch.tensor(0.0, device=self.accelerator.device)
        
        # broadcast metrics back to all ranks with mean 
        acc = self.accelerator.reduce(acc, reduction="mean").item()
        f1_score = self.accelerator.reduce(f1_score, reduction="mean").item()
        ap = self.accelerator.reduce(ap, reduction="mean").item()
        auroc = self.accelerator.reduce(auroc, reduction="mean").item()

        return avg_loss, acc, f1_score, ap, auroc

    def initialize_fold(self, base_model: torch.nn.Module, fold_idx: int, train_loader, val_loader):
        """Initialize model, optimizer, scheduler, and checkpoint paths for a fold.
        
        Args:
            base_model: The base model to initialize from
            fold_idx: The current fold index
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            KFold object containing initialized components
        """
        # Create fresh model copy
        model = copy.deepcopy(base_model)
        
        # Apply initial unfreezing strategy
        self._apply_unfreezing_strategy(model, current_epoch=0)
        
        # Get parameter groups
        backbone_params, head_params = self._get_parameter_groups(model)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {'params': head_params, 'lr': self.train_config.optim.base_lr}
        ]
        
        # Only add backbone parameters if there are any (not all frozen)
        if len(backbone_params) > 0:
            param_groups.append({'params': backbone_params, 'lr': self.train_config.optim.base_lr * 0.1})
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.train_config.optim.weight_decay
        )
        
        # Initialize scheduler if configured
        lr_scheduler = None
        if self.train_config.lr_scheduler and not self.args.progressive_unfreeze:
            num_training_steps = (len(train_loader) // self.accelerator.gradient_accumulation_steps) * self.train_config.epochs
            num_warmup_steps = int(self.train_config.lr_scheduler.warmup_ratio * num_training_steps)
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )
        
        # Prepare model, optimizer, and scheduler for DDP
        train_loader, val_loader, model, optimizer, lr_scheduler = self.accelerator.prepare(
            train_loader, val_loader, model, optimizer, lr_scheduler
        )
        
        # Setup checkpoint paths
        fold_checkpoint_dir = os.path.join(self.checkpoint_dir, f"fold_{fold_idx}" if fold_idx > 0 else "")
        if self.accelerator.is_main_process:
            os.makedirs(fold_checkpoint_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()  # Synchronize processes
        best_checkpoint = os.path.join(fold_checkpoint_dir, "best.pt") 
        
        # Initialize training state
        best_metric = -float("inf")
        start_epoch = 0
        
        # Resume if requested: always use best.pt inside the computed fold checkpoint dir
        if self.args.resume:
            if not os.path.exists(best_checkpoint):
                raise RuntimeError(f"No checkpoint found to resume. Expected a 'best.pt' at: {best_checkpoint}")

            if self.accelerator.is_main_process:
                ckpt = torch.load(best_checkpoint, map_location=self.accelerator.device)
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                if lr_scheduler and ckpt.get("scheduler_state"):
                    lr_scheduler.load_state_dict(ckpt["scheduler_state"])
                best_metric = ckpt.get("best_metric", best_metric)
                start_epoch = ckpt.get("epoch", 0)
                logger.info(
                    f"Fold {fold_idx if fold_idx > 0 else 'single'}: Resumed from epoch {start_epoch}, "
                    f"best_metric={best_metric:.4f} from {best_checkpoint}"
                )

            # Synchronize values across processes
            self.accelerator.wait_for_everyone()
            # Create tensors on each rank
            start_epoch_tensor = torch.tensor(start_epoch, device=self.accelerator.device)
            best_metric_tensor = torch.tensor(best_metric, device=self.accelerator.device)
            # Reduce to get values from rank 0
            start_epoch = self.accelerator.reduce(start_epoch_tensor, reduction="mean").item()
            best_metric = self.accelerator.reduce(best_metric_tensor, reduction="mean").item()
        
        return KFold(
            model, optimizer, lr_scheduler, fold_checkpoint_dir,
            best_checkpoint, best_metric, start_epoch, train_loader, val_loader
        )

    def train(self, fold_loaders, is_kfold, patience):
        """Train model with k-fold or single-split, returning the best model."""
        num_epochs = self.train_config.epochs
        best_metric_global = -float("inf")  
        best_model_global = None
        kfold_results = []

        # Iterate over folds
        for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders, start=0):
            if is_kfold:
                fold_idx = fold_idx + 1  
                log_prefix = f"fold{fold_idx}/"
            else:
                fold_idx = 0  
                log_prefix = ""  
                
            if self.accelerator.is_main_process:
                logger.info(f"{log_prefix} Training size: {len(train_loader)} Validation size: {len(val_loader)}")
            
            # Initialize KFold 
            kfold = self.initialize_fold(
                self.model,
                fold_idx,
                train_loader,
                val_loader
            )
            
            # Setup early stopping
            early_stopper = None
            if patience and self.accelerator.is_main_process:
                early_stopper = EarlyStopping(
                    patience=patience,
                    min_delta=self.train_config.early_stopping.min_delta,
                    mode=self.train_config.early_stopping.mode,
                    ckpt_path=kfold.best_checkpoint,
                    accelerator=self.accelerator
                )
            
            # Training loop
            for epoch in range(kfold.start_epoch, num_epochs):
                # Progressive unfreezing - handle before training starts to avoid race conditions
                if self.args.progressive_unfreeze and self.accelerator.is_main_process:
                    # Apply unfreezing strategy on main process
                    if epoch > 0:
                        self._apply_unfreezing_strategy(kfold.model, current_epoch=epoch)
                        self._update_optimizer_parameter_groups(kfold.optimizer, kfold.model)
                    
                    # Log unfreezing progress to wandb
                    model_info = self._get_model_layer_info(kfold.model)
                    layer_pattern = model_info['layer_pattern']
                    total_layers = model_info['total_layers']
                    
                    # Count unfrozen layers
                    unfrozen_layers = sum(1 for name, param in kfold.model.backbone.named_parameters() 
                                        if any(pattern.format(i) in name for i in range(total_layers) 
                                              for pattern in [layer_pattern]) and param.requires_grad)
                    
                    wandb.log({
                        f"{log_prefix}trainer/unfrozen_layers": unfrozen_layers,
                        f"{log_prefix}trainer/total_layers": total_layers,
                        f"{log_prefix}trainer/unfreeze_ratio": unfrozen_layers / total_layers,
                    })
                    
                    # Synchronize all processes to ensure unfreezing is applied consistently
                    self.accelerator.wait_for_everyone()
                
                train_loss, train_acc, train_auroc = self.train_per_epoch(
                    epoch, kfold.model, kfold.train_loader, kfold.optimizer,
                    kfold.lr_scheduler, log_prefix
                )
                if self.accelerator.is_main_process:
                    print(f'{log_prefix} Epoch {epoch+1} \t\t Train loss: {train_loss:.3f} \t\t Top1 Acc: {train_acc:.3f}')
                
                val_loss, val_acc, val_f1, val_ap, val_auroc = self.eval_per_epoch(
                    kfold.model, kfold.val_loader
                )
                if self.accelerator.is_main_process:
                    print(f'{log_prefix} Epoch {epoch+1} \t\t Val loss {val_loss:.3f} \t\t AUPRC {val_ap:.3f}')
                
                # Log metrics
                if self.accelerator.is_main_process:
                    wandb.log({
                        f"{log_prefix}train/loss_per_epoch": train_loss,
                        f"{log_prefix}train/ACC": train_acc,
                        f"{log_prefix}train/AUROC": train_auroc,
                        f"{log_prefix}val/loss_per_epoch": val_loss,
                        f"{log_prefix}val/ACC": val_acc,
                        f"{log_prefix}val/F1_Score": val_f1,
                        f"{log_prefix}val/AUPRC": val_ap,
                        f"{log_prefix}val/AUROC": val_auroc,  
                    })
                
                # Early stopping
                if early_stopper and self.accelerator.is_main_process:
                    early_stop, new_best_metric = early_stopper.step(val_ap, kfold.model, kfold.optimizer, kfold.lr_scheduler, epoch + 1)
                    if new_best_metric is not None:
                        kfold.best_metric = new_best_metric
                    if early_stop:
                        logger.info(f"{log_prefix}Stopping early at epoch {epoch + 1}")
                        break
                else:
                    # If not using early stopping, update best metric directly
                    if val_ap > kfold.best_metric:
                        kfold.best_metric = val_ap
                        if self.accelerator.is_main_process:
                            logger.info(f"{log_prefix}New best model for this fold with AUPRC={val_ap:.4f}")
                            # Save checkpoint
                            try:
                                # Get the unwrapped model state
                                model_state = self.accelerator.get_state_dict(kfold.model)
                                # Save checkpoint
                                checkpoint_data = {
                                    "epoch": epoch + 1,
                                    "model_state": model_state,
                                    "optimizer_state": kfold.optimizer.state_dict(),
                                    "scheduler_state": kfold.lr_scheduler.state_dict() if kfold.lr_scheduler else None,
                                    "best_metric": kfold.best_metric,
                                }
                                
                                # Add multi-view configuration if applicable
                                if hasattr(kfold.model, 'multi_view') and kfold.model.multi_view:
                                    checkpoint_data.update({
                                        "num_views": kfold.model.num_views,
                                        "view_fusion_type": kfold.model.view_fusion_type,
                                        "adapter_dim": getattr(kfold.model, 'adapter_dim', None),
                                        "view_fusion_hidden_dim": getattr(kfold.model, 'view_fusion_hidden_dim', None),
                                    })
                                
                                # Add Ark-specific configuration if applicable
                                if hasattr(kfold.model, 'use_backbone_projector'):
                                    checkpoint_data.update({
                                        "use_backbone_projector": kfold.model.use_backbone_projector,
                                    })
                                
                                torch.save(checkpoint_data, kfold.best_checkpoint)
                                logger.info(f"New best validation metric = {kfold.best_metric:.4f} at epoch {epoch+1}, saved best.pt")
                            except Exception as e:
                                logger.error(f"Failed to save checkpoint: {e}")
                                raise
            
            # Load best model for this fold on all ranks for consistency
            try:
                ckpt = torch.load(kfold.best_checkpoint, map_location='cpu')
                self.accelerator.unwrap_model(kfold.model).load_state_dict(ckpt["model_state"])
                best_model = kfold.model
                if self.accelerator.is_main_process:
                    logger.info(f"{log_prefix}Loaded best model checkpoint with AUPRC={ckpt['best_metric']:.4f}")
            except Exception as e:
                # Fallback to current model if no checkpoint exists
                logger.warning(f"{log_prefix}Failed to load best model checkpoint: {e}. Using current model instead.")
                best_model = kfold.model
                
            # Ensure all ranks are synchronized
            self.accelerator.wait_for_everyone()

            # If the training loop was skipped (e.g., start_epoch >= num_epochs),
            if kfold.start_epoch >= num_epochs:
                if self.accelerator.is_main_process:
                    logger.warning(
                        f"{log_prefix}Start epoch ({kfold.start_epoch}) >= num_epochs ({self.train_config.epochs}). Skipping training."
                    )
                if not isinstance(kfold.best_metric, (int, float)):
                    _, _, _, eval_ap, _ = self.eval_per_epoch(kfold.model, kfold.val_loader)
                    kfold.best_metric = eval_ap
            
            # Track fold results
            kfold_results.append({
                "fold": fold_idx or "single",
                "val_ap": kfold.best_metric
            })
            
            # Update global best model
            if kfold.best_metric > best_metric_global and self.accelerator.is_main_process:
                best_metric_global = kfold.best_metric
                best_model_global = copy.deepcopy(best_model)
                logger.info(f"{log_prefix}New global best model with AUPRC={best_metric_global:.4f}")
        
        # Log k-fold summary
        if is_kfold and self.accelerator.is_main_process:
            avg_ap = np.mean([res["val_ap"] for res in kfold_results])
            std_ap = np.std([res["val_ap"] for res in kfold_results])
            logger.info(f"K-fold results: Mean AUPRC={avg_ap:.4f} ± {std_ap:.4f}")
            wandb.log({"kfold/mean_ap": avg_ap, "kfold/std_ap": std_ap})
        
        if best_model_global is None and self.accelerator.is_main_process:
            if is_kfold:
                logger.warning("No best model was found during cross-validation. Using the last-fold model.")
            best_model_global = best_model
        return self.accelerator.unwrap_model(best_model_global) if best_model_global is not None else self.accelerator.unwrap_model(best_model)

    def export_onnx(self, model, model_name):
        """Export the model to ONNX format."""
        try:
            # Set the model to evaluation mode
            model = model.to("cpu")
            model.eval()
            
            # Get input size for the model
            input_size = MODEL_INPUT_SIZES[model_name]
            device = "cpu"
            
            # Check if model is multi-view and create appropriate dummy input
            if hasattr(model, 'multi_view') and model.multi_view:
                dummy_input = torch.randn(1, 4, 3, *input_size, device=device)  # [B, 4, C, H, W]
                logger.info(f"Exporting ONNX model for multi-view with input shape: {dummy_input.shape}")
            else:
                dummy_input = torch.randn(1, 3, *input_size, device=device)  # [B, C, H, W]
                logger.info(f"Exporting ONNX model for single-view with input shape: {dummy_input.shape}")
                        
            # Check if model supports attention maps
            if hasattr(model, 'return_attentions'):
                # DINO and MedSigLIP models
                include_attentions = model.return_attentions
            elif hasattr(model, 'backbone') and hasattr(model.backbone, 'return_attention'):
                # Ark models
                include_attentions = model.backbone.return_attention
            else:
                include_attentions = False
            
            # Build output configuration
            if include_attentions:
                output_names = ["logits", "all_attentions"]
                dynamic_axes = {
                    "pixel_values": {0: "batch_size"},   # batch dimension of input is dynamic
                    "logits": {0: "batch_size"},         # batch dimension of output is dynamic
                    "all_attentions": {1: "batch_size"}, # batch dimension of output is dynamic
                }
            else:
                output_names = ["logits"]
                dynamic_axes = {
                    "pixel_values": {0: "batch_size"}, # batch dimension of input is dynamic
                    "logits": {0: "batch_size"},       # batch dimension of output is dynamic
                }
            
            logger.info(f"ONNX export configuration: {len(output_names)} outputs - {output_names}")
            
            # Export the model
            onnx_path = os.path.join(self.checkpoint_dir, "best.onnx")
            torch.onnx.export(
                model,                        # model to be saved (on CPU, in eval mode)
                dummy_input,                  # model input (a single tensor or a tuple of tensors, matching model's forward())
                onnx_path,                    # where to save the model
                export_params=True,           # store the trained parameter weights inside the model file
                opset_version=17,             # the ONNX version to export the model to
                do_constant_folding=True,     # whether to execute constant folding for optimization
                input_names=["pixel_values"], # the model's input names
                output_names=output_names,    # the model's output names
                dynamic_axes=dynamic_axes,    # (Optional) allow dynamic batch sizes
                verbose=True                  # Enable verbose mode for debugging
            )
            logger.info(f"Successfully exported the best model to onnx model at {onnx_path}!")
            
        except Exception as e:
            logger.error(f"Failed to export ONNX model: {e}")
            logger.info("The best model state dict is still available in the checkpoint directory as 'best.pt'") 