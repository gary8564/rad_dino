import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from rad_dino.utils.plot_benchmark import visualize_evaluate_metrics
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def compute_binary_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                          output_dir: str, accelerator, label: Optional[str] = None) -> Dict[str, float]:
    """Compute metrics for binary classification task"""
    y_true_binary = y_true.squeeze()
    y_pred_binary = y_pred_prob.squeeze()
    
    pr_auc, roc_auc = visualize_evaluate_metrics(
        y_true_binary, y_pred_binary, output_dir, accelerator, label=label, axis=None
    )
    
    return {
        "AUROC": float(roc_auc),
        "AUPRC": float(pr_auc)
    }

def compute_multilabel_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                              class_labels: List[str], output_dir: str, accelerator) -> Dict[str, Dict[str, float]]:
    """Compute metrics for multilabel classification task"""
    metrics = {}
    
    for i, cls in enumerate(class_labels):
        pr_auc, roc_auc = visualize_evaluate_metrics(
            y_true[:, i], y_pred_prob[:, i], output_dir, accelerator, label=cls, axis=None
        )
        metrics[cls] = {
            "AUROC": float(roc_auc),
            "AUPRC": float(pr_auc)
        }
    
    return metrics

def compute_multiclass_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                              class_labels: List[str], output_dir: str, accelerator) -> Dict[str, Any]:
    """Compute metrics for multiclass classification task"""
    metrics = {}
    
    # 1) Overall F1 score & confusion matrix (better for imbalanced classes)
    pred_idx = np.argmax(y_pred_prob, axis=1)
    f1_macro = f1_score(y_true, pred_idx, average='macro')
    f1_micro = f1_score(y_true, pred_idx, average='micro')
    cm = confusion_matrix(y_true, pred_idx)
    
    # Save multiclass confusion matrix plot
    if accelerator.is_main_process:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    xticklabels=class_labels, yticklabels=class_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Multiclass Confusion Matrix")
        fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
        plt.close(fig)
    
    # Store overall metrics in a separate section for consistency
    metrics["overall"] = {
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro)
    }
    
    # 2) Macro / micro ROC-AUC (one-vs-rest)
    try:
        macro_roc = roc_auc_score(y_true, y_pred_prob,
                                multi_class="ovr", average="macro")
        micro_roc = roc_auc_score(y_true, y_pred_prob,
                                multi_class="ovr", average="micro")
        metrics["overall"]["roc_auc_macro_ovr"] = float(macro_roc)
        metrics["overall"]["roc_auc_micro_ovr"] = float(micro_roc)
    except ValueError:
        # Handle degenerate cases
        metrics["overall"]["roc_auc_macro_ovr"] = None
        metrics["overall"]["roc_auc_micro_ovr"] = None
    
    # 3) Per-class ROC/PR curves (consistent with multilabel structure)
    for i, cls in enumerate(class_labels):
        pr_auc, roc_auc = visualize_evaluate_metrics(
            (y_true == i).astype(int),
            y_pred_prob[:, i],
            output_dir, accelerator,
            label=cls, axis=None
        )
        metrics[cls] = {"AUROC": float(roc_auc), "AUPRC": float(pr_auc)}
    
    return metrics

def compute_evaluation_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                              task: str, class_labels: List[str], 
                              output_dir: str, accelerator) -> Dict[str, Any]:
    """
    Main function to compute evaluation metrics for all task types.
    
    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities
        task: Task type ('binary', 'multiclass', 'multilabel')
        class_labels: List of class labels
        output_dir: Directory to save plots
        accelerator: Accelerator object for distributed training
    
    Returns:
        Dictionary containing computed metrics
    """
    if task == "binary":
        return compute_binary_metrics(y_true, y_pred_prob, output_dir, accelerator)
    elif task == "multilabel":
        return compute_multilabel_metrics(y_true, y_pred_prob, class_labels, output_dir, accelerator)
    elif task == "multiclass":
        return compute_multiclass_metrics(y_true, y_pred_prob, class_labels, output_dir, accelerator)
    else:
        raise ValueError(f"Unsupported task type: {task}. Must be one of ['binary', 'multiclass', 'multilabel']")
