import torch
import os
import math
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision.transforms import ToPILImage
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_curve
from .roc_curve import auc_bootstrapping
from loggings.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

def plot_roc_curve(y_true, y_score, axis, bootstrapping=1000, drop_intermediate=False, fontdict={}, name='ROC', color='b', show_wp=True):
    # ----------- Bootstrapping ------------
    tprs, aucs, thrs, mean_fpr = auc_bootstrapping(y_true, y_score, bootstrapping, drop_intermediate)

    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0        
    std_tpr = np.nanstd(tprs, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    # ------ Averaged based on bootspraping ------
    mean_auc = np.nanmean(aucs)
    std_auc = np.nanstd(aucs, ddof=1)
  

    # --------- Specific Case -------------
    fprs, tprs, thrs = roc_curve(y_true, y_score, drop_intermediate=drop_intermediate)
    auc_val = auc(fprs, tprs)
    opt_idx = np.argmax(tprs - fprs)
    opt_tpr = tprs[opt_idx]
    opt_fpr = fprs[opt_idx]

  
    y_scores_bin = y_score>=thrs[opt_idx] # WANRING: Must be >= not > 
    conf_matrix = confusion_matrix(y_true, y_scores_bin) # [[TN, FP], [FN, TP]]
    


    axis.plot(fprs, tprs, color=color, label=rf"{name} (AUC = {auc_val:.2f} $\pm$ {std_auc:.2f})",
                lw=2, alpha=.8)
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    if show_wp:
        axis.hlines(y=opt_tpr, xmin=0.0, xmax=opt_fpr, color='g', linestyle='--')
        axis.vlines(x=opt_fpr, ymin=0.0, ymax=opt_tpr, color='g', linestyle='--')
    axis.plot(opt_fpr, opt_tpr, color=color, marker='o') 
    axis.plot([0, 1], [0, 1], linestyle='--', color='k')
    axis.set_xlim([0.0, 1.0])
    axis.set_ylim([0.0, 1.0])
    
    axis.legend(loc='lower right')
    axis.set_xlabel('1 - Specificity', fontdict=fontdict)
    axis.set_ylabel('Sensitivity', fontdict=fontdict)
    
    axis.grid(color='#dddddd')
    axis.set_axisbelow(True)
    axis.tick_params(colors='#dddddd', which='both')
    for xtick in axis.get_xticklabels():
        xtick.set_color('k')
    for ytick in axis.get_yticklabels():
        ytick.set_color('k')
    for child in axis.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_color('#dddddd')
 
    return tprs, fprs, auc_val, thrs, opt_idx, conf_matrix

def visualize_gradcam(model, input_tensor, class_labels, target_layer, image_id, path_out, accelerator, threshold=0.5):
    """
    Generate and save Grad-CAM heatmaps for positive labels.
    
    Args:
        model: Trained DinoClassifier model
        input_tensor: Input image tensor [1, C, H, W]
        class_labels: List of class names
        target_layer: Layer for Grad-CAM (e.g., backbone.blocks[-1])
        image_id: Image identifier
        path_out: Directory to save heatmaps
        accelerator: Accelerator instance
        threshold: Probability threshold for positive labels
    """
    if not accelerator.is_main_process:
        return None, None
    
    model.eval()
    input_tensor = input_tensor.to(accelerator.device)

    # Initialize Grad-CAM
    def reshape_transform(tensor):
        """
        transformer tensor: [B, T, C] with T = 1 + H*W
        returns      : [B, C, H, W]
        """
        # 1) remove cls token
        tensor = tensor[:, 1:, :]
        B, N, C = tensor.size()
        H = W = int(math.sqrt(N))
        # 2) reshape   : [B, N, C] -> [B, C, H, W]
        tensor = tensor.permute(0, 2, 1).reshape(B, C, H, W)
        return tensor
    
    cam = GradCAM(
        model=model, 
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)  # [1, num_classes]
    pred_probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Select positive labels or top-3
    positive_indices = np.where(pred_probs > threshold)[0]
    if len(positive_indices) == 0:
        logger.warning(f"No positive labels for {image_id}. Using top-3.")
        positive_indices = np.argsort(pred_probs)[-3:]

    # Convert input tensor to RGB for visualization
    input_img = input_tensor.cpu().squeeze(0)
    input_img = ToPILImage()(input_img)
    input_img_np = np.array(input_img) / 255.0

    # Save original image
    input_img.save(os.path.join(path_out, f'input_{image_id}.png'))

    # Generate overlay heatmaps
    for class_idx in positive_indices:
        targets = [ClassifierOutputTarget(class_idx)]
        # Pass eigen_smooth=True to apply smoothing; One image in a batch
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets, eigen_smooth=True)[0, :]
        visualization = show_cam_on_image(input_img_np, grayscale_cam, use_rgb=True)
        visualization = Image.fromarray(visualization)
        visualization.save(os.path.join(path_out, f"gradcam_{image_id}_{class_labels[class_idx]}.png"))

def visualize_evaluate_metrics(y_true, y_pred, label, output_dir, accelerator, axis=None):
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}
    
    if not accelerator.is_main_process:
        return None, None
    
    # ------------------------------- AUPRC ---------------------------------
    precision, recall, _ = precision_recall_curve(y_true,y_pred)
    auprc = auc(recall, precision)
    
    fig, axis_auprc = plt.subplots(ncols=1, nrows=1, figsize=(6, 6)) if axis is None else (None, axis)
    axis_auprc.plot(recall, precision, label=f"AP {label} = {auprc:.2f}")
    axis_auprc.set_xlim([0.0, 1.0])
    axis_auprc.set_ylim([0.0, 1.0])
    axis_auprc.set_xlabel("Recall")
    axis_auprc.set_ylabel("Precision")
    axis_auprc.set_title(f"AUPRC for {label}")
    axis_auprc.legend(loc="lower left")
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"auprc_{label.replace(' ','_')}.png"), dpi=300)
        plt.close(fig)
    
    # ------------------------------- ROC-AUC ---------------------------------
    fprs, tprs, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fprs, tprs)
    
    fig, axis_roc = plt.subplots(ncols=1, nrows=1, figsize=(6, 6)) if axis is None else (None, axis)
    axis_roc.plot(fprs, tprs, label=f"AUC {label} = {roc_auc:.2f}")
    axis_roc.plot([0, 1], [0, 1], 'k--')
    axis_roc.set_xlim([0.0, 1.0])
    axis_roc.set_ylim([0.0, 1.0])
    axis_roc.set_xlabel('False Positive Rate', fontdict=fontdict)
    axis_roc.set_ylabel('True Positive Rate', fontdict=fontdict)
    axis_roc.set_title(f'ROC Curve {label}', fontdict=fontdict)
    axis_roc.legend(loc="lower right")
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"roc_{label.replace(' ','_')}.png"), dpi=300)
        plt.close(fig)
        
    #  -------------------------- Confusion Matrix -------------------------
    y_pred = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    
    df_cm = pd.DataFrame(cm, columns=['Absent', 'Present'], index=['Absent', 'Present'])
    fig, axis_cm = plt.subplots(1, 1, figsize=(4, 4)) if axis is None else (None, axis)
    sns.heatmap(df_cm, ax=axis_cm, cbar=False, fmt='d', annot=True)
    axis_cm.set_title(f'Confusion Matrix {label}\nACC={acc:.2f}', fontdict=fontdict)
    axis_cm.set_xlabel('Prediction', fontdict=fontdict)
    axis_cm.set_ylabel('Ground-truth', fontdict=fontdict)
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"confusion_matrix_{label.replace(' ','_')}.png"), dpi=300)
        plt.close(fig)

    logger.info(f"------Label {label.replace(' ','_')}--------")
    logger.info(f"Number of GT=1: {np.sum(y_true)}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Sensitivity: {sens:.2f}")
    logger.info(f"Specificity: {spec:.2f}")
    return auprc, roc_auc