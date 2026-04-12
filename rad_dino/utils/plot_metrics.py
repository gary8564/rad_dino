"""Per-class evaluation plotting utilities (ROC, PR curve, confusion matrix)."""
import os
import logging
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
)

from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def visualize_evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    accelerator,
    label: Optional[str] = None,
    axis=None,
) -> Tuple[float, float]:
    """Plot AUPRC, ROC-AUC, and confusion matrix for a single binary class.

    Args:
        y_true: Ground-truth binary labels, shape (N,).
        y_pred: Predicted probabilities, shape (N,).
        output_dir: Directory to save the figure files.
        accelerator: HuggingFace ``Accelerator`` — plots are only saved on the
            main process.
        label: Optional class name used in titles and filenames.
        axis: If provided, draw into an existing ``Axes`` instead of creating a
            new figure (figures are not saved in this mode).

    Returns:
        ``(auprc, roc_auc)`` as floats, or ``(None, None)`` on non-main
        processes.
    """
    fontdict = {"fontsize": 10, "fontweight": "bold"}

    if not accelerator.is_main_process:
        return None, None

    if label is None:
        class_label = ""
        title = ""
        filename = ""
    else:
        class_label = label
        title = f"for {label}"
        filename = f"_{label.replace(' ', '_')}"

    # AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = auc(recall, precision)

    fig, axis_auprc = (
        plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        if axis is None
        else (None, axis)
    )
    axis_auprc.plot(recall, precision, label=f"AP {class_label} = {auprc:.2f}")
    axis_auprc.set_xlim([0.0, 1.0])
    axis_auprc.set_ylim([0.0, 1.0])
    axis_auprc.set_xlabel("Recall")
    axis_auprc.set_ylabel("Precision")
    axis_auprc.set_title(f"AUPRC {title}")
    axis_auprc.legend(loc="lower left")
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"auprc{filename}.png"), dpi=300)
        plt.close(fig)

    # ROC-AUC
    fprs, tprs, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fprs, tprs)

    fig, axis_roc = (
        plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        if axis is None
        else (None, axis)
    )
    axis_roc.plot(fprs, tprs, label=f"AUC {class_label} = {roc_auc:.2f}")
    axis_roc.plot([0, 1], [0, 1], "k--")
    axis_roc.set_xlim([0.0, 1.0])
    axis_roc.set_ylim([0.0, 1.0])
    axis_roc.set_xlabel("False Positive Rate", fontdict=fontdict)
    axis_roc.set_ylabel("True Positive Rate", fontdict=fontdict)
    axis_roc.set_title(f"ROC Curve {title}", fontdict=fontdict)
    axis_roc.legend(loc="lower right")
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"roc{filename}.png"), dpi=300)
        plt.close(fig)

    # Confusion matrix — threshold via Youden's J statistic
    youden_j = tprs - fprs
    best_thr = float(thresholds[youden_j.argmax()])
    logger.info(f"Optimal threshold (Youden's J): {best_thr:.3f}")

    y_pred_bin = (y_pred >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred_bin)
    acc = accuracy_score(y_true, y_pred_bin)

    sens = (
        cm[1, 1] / (cm[1, 1] + cm[1, 0])
        if cm.shape[0] > 1 and (cm[1, 1] + cm[1, 0]) > 0
        else 0
    )
    spec = (
        cm[0, 0] / (cm[0, 0] + cm[0, 1])
        if cm.shape[1] > 1 and (cm[0, 0] + cm[0, 1]) > 0
        else 0
    )

    df_cm = pd.DataFrame(cm, columns=["Negative", "Positive"], index=["Negative", "Positive"])
    fig, axis_cm = (
        plt.subplots(1, 1, figsize=(5, 5)) if axis is None else (None, axis)
    )
    sns.heatmap(df_cm, ax=axis_cm, cbar=False, fmt="d", annot=True, cmap="Blues")
    axis_cm.set_title(f"Confusion Matrix {title}\nACC={acc:.2f}", fontdict=fontdict)
    axis_cm.set_xlabel("Prediction", fontdict=fontdict)
    axis_cm.set_ylabel("Ground-truth", fontdict=fontdict)
    if axis is None:
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, f"confusion_matrix{filename}.png"), dpi=300
        )
        plt.close(fig)

    logger.info(f"------Label {class_label}--------")
    logger.info(f"Number of positive samples: {np.sum(y_true)}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Sensitivity: {sens:.3f}")
    logger.info(f"Specificity: {spec:.3f}")
    return auprc, roc_auc
