import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_curve
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def visualize_evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str, 
                              accelerator, label: Optional[str] = None, axis=None) -> Tuple[float, float]:
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}
    
    if not accelerator.is_main_process:
        return None, None
    
    if label is None:
        class_label = ""
        title = ""
        filename = ""
    else:
        class_label = label
        title = f"for {label}"
        filename = f"_{label.replace(' ','_')}"
    
    # ------------------------------- AUPRC ---------------------------------
    precision, recall, _ = precision_recall_curve(y_true,y_pred)
    auprc = auc(recall, precision)
    
    fig, axis_auprc = plt.subplots(ncols=1, nrows=1, figsize=(6, 6)) if axis is None else (None, axis)
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
    
    # ------------------------------- ROC-AUC ---------------------------------
    fprs, tprs, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fprs, tprs)
    
    fig, axis_roc = plt.subplots(ncols=1, nrows=1, figsize=(6, 6)) if axis is None else (None, axis)
    axis_roc.plot(fprs, tprs, label=f"AUC {class_label} = {roc_auc:.2f}")
    axis_roc.plot([0, 1], [0, 1], 'k--')
    axis_roc.set_xlim([0.0, 1.0])
    axis_roc.set_ylim([0.0, 1.0])
    axis_roc.set_xlabel('False Positive Rate', fontdict=fontdict)
    axis_roc.set_ylabel('True Positive Rate', fontdict=fontdict)
    axis_roc.set_title(f'ROC Curve {title}', fontdict=fontdict)
    axis_roc.legend(loc="lower right")
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"roc{filename}.png"), dpi=300)
        plt.close(fig)
        
    #  -------------------------- Confusion Matrix -------------------------
    # Find optimal threshold via Youden's J statistic (J = TPR - FPR)
    youden_j = tprs - fprs
    best_idx = youden_j.argmax()
    best_thr = float(thresholds[best_idx])
    logger.info(f"Optimal threshold (Youden's J): {best_thr:.3f}")

    y_pred = (y_pred >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm.shape[0] > 1 and (cm[1, 1] + cm[1, 0]) > 0 else 0
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape[1] > 1 and (cm[0, 0] + cm[0, 1]) > 0 else 0
    
    df_cm = pd.DataFrame(cm, columns=['Negative', 'Positive'], index=['Negative', 'Positive'])
    fig, axis_cm = plt.subplots(1, 1, figsize=(5, 5)) if axis is None else (None, axis)
    sns.heatmap(df_cm, ax=axis_cm, cbar=False, fmt='d', annot=True, cmap='Blues')
    axis_cm.set_title(f'Confusion Matrix {title}\nACC={acc:.2f}', fontdict=fontdict)
    axis_cm.set_xlabel('Prediction', fontdict=fontdict)
    axis_cm.set_ylabel('Ground-truth', fontdict=fontdict)
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"confusion_matrix{filename}.png"), dpi=300)
        plt.close(fig)

    logger.info(f"------Label {class_label}--------")
    logger.info(f"Number of positive samples: {np.sum(y_true)}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Sensitivity: {sens:.3f}")
    logger.info(f"Specificity: {spec:.3f}")
    return auprc, roc_auc

def _parse_model_and_approach(model_name: str) -> Tuple[str, str]:
    """Derive a clean model name and approach label (LP vs FT) from model_name."""
    lower = model_name.lower()
    is_ft = (
        ("unfrozen" in lower)
        or ("unfreeze" in lower)
        or ("fine" in lower)
        or lower.endswith("(ft)")
        or (" ft" in lower)
    )
    approach = "FT" if is_ft else "LP"

    model = model_name
    for marker in [
        "(unfrozen)",
        "unfreeze_backbone",
        "unfrozen",
        "unfreeze",
        "(ft)",
        "(FT)",
        "(fine-tune)",
        "(fine tuning)",
        "(fine-tuning)",
    ]:
        model = model.replace(marker, "")
    model = model.replace("--", "-").replace("__", "_").strip().rstrip("-_ ")
    return model, approach


def _lighten_color(color_in, factor: float = 0.65):
    """Return a lighter shade of the input color by blending towards white.

    factor in [0,1]: higher means more lightening.
    """
    rgb = np.asarray(mcolors.to_rgb(color_in))
    return tuple(1.0 - factor * (1.0 - rgb))


def visualize_benchmark_results(results, output_dir, classes, metric="AUPRC", task="multi-label"):
    """
    Plot benchmark metrics across classes with optional grouping by model and approach.

    - Binary task: one bar per model
    - Multi-label/Multi-class: for each class (x-axis), create sub-groups per base model; inside
      each model sub-group draw two adjacent bars (LP, FT). Bars share a color per model; LP uses
      a lighter shade, FT uses the saturated/base color. Legends are split into: Model (colors)
      at the bottom center, and Approach (LP/FT) at the top-right.
    
    Parameters
    ----------
    results_dict : dict
        A mapping from model name (str) to either:
        1. Multi-class/Multi-label: { <class_name>: {"AUROC":..., "AUPRC":...} }
        2. Binary: {"AUROC":..., "AUPRC":...}
    output_dir: str
        The directory to save the plot.
    classes: list
        The list of classes to plot. For binary classification, can be a single descriptive name.
    metric: str
        Either "AUROC" or "AUPRC"
    """
    
    assert task in ["binary", "multi-class", "multi-label"], "Invalid task argument."
    
    # 1) Flatten the input structure to metric arrays per model key
    flattened: Dict[str, List[float]] = {}
    for model_key, class_dict in results.items():
        if task == "binary":
            flattened[model_key] = class_dict[metric]
        else:
            values_for_model: List[float] = []
            for cls in classes:
                if cls not in class_dict:
                    raise ValueError(f"Class {cls} not found in {model_key}")
                values_for_model.append(class_dict[cls][metric])
            flattened[model_key] = values_for_model
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.yaxis.grid(True, linestyle="--", color="lightgrey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(metric, fontsize=20, fontfamily="sans-serif")
    ax.tick_params(axis="y", labelsize=14)
    
    approach_legend_needed = False  
    
    if task == "binary":
        # Group by base model and approach (LP vs FT)
        base_groups: Dict[str, Dict[str, Optional[float]]] = {}
        base_order: List[str] = []
        for model_key, value in flattened.items():
            base, approach = _parse_model_and_approach(model_key)
            if base not in base_groups:
                base_groups[base] = {"LP": None, "FT": None}
                base_order.append(base)
            base_groups[base][approach] = value

        # Approach legend only if any FT exists
        approach_legend_needed = any(base_groups[base]["FT"] is not None for base in base_order)

        n_bases = len(base_order)
        x = np.arange(n_bases)

        # Visual sizing for LP/FT pair per base model
        total_pair_width = 0.65
        inner_gap = 0.06
        bar_width = (total_pair_width - inner_gap) / 2.0

        # Color map per base model
        cmap = plt.get_cmap("tab10")
        base_to_color = {base: cmap(i % 10) for i, base in enumerate(base_order)}

        for idx, base in enumerate(base_order):
            color = base_to_color[base]
            lp_val = base_groups[base]["LP"]
            ft_val = base_groups[base]["FT"]

            if (lp_val is not None) and (ft_val is not None):
                lp_x = x[idx] - (inner_gap + bar_width) / 2.0
                ft_x = x[idx] + (inner_gap + bar_width) / 2.0
                lp_w = ft_w = bar_width
            else:
                # Single bar centered
                lp_x = ft_x = x[idx]
                # Slightly wider when only one approach is present
                lp_w = ft_w = total_pair_width * (0.95 if not approach_legend_needed else 0.8)

            if lp_val is not None:
                ax.bar(
                    lp_x,
                    lp_val,
                    lp_w,
                    color=_lighten_color(color, factor=0.6),
                    edgecolor="black",
                    linewidth=0.5,
                )
            if ft_val is not None:
                ax.bar(
                    ft_x,
                    ft_val,
                    ft_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Axes text
        ax.set_xticks(x)
        ax.set_xticklabels(base_order, fontsize=14, fontfamily="sans-serif", rotation=0, ha="center")
        ax.set_title(
            f"Binary classification ({classes[0]})",
            loc="left",
            fontsize=24,
            fontfamily="sans-serif",
            color="#000000"
        )

        # Legends
        model_handles = [Patch(facecolor=base_to_color[base], edgecolor="black", label=base) for base in base_order]
        fig.legend(
            handles=model_handles,
            title="Model",
            ncol=min(5, max(1, len(base_order))),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            frameon=False,
            fontsize=12
        )
        if approach_legend_needed:
            regimen_handles = [
                Patch(facecolor="#cfcfcf", edgecolor="black", label="LP"),
                Patch(facecolor="#6e6e6e", edgecolor="black", label="FT"),
            ]
            ax.legend(
                handles=regimen_handles,
                title="Approach",
                loc="upper left",
                bbox_to_anchor=(0.995, 1.0),
                frameon=False,
                fontsize=12
            )
    else:
        # 2) Prepare multi-class/multi-label layout
        title = "Multi-label classification" if task == "multi-label" else "Multi-class classification"
        n_classes = len(classes)
        x = np.arange(n_classes)

        # Group by base model and approach (LP vs FT)
        base_groups: Dict[str, Dict[str, Optional[List[float]]]] = {}
        base_order: List[str] = []
        for model_key, vals in flattened.items():
            base, approach = _parse_model_and_approach(model_key)
            if base not in base_groups:
                base_groups[base] = {"LP": None, "FT": None}
                base_order.append(base)
            base_groups[base][approach] = vals

        # Determine whether to show the Approach legend:
        # show it if ANY fine-tuned (FT) models are present in the keys
        ft_present = any(base_groups[base]["FT"] is not None for base in base_order)
        approach_legend_needed = ft_present

        n_bases = len(base_order)
        total_width = 0.86
        # gap between different model groups inside each class slot
        # make groups closer when no FT bars are present
        between_model_gap = (0.02 if not approach_legend_needed else 0.06) if n_bases > 1 else 0.0
        group_area = total_width - between_model_gap * max(0, n_bases - 1)
        group_width = group_area / max(n_bases, 1)
        # very small inner gap so LP/FT appear as a pair
        inner_gap = group_width * 0.05
        bar_width = (group_width - inner_gap) / 2.0
        single_regimen_width = group_width * (0.95 if not approach_legend_needed else 0.8)

        # Color map: one color per base model
        cmap = plt.get_cmap("tab10")
        base_to_color = {base: cmap(i % 10) for i, base in enumerate(base_order)}

        # Compute offsets for each base group (center of LP+FT pair within the class slot)
        # We center the whole base-group span within [-total_width/2, total_width/2]
        if n_bases > 0:
            start = -total_width/2 + group_width/2
            base_offsets = np.array([start + i * (group_width + between_model_gap) for i in range(n_bases)])
        else:
            base_offsets = np.array([])

        # 3) Draw bars per class
        for base_idx, base in enumerate(base_order):
            color = base_to_color[base]
            lp_vals = base_groups[base]["LP"]
            ft_vals = base_groups[base]["FT"]

            # Positions for the two bars inside the base group
            center_offset = base_offsets[base_idx]
            if (lp_vals is not None) and (ft_vals is not None):
                lp_offset = center_offset - (inner_gap + bar_width) / 2.0
                ft_offset = center_offset + (inner_gap + bar_width) / 2.0
            else:
                # If only one regimen is available, center it within the group
                lp_offset = center_offset
                ft_offset = center_offset

            # LP bars (lighter shade)
            if lp_vals is not None:
                ax.bar(
                    x + lp_offset,
                    lp_vals,
                    bar_width if (ft_vals is not None) else single_regimen_width,
                    label=None,
                    color=_lighten_color(color, factor=0.6),
                    edgecolor="black",
                    linewidth=0.5,
                )

            # FT bars: saturated/base color (no hatch)
            if ft_vals is not None:
                ax.bar(
                    x + ft_offset,
                    ft_vals,
                    bar_width if (lp_vals is not None) else single_regimen_width,
                    label=None,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # 4) Titles and ticks
        ax.set_title(
            title,
            loc="left",
            fontsize=24,
            fontfamily="sans-serif",
            color="#000000"
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        # Display class names with underscores shown as spaces, then wrap spaces to newlines
        display_classes = [c.replace("_", " ") for c in classes]
        xticks = [c.replace(" ", "\n") for c in display_classes]
        ax.set_xticklabels(xticks, fontsize=14, fontfamily="sans-serif")

        # 5) Legends: one for models (colors), one for approach (grey scale)
        model_handles = [Patch(facecolor=base_to_color[base], edgecolor="black", label=base) for base in base_order]
        # Place model legend at the figure bottom (outside axes)
        fig.legend(
            handles=model_handles,
            title="Model",
            ncol=min(5, max(1, len(base_order))),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            frameon=False,
            fontsize=12
        )

        if approach_legend_needed:
            regimen_handles = [
                Patch(facecolor="#cfcfcf", edgecolor="black", label="LP"),
                Patch(facecolor="#6e6e6e", edgecolor="black", label="FT"),
            ]
            ax.legend(
                handles=regimen_handles,
                title="Approach",
                loc="upper left",
                bbox_to_anchor=(0.995, 1.0),
                frameon=False,
                fontsize=12
            )
    
    # 6) Reserve margins (use wider right margin only when approach legend is shown)
    right_margin = 0.95 if approach_legend_needed else 0.98
    plt.tight_layout(rect=(0.05, 0.13, right_margin, 0.93))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"benchmark_results_{metric}.png"), dpi=300, bbox_inches="tight")
    plt.close()

    