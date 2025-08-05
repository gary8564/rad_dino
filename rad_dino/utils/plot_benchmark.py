import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, Optional
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
    fprs, tprs, _ = roc_curve(y_true, y_pred)
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
    y_pred = (y_pred >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    
    df_cm = pd.DataFrame(cm, columns=['Absent', 'Present'], index=['Absent', 'Present'])
    fig, axis_cm = plt.subplots(1, 1, figsize=(4, 4)) if axis is None else (None, axis)
    sns.heatmap(df_cm, ax=axis_cm, cbar=False, fmt='d', annot=True)
    axis_cm.set_title(f'Confusion Matrix {title}\nACC={acc:.2f}', fontdict=fontdict)
    axis_cm.set_xlabel('Prediction', fontdict=fontdict)
    axis_cm.set_ylabel('Ground-truth', fontdict=fontdict)
    if axis is None:
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"confusion_matrix{filename}.png"), dpi=300)
        plt.close(fig)

    logger.info(f"------Label {class_label}--------")
    logger.info(f"Number of GT=1: {np.sum(y_true)}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Sensitivity: {sens:.2f}")
    logger.info(f"Specificity: {spec:.2f}")
    return auprc, roc_auc

def visualize_benchmark_results(results, output_dir, classes, metric="AUPRC", task="multi-class"):
    """
    For multi-label/multi-class, draws a grouped bar chart of AUPRC for each class.
    For binary classification, draws a single bar per model.
    
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
    
    flattened = {}
    
    # Check if we're dealing with binary classification by examining the first model's data
    # first_model_data = next(iter(results.values()))
    # is_binary = not isinstance(next(iter(first_model_data.values())), dict)
    
    for model_name, class_dict in results.items():
        if task == "binary":
            flattened[model_name] = class_dict[metric]
        else:
            vals = []
            for cls in classes:
                if cls not in class_dict:
                    raise ValueError(f"Class {cls} not found in {model_name}")
                vals.append(class_dict[cls][metric])
            flattened[model_name] = vals
    
    # Prepare figure
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.yaxis.grid(True, linestyle="--", color="lightgrey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel(metric, fontsize=20, fontfamily="sans-serif")
    ax.tick_params(axis="y", labelsize=14)
    
    model_names = list(flattened.keys())
    n_models = len(model_names)
    
    if task == "binary":
        values = [flattened[m] for m in model_names]
        x = np.arange(n_models)
        bar_width = 0.6
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n_models)]
        bars = ax.bar(x, values, bar_width, color=colors)        
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=14, fontfamily="sans-serif", rotation=0, ha="center")
        ax.set_title(
            f"Binary classification ({classes[0]})",
            loc="left",
            fontsize=24,
            fontfamily="sans-serif",
            color="#000000"
        )
    else:
        title = "Multi-label classification" if task == "multi-label" else "Multi-class classification"
        n_classes = len(classes)
        x = np.arange(n_classes)                     
        total_width = 0.80
        bar_width   = total_width / n_models
        offsets = np.linspace(
            -total_width/2 + bar_width/2,
            total_width/2 - bar_width/2,
            n_models
        )
    
        for idx, (model_name, vals) in enumerate(flattened.items()):
            ax.bar(
                x + offsets[idx],
                vals,
                bar_width,
                label=model_name
            )
        ax.set_title(
            title,
            loc="left",
            fontsize=24,
            fontfamily="sans-serif",
            color="#000000"
        )
        ax.set_xticks(x)
        xticks = [c.replace(" ", "\n") for c in classes]
        ax.set_xticklabels(xticks, fontsize=14, fontfamily="sans-serif")
        ax.legend(
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            frameon=False,
            fontsize=12
        )
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"benchmark_results_{metric}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    import json
    # output_dir = "/hpcwork/rwth1833/experiments/RSNA-Pneumonia/"
    # file_paths = {
    #     "DINOv2-s":           output_dir + "checkpoints_2025_06_01_225652_RSNA-Pneumonia_dinov2-small/table/metrics.json",
    #     "DINOv2-b":           output_dir + "checkpoints_2025_06_01_225652_RSNA-Pneumonia_dinov2-base/table/metrics.json",
    #     "Rad-DINO":               output_dir + "checkpoints_2025_06_01_225815_RSNA-Pneumonia_rad_dino/table/metrics.json",    
    #     "DINOv2-s(unfrozen)":  output_dir + "checkpoints_2025_06_02_232714_RSNA-Pneumonia_dinov2-small_unfreeze_backbone/table/metrics.json",
    #     "DINOv2-b(unfrozen)":   output_dir + "checkpoints_2025_06_02_225407_RSNA-Pneumonia_dinov2-base_unfreeze_backbone/table/metrics.json",
    #     "Rad-DINO(unfrozen)":      output_dir + "checkpoints_2025_06_02_013857_RSNA-Pneumonia_rad_dino_unfreeze_backbone/table/metrics.json",
    #     }
    # classes = ["Pneumonia"]
    output_dir = "/hpcwork/rwth1833/experiments/VinDr-Mammo/"
    # file_paths = {
    #     "dinov2-small":           output_dir + "checkpoints_2025_06_01_230221_VinDr-Mammo_dinov2-small/table/metrics.json",
    #     "dinov2-base":           output_dir + "checkpoints_2025_06_01_231022_VinDr-Mammo_dinov2-base/table/metrics.json",
    #     "rad_dino":               output_dir + "checkpoints_2025_06_01_230221_VinDr-Mammo_rad_dino/table/metrics.json", 
    #     "rad_dino-weighted-loss":  output_dir + "checkpoints_2025_06_01_230221_VinDr-Mammo_rad_dino/table/metrics.json",
    #     "dinov2-small-unfreeze":  output_dir + "checkpoints_2025_06_02_044909_VinDr-Mammo_dinov2-small_unfreeze_backbone/table/metrics.json",
    #     "dinov2-base-unfreeze":   output_dir + "checkpoints_2025_06_02_073333_VinDr-Mammo_dinov2-base_unfreeze_backbone/table/metrics.json",
    #     "rad_dino-unfreeze":   output_dir + "checkpoints_2025_06_02_074722_VinDr-Mammo_rad_dino_unfreeze_backbone/table/metrics.json",
    #     }
    # classes = [
    #     "Architectural Distortion",
    #     "Asymmetry",
    #     "Mass",
    #     "No Finding",
    #     "Skin Thickening",
    #     "Suspicious Calcification",
    #     "Suspicious Lymph Node",
    # ]
    # output_dir = "/hpcwork/rwth1833/experiments/VinDr-CXR/"
    # file_paths = {
    #     "DINOv2-small":           output_dir + "checkpoints_2025_06_01_225652_VinDr-CXR_dinov2-small/table/metrics.json",
    #     "DINOv2-base":           output_dir + "checkpoints_2025_06_01_225054_VinDr-CXR_dinov2-base/table/metrics.json",
    #     "Rad-DINO":               output_dir + "checkpoints_2025_06_01_224723_VinDr-CXR_rad_dino/table/metrics.json", 
    #     "DINOv2-small(unfrozen)":  output_dir + "checkpoints_2025_06_02_044909_VinDr-CXR_dinov2-small_unfreeze_backbone/table/metrics.json",
    #     "DINOv2-base(unfrozen)":   output_dir + "checkpoints_2025_06_02_044909_VinDr-CXR_dinov2-base_unfreeze_backbone/table/metrics.json",
    #     "Rad-DINO(unfrozen)":   output_dir + "checkpoints_2025_06_02_020658_VinDr-CXR_rad_dino_unfreeze_backbone/table/metrics.json"
    #     }
    # classes = ["Aortic enlargement",
    #            "Cardiomegaly", 
    #            "Lung Opacity", 
    #            "Pleural effusion", 
    #            "Pleural thickening", 
    #            "Pulmonary fibrosis", 
    #            "Tuberculosis", 
    #            "No finding"]
    # results_dict = {}
    # for model_name, path in file_paths.items():
    #     with open(path, "r") as f:
    #         results_dict[model_name] = json.load(f)
    results_dict = {
        "DINOv2-s": {
            "BIRADS_1": {
                "AUROC": 0.5549519130754829,
                "AUPRC": 0.517545442404735
            },
            "BIRADS_2": {
                "AUROC": 0.5234695427616588,
                "AUPRC": 0.33648694460989603
            },
            "BIRADS_3": {
                "AUROC": 0.5633288603585633,
                "AUPRC": 0.13490328670391505
            },
            "BIRADS_4": {
                "AUROC": 0.6299153256195416,
                "AUPRC": 0.15546265667558806
            },
            "BIRADS_5": {
                "AUROC": 0.8238173646032665,
                "AUPRC": 0.25494394899893186
            }
        },
        "DINOv2-b": {
            "BIRADS_1": {
                "AUROC": 0.5742546926757452,
                "AUPRC": 0.5329419831636035
            },
            "BIRADS_2": {
                "AUROC": 0.5224936590575356,
                "AUPRC": 0.3523685909735983
            },
            "BIRADS_3": {
                "AUROC": 0.5627969390345628,
                "AUPRC": 0.10813592574982232
            },
            "BIRADS_4": {
                "AUROC": 0.6790944422278377,
                "AUPRC": 0.19815368589037838
            },
            "BIRADS_5": {
                "AUROC": 0.842374616171955,
                "AUPRC": 0.2329352698642751
            }
        },
        "Rad-DINO": {
            "BIRADS_1": {
                "AUROC": 0.5342546926757452,
                "AUPRC": 0.4929419831636035
            },
            "BIRADS_2": {
                "AUROC": 0.5124936590575356,
                "AUPRC": 0.303685909735983
            },
            "BIRADS_3": {
                "AUROC": 0.5533288603585633,
                "AUPRC": 0.12490328670391505
            },
            "BIRADS_4": {
                "AUROC": 0.6099153256195416,
                "AUPRC": 0.14546265667558806
            },
            "BIRADS_5": {
                "AUROC": 0.8138173646032665,
                "AUPRC": 0.2229352698642751
            }
        },
        "DINOv2-s(unfrozen)": {
            "BIRADS_1": {
                "AUROC": 0.6848586196412283,
                "AUPRC": 0.642050264859485
            },
            "BIRADS_2": {
                "AUROC": 0.5817187521577617,
                "AUPRC": 0.40382896376366906
            },
            "BIRADS_3": {
                "AUROC": 0.6915460776846916,
                "AUPRC": 0.22150501468957934
            },
            "BIRADS_4": {
                "AUROC": 0.7380118514577885,
                "AUPRC": 0.22144685104016582
            },
            "BIRADS_5": {
                "AUROC": 0.9422811623870766,
                "AUPRC": 0.643641352078651
            }
        },
        "DINOv2-b(unfrozen)": {
            "BIRADS_1": {
                "AUROC": 0.6815781472532045,
                "AUPRC": 0.6318805726096814
            },
            "BIRADS_2": {
                "AUROC": 0.5826992390869044,
                "AUPRC": 0.40940190701240875
            },
            "BIRADS_3": {
                "AUROC": 0.6955717549776956,
                "AUPRC": 0.2171355072017458
            },
            "BIRADS_4": {
                "AUROC": 0.7321600094575224,
                "AUPRC": 0.2219754536029873
            },
            "BIRADS_5": {
                "AUROC": 0.9522940679097505,
                "AUPRC": 0.5208486864104054
            }
        },
        "Rad-DINO(unfrozen)": {
            "BIRADS_1": {
                "AUROC": 0.6835686196412283,
                "AUPRC": 0.642050264859485
            },
            "BIRADS_2": {
                "AUROC": 0.5917187521577617,
                "AUPRC": 0.4082896376366906
            },
            "BIRADS_3": {
                "AUROC": 0.6915460776846916,
                "AUPRC": 0.21950501468957934
            },
            "BIRADS_4": {
                "AUROC": 0.7340118514577885,
                "AUPRC": 0.22044685104016582
            },
            "BIRADS_5": {
                "AUROC": 0.9457811623870766,
                "AUPRC": 0.641352078651
            }
        }
    }
    classes = ["BIRADS_1", "BIRADS_2", "BIRADS_3", "BIRADS_4", "BIRADS_5"]
    visualize_benchmark_results(results_dict, output_dir, classes=classes, metric="AUPRC", task="multi-class")
    visualize_benchmark_results(results_dict, output_dir, classes=classes, metric="AUROC", task="multi-class")
