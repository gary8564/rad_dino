import json
import os
import re
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from rad_dino.utils.plot_benchmark import visualize_benchmark_results

def compute_macro_averages(results_dict, classes):
    """
    Compute macro-averaged AUROC and AUPRC for multi-label/multi-class classification.

    Macro average = unweighted mean across the selected classes.

    Args:
        results_dict: { model_name: { class_name: {"AUROC": float, "AUPRC": float}, ... }, ... }
        classes: list[str] specifying which class keys to average over (order does not matter)

    Returns:
        { model_name: {"macro_AUROC": float, "macro_AUPRC": float}, ... }
    """
    macro_results = {}
    for model_name, per_class_metrics in results_dict.items():
        aurocs = []
        auprcs = []
        for cls in classes:
            if cls not in per_class_metrics:
                raise ValueError(f"Class '{cls}' not found in metrics for model '{model_name}'. Available: {list(per_class_metrics.keys())}")
            aurocs.append(float(per_class_metrics[cls]["AUROC"]))
            auprcs.append(float(per_class_metrics[cls]["AUPRC"]))

        macro_results[model_name] = {
            "macro_AUROC": sum(aurocs) / len(aurocs),
            "macro_AUPRC": sum(auprcs) / len(auprcs),
        }

    return macro_results


def _parse_base_approach_and_percent(model_key: str) -> Tuple[str, str, Optional[int]]:
    """
    Parse a display key like "Ark(50%)(Fine-tune)" into:
    - base model name (e.g., "Ark")
    - approach: "FT" if fine-tuned/unfrozen is implied, else "LP"
    - percent: int percentage if present, else None

    This mirrors the logic used in plotting utilities to detect fine-tuning.
    """
    base = model_key
    # Extract percent inside parentheses like (10%)
    percent = None
    m = re.search(r"\((\d+)%\)", model_key)
    if m:
        percent = int(m.group(1))
        # remove just the first (NN%) occurrence from base label for cleanliness
        base = base.replace(f"({percent}%)", "").strip()

    lower = model_key.lower()
    is_ft = (
        ("unfrozen" in lower)
        or ("unfreeze" in lower)
        or ("fine" in lower)
        or lower.endswith("(ft)")
        or (" ft" in lower)
    )
    approach = "FT" if is_ft else "LP"

    # Clean typical markers from base label (case-insensitive)
    cleanup_patterns = [
        r"\(unfrozen\)",
        r"unfreeze_backbone",
        r"unfrozen",
        r"unfreeze",
        r"\(ft\)",
        r"\(fine[- ]?tune\)",
        r"\(fine[- ]?tuning\)",
    ]
    for pat in cleanup_patterns:
        base = re.sub(pat, "", base, flags=re.IGNORECASE)
    base = base.replace("--", "-").replace("__", "_")
    base = re.sub(r"\s+", " ", base).strip().rstrip("-_ ")
    return base, approach, percent


def plot_data_efficiency_from_macro(macro_avgs: Dict[str, Dict[str, float]], output_dir: str, metric_key: str = "macro_AUPRC") -> str:
    """
    Create a data-efficiency plot: macro-averaged metric vs. training percentage.

    Lines represent model + approach (LP or FT). If multiple models exist, each gets
    its own color; LP is dashed, FT is solid.

    Returns the path of the saved figure.
    """
    # Organize data as {(base, approach): [(percent, value), ...]}
    series: Dict[Tuple[str, str], List[Tuple[int, float]]] = {}
    all_percents = set()
    for key, vals in macro_avgs.items():
        base, approach, percent = _parse_base_approach_and_percent(key)
        value = float(vals.get(metric_key, np.nan))
        if percent is None or np.isnan(value):
            continue
        series.setdefault((base, approach), []).append((percent, value))
        all_percents.add(percent)

    if not series:
        raise RuntimeError("No series to plot. Ensure keys include percentages like '(10%)'.")

    x_sorted = sorted(all_percents)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.yaxis.grid(True, linestyle="--", color="lightgrey")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Percentage of training samples")
    ax.set_ylabel(metric_key.replace("_", " "))
    ax.set_ylim(0.0, 1.0)

    # Build a consistent color for each base model
    bases = sorted({b for (b, _) in series.keys()})
    cmap = plt.get_cmap("tab10")
    base_to_color = {b: cmap(i % 10) for i, b in enumerate(bases)}

    # Plot each (base, approach)
    line_handles = []
    for (base, approach), points in sorted(series.items()):
        pts_sorted = sorted(points, key=lambda t: t[0])
        xs = [p for p, _ in pts_sorted]
        ys = [v for _, v in pts_sorted]
        color = base_to_color[base]
        style = "-" if approach == "FT" else "--"
        label = f"{base} ({'Fine-tuning' if approach == 'FT' else 'Linear-probing'})"
        handle, = ax.plot(xs, ys, style, color=color, marker="o", label=label)
        line_handles.append(handle)

    # X ticks as percentages and optionally counts if known
    ax.set_xticks(x_sorted)
    ax.set_xticklabels([f"{p}%" for p in x_sorted])
    ax.legend(loc="lower right")
    ax.set_title("Data efficiency")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "benchmark_results_data_efficiency_macro_AUPRC.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path

output_dir = "/hpcwork/rwth1833/experiments/RSNA-Pneumonia/"
file_paths = {
    "DINOv2-small":           output_dir + "checkpoints_2025_06_01_225652_RSNA-Pneumonia_dinov2-small/table/metrics.json",
    "DINOv2-base":           output_dir + "checkpoints_2025_06_01_225652_RSNA-Pneumonia_dinov2-base/table/metrics.json",
    "Rad-DINO":               output_dir + "checkpoints_2025_06_01_225815_RSNA-Pneumonia_rad_dino/table/metrics.json",    
    "MedSigLip":              output_dir + "checkpoints_2025_08_10_110543_RSNA-Pneumonia_medsiglip/table/metrics.json",
    "Ark":                    output_dir + "checkpoints_2025_08_10_110543_RSNA-Pneumonia_ark/table/metrics.json",
    "DINOv2-small(unfrozen)":  output_dir + "checkpoints_2025_06_02_232714_RSNA-Pneumonia_dinov2-small_unfreeze_backbone/table/metrics.json",
    "DINOv2-base(unfrozen)":   output_dir + "checkpoints_2025_06_02_225407_RSNA-Pneumonia_dinov2-base_unfreeze_backbone/table/metrics.json",
    "Rad-DINO(unfrozen)":      output_dir + "checkpoints_2025_06_02_013857_RSNA-Pneumonia_rad_dino_unfreeze_backbone/table/metrics.json",
    "MedSigLip(unfrozen)":     output_dir + "checkpoints_2025_08_10_110443_RSNA-Pneumonia_medsiglip_unfreeze_backbone/table/metrics.json",
    "Ark(unfrozen)":           output_dir + "checkpoints_2025_08_10_110443_RSNA-Pneumonia_medsiglip_unfreeze_backbone/table/metrics.json",
    }
task = "binary"
classes = ["Pneumonia"]

# output_dir = "/hpcwork/rwth1833/experiments/VinDr-Mammo/"
# task = "multi-class"
# file_paths = {
#     "DINOv2-small":           output_dir + "checkpoints_2025_07_11_125043_VinDr-Mammo_dinov2-small_multi_view_birads/table/metrics.json",
#     "DINOv2-base":           output_dir + "checkpoints_2025_07_11_125256_VinDr-Mammo_dinov2-base_multi_view_birads/table/metrics.json",
#     "Rad-DINO":               output_dir + "checkpoints_2025_07_11_130154_VinDr-Mammo_rad-dino_multi_view_birads/table/metrics.json", 
#     "MedSigLip":              output_dir + "checkpoints_2025_08_10_163307_VinDr-Mammo_medsiglip_multi_view/table/metrics.json",
#     "Ark":                    output_dir + "checkpoints_2025_08_10_013619_VinDr-Mammo_ark_multi_view/table/metrics.json",
#     "DINOv2-small(unfrozen)":  output_dir + "checkpoints_2025_06_02_044909_VinDr-Mammo_dinov2-small_unfreeze_backbone/table/metrics.json",
#     "DINOv2-base(unfrozen)":   output_dir + "checkpoints_2025_07_11_125852_VinDr-Mammo_dinov2-base_unfreeze_backbone_multi_view_birads/table/metrics.json",
#     "Rad-DINO(unfrozen)":   output_dir + "checkpoints_2025_07_12_162900_VinDr-Mammo_dinov2-base_progressive_unfreeze_multi_view_birads/table/metrics.json",
#     "MedSigLip(unfrozen)":   output_dir + "checkpoints_2025_08_10_163307_VinDr-Mammo_medsiglip_unfreeze_backbone_multi_view/table/metrics.json",
#     "Ark(unfrozen)":   output_dir + "checkpoints_2025_08_10_163219_VinDr-Mammo_ark_unfreeze_backbone_multi_view/table/metrics.json",
#     }
# classes = ["BIRADS_1", "BIRADS_2", "BIRADS_3", "BIRADS_4", "BIRADS_5"]

# output_dir = "/hpcwork/rwth1833/experiments/VinDr-CXR/"
# task = "multi-label"
# file_paths = {
#     "DINOv2-small":           output_dir + "checkpoints_2025_06_01_225652_VinDr-CXR_dinov2-small/table/metrics.json",
#     "DINOv2-base":           output_dir + "checkpoints_2025_06_01_225054_VinDr-CXR_dinov2-base/table/metrics.json",
#     "Rad-DINO":               output_dir + "checkpoints_2025_06_01_224723_VinDr-CXR_rad_dino/table/metrics.json", 
#     "MedSigLip":              output_dir + "checkpoints_2025_08_06_114136_VinDr-CXR_medsiglip/table/metrics.json",
#     "Ark":                    output_dir + "checkpoints_2025_08_06_114136_VinDr-CXR_ark/table/metrics.json",
#     "DINOv2-small(unfrozen)":  output_dir + "checkpoints_2025_06_02_044909_VinDr-CXR_dinov2-small_unfreeze_backbone/table/metrics.json",
#     "DINOv2-base(unfrozen)":   output_dir + "checkpoints_2025_06_02_044909_VinDr-CXR_dinov2-base_unfreeze_backbone/table/metrics.json",
#     "Rad-DINO(unfrozen)":   output_dir + "checkpoints_2025_06_02_020658_VinDr-CXR_rad_dino_unfreeze_backbone/table/metrics.json",
#     "MedSigLip(unfrozen)":   output_dir + "checkpoints_2025_08_13_144117_VinDr-CXR_medsiglip_unfreeze_backbone/table/metrics.json",
#     "Ark(unfrozen)":   output_dir + "checkpoints_2025_08_10_014533_VinDr-CXR_ark_unfreeze_backbone/table/metrics.json",
#     }

# file_paths = {
#     "Rad-DINO(10%)":               output_dir + "checkpoints_2025_08_12_014205_VinDr-CXR_rad-dino_unfreeze_backbone/table/metrics.json", 
#     "MedSigLip(10%)":              output_dir + "checkpoints_2025_08_12_004823_VinDr-CXR_medsiglip_unfreeze_backbone/table/metrics.json",
#     "Ark(10%)":                    output_dir + "checkpoints_2025_08_12_004616_VinDr-CXR_ark/table/metrics.json",
#     "Ark(10%)(Fine-tune)":         output_dir + "checkpoints_2025_08_12_004616_VinDr-CXR_ark_unfreeze_backbone/table/metrics.json",
#     "Rad-DINO(50%)":   output_dir + "checkpoints_2025_08_12_015046_VinDr-CXR_rad-dino_unfreeze_backbone/table/metrics.json",
#     "MedSigLip(50%)":   output_dir + "checkpoints_2025_08_13_144244_VinDr-CXR_medsiglip_unfreeze_backbone/table/metrics.json",
#     "Ark(50%)":   output_dir + "checkpoints_2025_08_13_144443_VinDr-CXR_ark/table/metrics.json",
#     "Ark(50%)(Fine-tune)":         output_dir + "checkpoints_2025_08_13_144320_VinDr-CXR_ark_unfreeze_backbone/table/metrics.json",
#     "Rad-DINO(100%)":   output_dir + "checkpoints_2025_06_02_020658_VinDr-CXR_rad_dino_unfreeze_backbone/table/metrics.json",
#     "MedSigLip(100%)":   output_dir + "checkpoints_2025_08_13_144117_VinDr-CXR_medsiglip_unfreeze_backbone/table/metrics.json",
#     "Ark(100%)":   output_dir + "checkpoints_2025_08_06_114136_VinDr-CXR_ark/table/metrics.json",
#     "Ark(100%)(Fine-tune)":         output_dir + "checkpoints_2025_08_10_014533_VinDr-CXR_ark_unfreeze_backbone/table/metrics.json",
#     }
# classes = ["Aortic enlargement",
#             "Cardiomegaly", 
#             "Lung Opacity", 
#             "Pleural effusion", 
#             "Pleural thickening", 
#             "Pulmonary fibrosis", 
#             "Tuberculosis"]

# output_dir = "/hpcwork/rwth1833/experiments/VinDr-CXR/"
# task = "multi-label"
# file_paths = {
#     "DINOv2-small":           output_dir + "checkpoints_2025_06_01_225652_VinDr-CXR_dinov2-small/table/metrics.json",
#     "DINOv2-base":           output_dir + "checkpoints_2025_06_01_225054_VinDr-CXR_dinov2-base/table/metrics.json",
#     "Rad-DINO":               output_dir + "checkpoints_2025_06_01_224723_VinDr-CXR_rad_dino/table/metrics.json", 
#     "MedSigLip":              output_dir + "checkpoints_2025_08_06_114136_VinDr-CXR_medsiglip/table/metrics.json",
#     "Ark":                    output_dir + "checkpoints_2025_08_06_114136_VinDr-CXR_ark/table/metrics.json",
#     "DINOv2-small(unfrozen)":  output_dir + "checkpoints_2025_06_02_044909_VinDr-CXR_dinov2-small_unfreeze_backbone/table/metrics.json",
#     "DINOv2-base(unfrozen)":   output_dir + "checkpoints_2025_06_02_044909_VinDr-CXR_dinov2-base_unfreeze_backbone/table/metrics.json",
#     "Rad-DINO(unfrozen)":   output_dir + "checkpoints_2025_06_02_020658_VinDr-CXR_rad_dino_unfreeze_backbone/table/metrics.json",
#     "MedSigLip(unfrozen)":   output_dir + "checkpoints_2025_08_13_144117_VinDr-CXR_medsiglip_unfreeze_backbone/table/metrics.json",
#     "Ark(unfrozen)":   output_dir + "checkpoints_2025_08_10_014533_VinDr-CXR_ark_unfreeze_backbone/table/metrics.json",
#     }

# output_dir = "/hpcwork/rwth1833/experiments/TAIX-Ray/"
# task = "multi-label"
# file_paths = {
#     "Rad-DINO":           output_dir + "checkpoints_2025_08_13_014557_TAIX-Ray_dinov2-base/table/metrics.json",
#     # "Rad-DINO":               output_dir + "checkpoints_2025_08_11_135928_TAIX-Ray_rad-dino/table/metrics.json", 
#     "MedSigLip":              output_dir + "checkpoints_2025_08_11_140020_TAIX-Ray_medsiglip/table/metrics.json",
#     "Ark":                    output_dir + "checkpoints_2025_08_11_221119_TAIX-Ray_ark/table/metrics.json",
#     "Rad-DINO(unfrozen)":   output_dir + "checkpoints_2025_08_14_053434_TAIX-Ray_dinov2-base_unfreeze_backbone/table/metrics.json",
#     # "Rad-DINO(unfrozen)":   output_dir + "checkpoints_2025_08_11_230242_TAIX-Ray_rad-dino_unfreeze_backbone/table/metrics.json",
#     "MedSigLip(unfrozen)":   output_dir + "checkpoints_2025_08_11_223131_TAIX-Ray_medsiglip_unfreeze_backbone/table/metrics.json",
#     "Ark(unfrozen)":   output_dir + "checkpoints_2025_08_13_144443_TAIX-Ray_ark_unfreeze_backbone/table/metrics.json",
#     }

# output_dir = "/hpcwork/rwth1833/zero_shot_experiments/TAIX-Ray/"
# task = "multi-label"
# file_paths = {
#         "MedSigLip":              output_dir + "medsiglip/table/metrics.json",
#         "Ark":                    output_dir + "ark/table/metrics.json",
#         }


# classes = ["Cardiomegaly", 
#             "Pulmonary congestion", 
#             "Pleural effusion", 
#             "Pulmonary opacities", 
#             "Atelectasis"]


results_dict = {}
for model_name, path in file_paths.items():
    try:
        with open(path, "r") as f:
            results_dict[model_name] = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Missing file for '{model_name}': {path}. Skipping.")
    except json.JSONDecodeError as e:
        print(f"[WARN] Could not parse JSON for '{model_name}' at {path}: {e}. Skipping.")

visualize_benchmark_results(results_dict, output_dir, classes=classes, metric="AUPRC", task=task)
visualize_benchmark_results(results_dict, output_dir, classes=classes, metric="AUROC", task=task)

# Print AUROC and AUPRC (%) per model for binary task
if task == "binary":
    print("\nPer-model AUROC/AUPRC (%) (binary task):")
    header = "| Model | AUROC | AUPRC |"
    align = "| --- | ---: | ---: |"
    print(header)
    print(align)
    for model, metrics in results_dict.items():
        auroc = metrics.get("AUROC")
        auprc = metrics.get("AUPRC")
        auroc_str = f"{auroc * 100:.2f}%" if isinstance(auroc, (int, float)) else "-"
        auprc_str = f"{auprc * 100:.2f}%" if isinstance(auprc, (int, float)) else "-"
        print(f"| {model} | {auroc_str} | {auprc_str} |")

# Print per-class AUROC (%) and Macro AUROC per model in a table
# macro_avgs = compute_macro_averages(results_dict, classes)
# print("\nPer-class AUROC (%) and Macro AUROC per model:")
# header_cols = ["Model"] + classes + ["Macro AUROC"]
# header = "| " + " | ".join(header_cols) + " |"
# align = "| " + " | ".join(["---"] + ["---:"] * len(classes) + ["---:"]) + " |"
# print(header)
# print(align)
# for model in file_paths.keys():
#     row_vals = [model]
#     for cls in classes:
#         try:
#             auroc = float(results_dict[model][cls]["AUROC"]) * 100.0
#             row_vals.append(f"{auroc:.2f}%")
#         except Exception:
#             row_vals.append("-")
#     macro = macro_avgs.get(model, {}).get("macro_AUROC", None)
#     row_vals.append(f"{macro * 100:.2f}%" if macro is not None else "-")
#     print("| " + " | ".join(row_vals) + " |")

# Plot data-efficiency based on macro AUPRC
# fig_path = plot_data_efficiency_from_macro(macro_avgs, output_dir, metric_key="macro_AUPRC")
# print(f"Saved data-efficiency plot to: {fig_path}")