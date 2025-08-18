from __future__ import annotations
import re
import torch
import logging
from typing import Dict, List, Optional, Tuple
from rad_dino.configs.ark_zero_shot_config import ARK_PRETRAINED_TASKS, DATASET_LABEL_ALIASES
from rad_dino.loggings.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

def normalize_label(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9\s/+]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def get_ark_disease_names() -> List[str]:
    diseases: List[str] = []
    for task in ARK_PRETRAINED_TASKS.values():
        diseases.extend(task["diseases"])
    return diseases

def build_pretrained_ark_label_lookup() -> Tuple[List[str], Dict[str, List[int]]]:
    """Build a lookup table from normalized pretrained Ark disease labels to their indices.

    Returns:
        label_to_indices: Dict mapping normalized pretrained labels to their indices
    """
    pretrained_labels_norm = [normalize_label(c) for c in get_ark_disease_names()]
    label_to_indices = {}
    for idx, label in enumerate(pretrained_labels_norm):
        label_to_indices.setdefault(label, []).append(idx)
    return label_to_indices

def find_pretrained_ark_indices_for_synonyms(
    synonyms: List[str],
    label_to_indices: Dict[str, List[int]],
) -> List[int]:
    """Find pretrained Ark disease that match the provided synonyms and return the indices of the matched diseases.
    Args:
        synonyms: List of synonyms to match
        label_to_indices: Dict mapping normalized labels to their indices
    Returns:
        List of indices of pretrained Ark disease labels that match the provided synonyms
    """
    matched = []
    syns = [normalize_label(s) for s in synonyms]
    for s in syns:
        matched.extend(label_to_indices.get(s, []))
    return sorted(set(matched))

def build_target_to_pretrained_ark_indices(
    dataset_name: str,
    task_type: str,
    downstream_target_labels: Optional[List[str]],
) -> Dict[str, List[int]]:
    """
    Build mapping from downstream task labels to pretrained Ark disease indices.

    Returns:
      target_to_pretrained_ark_indices: dict downstream_label_norm -> list of pretrained Ark indices
    """
    alias_map = DATASET_LABEL_ALIASES.get(dataset_name, {})
    label_to_indices = build_pretrained_ark_label_lookup()

    target_to_ark_indices = {}

    if task_type == "binary":
        positive_label = next(iter(alias_map.keys()))
        synonyms = alias_map.get(positive_label, [positive_label])
        target_to_ark_indices[positive_label] = find_pretrained_ark_indices_for_synonyms(synonyms, label_to_indices)
        return target_to_ark_indices

    if not downstream_target_labels:
        raise ValueError("`downstream_target_labels` must be provided if the downstream classification task is not a binary case.")
    for label in downstream_target_labels:
        key = normalize_label(str(label))
        synonyms = alias_map.get(key, [key])
        target_to_ark_indices[key] = find_pretrained_ark_indices_for_synonyms(synonyms, label_to_indices)
    return target_to_ark_indices


def aggregate_targeted_pred_probs(
    pred_probs: torch.Tensor,
    target_to_pretrained_label_indices: Dict[str, List[int]],
    task_type: str,
    downstream_target_labels: Optional[List[str]],
) -> torch.Tensor:
    """Average predicted probabilities over all matched pretrained disease label indices."""
    batch_size = pred_probs.shape[0]
    device = pred_probs.device
    if task_type == "binary":
        indices = next(iter(target_to_pretrained_label_indices.values()), [])
        if not indices:
            raise ValueError(f"No pretrained Ark disease labels found for the positive label: {key}")
        return pred_probs[:, indices].mean(dim=1, keepdim=True)

    assert downstream_target_labels is not None, "`downstream_target_labels` must be provided if the downstream classification task is not a binary case."
    out = torch.zeros((batch_size, len(downstream_target_labels)), dtype=pred_probs.dtype, device=device)
    for idx, label in enumerate(downstream_target_labels):
        key = normalize_label(str(label))
        indices = target_to_pretrained_label_indices.get(key, [])
        if not indices:
            logger.warning(f"No pretrained Ark disease labels found for the label: {key}")
            continue
        out[:, idx] = pred_probs[:, indices].mean(dim=1)
    return out



