from __future__ import annotations
import torch

def rsna_multiclass_logits_to_binary_logits(multclass_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert RSNA-Pneumonia multiclass logits ["No Lung Opacity/Not Normal", "Normal", "Lung Opacity"]
    into a single binary logit for pneumonia.

    Args:
        multclass_logits: Tensor of shape [B, 3]

    Returns:
        Tensor of shape [B, 1] representing the binary logit for pneumonia (positive).
    """
    if multclass_logits.dim() != 2 or multclass_logits.size(1) != 3:
        raise ValueError(f"Expected logits of shape [B, 3], got {tuple(multclass_logits.shape)}")

    # Convert to probabilities and extract positive class
    probs = torch.softmax(multclass_logits, dim=1)
    prob_pos = probs[:, 2]

    # Convert probability back to logit for downstream binary evaluation pipeline
    eps = 1e-6
    binary_logits = torch.logit(torch.clamp(prob_pos, eps, 1 - eps))
    return binary_logits.unsqueeze(1)


