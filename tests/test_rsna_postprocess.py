import pytest
import torch

from rad_dino.utils.zero_shot_transfer.rsna_postprocess import (
    rsna_multiclass_logits_to_binary_logits,
)

def test_rsna_binary_logits_shape():
    logits = torch.randn(4, 3)
    out = rsna_multiclass_logits_to_binary_logits(logits)
    assert out.shape == (4, 1)


def test_rsna_binary_logits_prob_equivalence():
    # Construct logits so that positive class (index 2) has clearly higher score
    logits = torch.tensor([
        [0.0, 0.0, 5.0],
        [1.0, 2.0, 3.0],
    ], dtype=torch.float32)

    out_logits = rsna_multiclass_logits_to_binary_logits(logits)
    # Sigmoid of returned logits should equal softmax prob of class index 2
    recovered_prob = torch.sigmoid(out_logits.squeeze(1))
    expected_prob_pos = torch.softmax(logits, dim=1)[:, 2]
    assert torch.allclose(recovered_prob, expected_prob_pos, atol=1e-6)


def test_rsna_binary_logits_invalid_input():
    with pytest.raises(ValueError):
        rsna_multiclass_logits_to_binary_logits(torch.randn(3))  # 1D
    with pytest.raises(ValueError):
        rsna_multiclass_logits_to_binary_logits(torch.randn(2, 4))  # wrong class dim
    with pytest.raises(ValueError):
        rsna_multiclass_logits_to_binary_logits(torch.randn(2, 3, 1))  # 3D


def test_rsna_binary_logits_numerical_stability():
    # Extremely confident positive class; softmax ~ 1.0 for index 2
    big = 100.0
    logits = torch.tensor([[0.0, 0.0, big]], dtype=torch.float32)
    out = rsna_multiclass_logits_to_binary_logits(logits)
    # logits should be finite
    assert torch.isfinite(out).all()
    prob = torch.sigmoid(out)[0, 0].item()
    # Should be numerically near 1, allowing saturation to exactly 1.0
    assert prob <= 1.0 and prob >= 1.0 - 1e-5


