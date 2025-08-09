import torch

from rad_dino.utils.zero_shot_transfer.ark_zero_shot_postprocess import (
    normalize_label,
    get_ark_disease_names,
    build_pretrained_ark_label_lookup,
    find_pretrained_ark_indices_for_synonyms,
    build_target_to_pretrained_ark_indices,
    aggregate_targeted_pred_probs,
)
from rad_dino.configs.ark_zero_shot_config import ARK_PRETRAINED_TASKS


def test_normalize_label_basic_cases():
    assert normalize_label("Pleural_Effusion!!") == "pleural effusion"
    assert normalize_label("No Finding") == "no finding"
    assert normalize_label("A/B+C") == "a/b+c"
    assert normalize_label("  Multi   Space  ") == "multi space"
    # punctuation removed, underscore to space, numbers preserved
    assert normalize_label("Fibrosis-Stage_2") == "fibrosisstage 2"


def test_get_ark_disease_names_contains_expected_and_length():
    names = get_ark_disease_names()
    expected_len = sum(len(task["diseases"]) for task in ARK_PRETRAINED_TASKS.values())
    assert len(names) == expected_len
    # spot-check a few known labels
    assert "Pneumonia" in names
    assert any("Effusion" in n for n in names)


def test_build_pretrained_ark_label_lookup_has_multiple_indices_for_common_labels():
    lookup = build_pretrained_ark_label_lookup()
    # very common across Ark tasks
    assert "pneumonia" in lookup and len(lookup["pneumonia"]) >= 3
    # appears in at least two Ark tasks
    assert "pleural effusion" in lookup and len(lookup["pleural effusion"]) >= 2


def test_find_pretrained_indices_for_synonyms_unions_matches():
    lookup = build_pretrained_ark_label_lookup()
    result = find_pretrained_ark_indices_for_synonyms([
        "pleural effusion",
        "effusion",
    ], lookup)
    expected = set(lookup.get("pleural effusion", [])) | set(lookup.get("effusion", []))
    assert set(result) == expected


def test_build_target_to_pretrained_ark_indices_multiclass_with_aliases():
    # Use dataset that defines aliases mapping
    dataset_name = "VinDr-CXR"
    downstream_labels = [
        "Pleural Effusion",
        "Tuberculosis",
        "Aortic Enlargement",  # maps to "enlarged cardiomediastinum"
    ]
    mapping = build_target_to_pretrained_ark_indices(
        dataset_name=dataset_name,
        task_type="multiclass",
        downstream_target_labels=downstream_labels,
    )

    # keys are normalized
    assert "pleural effusion" in mapping and len(mapping["pleural effusion"]) > 0
    assert "tuberculosis" in mapping and len(mapping["tuberculosis"]) > 0
    assert "aortic enlargement" in mapping and len(mapping["aortic enlargement"]) > 0

    # verify aliasing routed to pretrained labels as expected
    pretrained_lookup = build_pretrained_ark_label_lookup()
    cardiomediastinum_indices = pretrained_lookup.get("enlarged cardiomediastinum", [])
    assert set(mapping["aortic enlargement"]) == set(cardiomediastinum_indices)


def test_build_target_to_pretrained_ark_indices_binary_single_positive():
    # RSNA-Pneumonia defines aliases only for "pneumonia" which suits binary path
    dataset_name = "RSNA-Pneumonia"
    mapping = build_target_to_pretrained_ark_indices(
        dataset_name=dataset_name,
        task_type="binary",
        downstream_target_labels=["pneumonia"],  # required by current implementation
    )
    assert "pneumonia" in mapping and len(mapping["pneumonia"]) > 0


def test_aggregate_targeted_pred_probs_multiclass_values():
    # prepare mapping for two downstream labels with clear index sets
    dataset_name = "VinDr-CXR"
    downstream_labels = ["Pleural Effusion", "Tuberculosis"]
    mapping = build_target_to_pretrained_ark_indices(
        dataset_name=dataset_name,
        task_type="multiclass",
        downstream_target_labels=downstream_labels,
    )

    # build deterministic prediction tensor where value equals class index
    total_classes = len(get_ark_disease_names())
    row0 = torch.arange(total_classes, dtype=torch.float32)
    row1 = torch.arange(total_classes, dtype=torch.float32) + 1.0
    preds = torch.stack([row0, row1], dim=0)

    out = aggregate_targeted_pred_probs(
        pred_probs=preds,
        target_to_pretrained_label_indices=mapping,
        task_type="multiclass",
        downstream_target_labels=[normalize_label(l) for l in downstream_labels],
    )

    assert out.shape == (2, 2)

    # expected: mean of the selected indices in each row; row1 is row0 + 1
    pleural_effusion_indices = mapping["pleural effusion"]
    tuberculosis_indices = mapping["tuberculosis"]

    def mean_of(indices):
        return torch.tensor(indices, dtype=torch.float32).mean()

    expected_row0 = torch.tensor([
        mean_of(pleural_effusion_indices),
        mean_of(tuberculosis_indices),
    ])
    expected_row1 = expected_row0 + 1.0

    assert torch.allclose(out[0], expected_row0)
    assert torch.allclose(out[1], expected_row1)


def test_aggregate_targeted_pred_probs_binary_values():
    dataset_name = "RSNA-Pneumonia"
    mapping = build_target_to_pretrained_ark_indices(
        dataset_name=dataset_name,
        task_type="binary",
        downstream_target_labels=["pneumonia"],
    )

    total_classes = len(get_ark_disease_names())
    row0 = torch.arange(total_classes, dtype=torch.float32)
    row1 = torch.arange(total_classes, dtype=torch.float32) * 2.0
    preds = torch.stack([row0, row1], dim=0)

    out = aggregate_targeted_pred_probs(
        pred_probs=preds,
        target_to_pretrained_label_indices=mapping,
        task_type="binary",
        downstream_target_labels=None,
    )

    assert out.shape == (2, 1)
    pneumonia_indices = mapping["pneumonia"]
    expected0 = torch.tensor(row0[pneumonia_indices].mean())
    expected1 = torch.tensor(row1[pneumonia_indices].mean())
    assert torch.allclose(out[0, 0], expected0)
    assert torch.allclose(out[1, 0], expected1)


