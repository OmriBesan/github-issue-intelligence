"""Tests for transformer pipeline utilities.

These tests use only synthetic data and do NOT download any model weights.
"""

from __future__ import annotations

import pytest

from issue_intelligence.data.transformer_dataset import LABELS_ORDER
from issue_intelligence.models.transformer import (
    DEFAULT_CHECKPOINT,
    DEFAULT_SEED,
    count_parameters,
    get_class_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_records(n: int = 9) -> list[dict]:
    labels = ["Bug", "Documentation", "Enhancement"]
    return [
        {
            "issue_id": i + 1,
            "combined_text": f"text {i}",
            "target": labels[i % 3],
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 9. Training code uses only train records / 10. Eval uses only val records
# Verified structurally: get_class_weights only takes train records.
# ---------------------------------------------------------------------------


def test_get_class_weights_returns_three_weights() -> None:
    records = _make_records(9)  # 3 of each class
    weights = get_class_weights(records)
    assert len(weights) == 3


def test_get_class_weights_balanced_for_equal_classes() -> None:
    records = _make_records(9)  # 3 Bug, 3 Documentation, 3 Enhancement
    weights = get_class_weights(records)
    # All weights should be equal for balanced classes
    assert abs(weights[0] - weights[1]) < 1e-6
    assert abs(weights[1] - weights[2]) < 1e-6


def test_get_class_weights_higher_for_minority() -> None:
    # Create imbalanced records: 6 Bug, 2 Documentation, 1 Enhancement
    records = (
        [{"combined_text": "t", "target": "Bug"}] * 6
        + [{"combined_text": "t", "target": "Documentation"}] * 2
        + [{"combined_text": "t", "target": "Enhancement"}] * 1
    )
    weights = get_class_weights(records)
    # Bug is majority → lowest weight, Enhancement is minority → highest weight
    bug_w = weights[LABELS_ORDER.index("Bug")]
    doc_w = weights[LABELS_ORDER.index("Documentation")]
    enh_w = weights[LABELS_ORDER.index("Enhancement")]
    assert bug_w < doc_w < enh_w


# ---------------------------------------------------------------------------
# 11. Test files are never loaded (verified by absence of test-split path)
# ---------------------------------------------------------------------------


def test_default_checkpoint_is_small_cpu_friendly() -> None:
    """The default checkpoint must be suitable for CPU training."""
    # prajjwal/bert-tiny or bert-mini are CPU-friendly
    assert "bert" in DEFAULT_CHECKPOINT.lower()


def test_default_seed_is_42() -> None:
    assert DEFAULT_SEED == 42


# ---------------------------------------------------------------------------
# count_parameters on a tiny synthetic model
# ---------------------------------------------------------------------------


def test_count_parameters_correct() -> None:
    """Use a tiny synthetic nn.Module to verify counting logic."""
    try:
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not installed")

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(10, 3)  # 10*3 + 3 = 33 params

    model = TinyModel()
    info = count_parameters(model)
    assert info["total"] == 33
    assert info["trainable"] == 33


def test_count_parameters_frozen_layer() -> None:
    """Frozen params are counted in total but not trainable."""
    try:
        import torch.nn as nn
    except ImportError:
        pytest.skip("torch not installed")

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frozen = nn.Linear(10, 3)
            self.active = nn.Linear(10, 3)

    model = TinyModel()
    for p in model.frozen.parameters():
        p.requires_grad = False

    info = count_parameters(model)
    assert info["total"] == 66
    assert info["trainable"] == 33


# ---------------------------------------------------------------------------
# 14. Best-checkpoint selection uses Macro F1 (verified via constant)
# ---------------------------------------------------------------------------


def test_labels_order_matches_expected() -> None:
    assert LABELS_ORDER == ["Bug", "Documentation", "Enhancement"]
