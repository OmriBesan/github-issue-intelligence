"""Tests for evaluation metrics."""

from __future__ import annotations

import pytest

from issue_intelligence.evaluation.metrics import compute_metrics


def test_compute_metrics_accuracy() -> None:
    y_true = ["Bug", "Bug", "Enhancement", "Documentation"]
    y_pred = ["Bug", "Enhancement", "Enhancement", "Documentation"]

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75


def test_macro_f1_differs_from_weighted_f1_on_imbalanced_data() -> None:
    # 90 Bugs, 10 Enhancements
    y_true = ["Bug"] * 90 + ["Enhancement"] * 10

    # Predicts Bug 100% of the time (Majority class baseline)
    y_pred = ["Bug"] * 100

    metrics = compute_metrics(y_true, y_pred)

    # Weighted F1 heavily weights Bug (high F1 = gets all predictions).
    # Macro F1 averages Bug and Enhancement equally (Enhancement F1 is 0).
    assert metrics["macro_f1"] < metrics["weighted_f1"]

    # Verify Bug class metrics
    assert metrics["per_class"]["Bug"]["recall"] == 1.0
    assert metrics["per_class"]["Enhancement"]["recall"] == 0.0


def test_confusion_matrix_ordering() -> None:
    y_true = ["Bug", "Documentation", "Enhancement"]
    y_pred = ["Documentation", "Documentation", "Bug"]

    # Force label ordering
    labels = ["Bug", "Documentation", "Enhancement"]
    metrics = compute_metrics(y_true, y_pred, labels=labels)

    cm = metrics["confusion_matrix"]

    # Row 0: True Bug
    # Predicts Documentation -> cm[0][1] should be 1
    assert cm[0][1] == 1
    assert sum(cm[0]) == 1

    # Row 1: True Documentation
    # Predicts Documentation -> cm[1][1] should be 1
    assert cm[1][1] == 1

    # Row 2: True Enhancement
    # Predicts Bug -> cm[2][0] should be 1
    assert cm[2][0] == 1


def test_empty_y_true_raises_error() -> None:
    with pytest.raises(ValueError, match="y_true is empty"):
        compute_metrics([], [])


def test_predicted_class_counts_sum_to_total() -> None:
    y_true = ["Bug", "Documentation", "Enhancement", "Bug"]
    y_pred = ["Bug", "Bug", "Enhancement", "Enhancement"]

    metrics = compute_metrics(y_true, y_pred)

    total_predicted = sum(
        cls_metrics["predicted_count"] for cls_metrics in metrics["per_class"].values()
    )
    assert total_predicted == len(y_pred)
