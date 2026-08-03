"""Evaluation metrics for issue classification."""

from __future__ import annotations

import collections
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: Optional list of labels to enforce ordering for per-class metrics.

    Returns:
        Dictionary containing all metrics.
    """
    if not y_true:
        raise ValueError("y_true is empty")

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }

    precisions = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    pred_counts = collections.Counter(y_pred)

    for i, label in enumerate(labels):
        metrics["per_class"][label] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "predicted_count": pred_counts.get(label, 0),
        }

    return metrics
