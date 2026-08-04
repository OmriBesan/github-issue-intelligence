from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """
    Computes classification metrics including precision, recall, and f1-score.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        A dictionary containing the classification report.
    """
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def plot_confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str] = None
) -> plt.Figure:
    """
    Plots a confusion matrix and returns the figure.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of labels to include in the matrix. If None, infers from data.

    Returns:
        The matplotlib Figure object containing the plot.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return fig


def extract_errors(
    X: list[str], y_true: list[str], y_pred: list[str]
) -> list[dict[str, str]]:
    """
    Extracts instances where the prediction did not match the ground truth.

    Args:
        X: The input texts.
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        A list of dictionaries with text, true_label, and pred_label for errors.
    """
    errors = []
    for text, true_label, pred_label in zip(X, y_true, y_pred, strict=False):
        if true_label != pred_label:
            errors.append(
                {
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                }
            )
    return errors
