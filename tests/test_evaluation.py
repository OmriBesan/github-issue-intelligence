import matplotlib.pyplot as plt

from issue_intelligence.evaluation import (
    compute_metrics,
    extract_errors,
    plot_confusion_matrix,
)


def test_compute_metrics():
    y_true = ["Bug", "Bug", "Enhancement"]
    y_pred = ["Bug", "Enhancement", "Enhancement"]

    metrics = compute_metrics(y_true, y_pred)
    assert "Bug" in metrics
    assert "Enhancement" in metrics
    assert "accuracy" in metrics
    assert metrics["Bug"]["recall"] == 0.5


def test_plot_confusion_matrix():
    y_true = ["Bug", "Documentation", "Enhancement"]
    y_pred = ["Bug", "Bug", "Enhancement"]

    fig = plot_confusion_matrix(y_true, y_pred)
    assert isinstance(fig, plt.Figure)
    # Check that it plotted axes
    assert len(fig.axes) > 0


def test_extract_errors():
    X = ["text1", "text2", "text3"]
    y_true = ["Bug", "Bug", "Enhancement"]
    y_pred = ["Bug", "Enhancement", "Enhancement"]

    errors = extract_errors(X, y_true, y_pred)
    assert len(errors) == 1
    assert errors[0]["text"] == "text2"
    assert errors[0]["true_label"] == "Bug"
    assert errors[0]["predicted_label"] == "Enhancement"
