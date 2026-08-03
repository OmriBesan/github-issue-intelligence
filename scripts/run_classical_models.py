"""CLI script to run classical NLP models."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.evaluation.metrics import compute_metrics
from issue_intelligence.models.classical import get_top_features
from issue_intelligence.models.model_registry import CLASSICAL_MODELS, get_model


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_data(filepath: Path) -> tuple[list[str], list[str]]:
    """Load combined_text and target labels from a JSONL file."""
    X = []
    y = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "combined_text" not in rec or "target" not in rec:
                raise ValueError(f"Missing required fields in {filepath}")
            X.append(rec["combined_text"])
            y.append(rec["target"])

    if not X:
        raise ValueError(f"Empty data in {filepath}")

    return X, y


def plot_confusion_matrix(
    cm: list[list[int]], labels: list[str], title: str, out_path: Path
) -> None:
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_metric_comparison(
    results: dict[str, Any], metric: str, title: str, out_path: Path
) -> None:
    """Plot a bar chart comparing models on a specific metric."""
    models = list(results.keys())
    values = [results[m][metric] for m in models]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=values, y=models, palette="viridis")
    plt.title(title)
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Model")
    plt.xlim(0, 1.0)
    for i, v in enumerate(values):
        plt.text(v + 0.01, i, f"{v:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def run_split_evaluation(
    split_dir: Path,
    split_name: str,
    results_dir: Path,
    figures_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    """Run classical models on a specific split (temporal or random)."""
    logging.info(f"--- Evaluating {split_name.upper()} split ---")

    train_file = split_dir / "train.jsonl"
    val_file = split_dir / "validation.jsonl"

    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)

    expected_labels = {"Bug", "Documentation", "Enhancement"}
    train_labels = set(y_train)
    if not expected_labels.issubset(train_labels):
        raise ValueError(f"Unknown or missing labels in training data: {train_labels}")

    labels_order = ["Bug", "Documentation", "Enhancement"]

    results = {
        "split": split_name,
        "train_size": len(y_train),
        "validation_size": len(y_val),
        "models": {},
    }

    model_metrics = {}

    for model_name in CLASSICAL_MODELS.keys():
        logging.info(f"Fitting {model_name}...")
        pipeline = get_model(model_name)

        t0 = time.time()
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        preds = pipeline.predict(X_val).tolist()
        pred_time = time.time() - t0

        metrics = compute_metrics(y_val, preds, labels=labels_order)
        metrics["training_time_s"] = fit_time
        metrics["prediction_time_s"] = pred_time

        # Vocab size
        tfidf = pipeline.named_steps["tfidf"]
        vocab_size = len(tfidf.vocabulary_) if hasattr(tfidf, "vocabulary_") else 0
        metrics["vocabulary_size"] = vocab_size
        metrics["feature_matrix_dimensions"] = [len(y_train), vocab_size]

        # Top features
        top_feats = get_top_features(pipeline, labels_order, top_n=20)
        if top_feats:
            metrics["top_features"] = top_feats

        results["models"][model_name] = metrics
        model_metrics[model_name] = metrics

        # Confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            labels_order,
            f"Confusion Matrix: {model_name} ({split_name.capitalize()})",
            figures_dir / f"{split_name}_{model_name.lower()}_confusion_matrix.png",
        )

        summary_rows.append(
            {
                "Split": split_name,
                "Model": model_name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "Macro F1": f"{metrics['macro_f1']:.4f}",
                "Weighted F1": f"{metrics['weighted_f1']:.4f}",
                "Vocab Size": vocab_size,
            }
        )

    # Plot comparisons
    plot_metric_comparison(
        model_metrics,
        "macro_f1",
        f"Macro F1 Comparison ({split_name.capitalize()})",
        figures_dir / f"{split_name}_macro_f1_comparison.png",
    )
    plot_metric_comparison(
        model_metrics,
        "accuracy",
        f"Accuracy Comparison ({split_name.capitalize()})",
        figures_dir / f"{split_name}_accuracy_comparison.png",
    )

    # Save JSON results
    out_json = results_dir / f"classical_{split_name}_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved {out_json}")


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Run classical NLP models.")
    parser.add_argument(
        "--splits-dir",
        required=True,
        type=Path,
        help="Directory containing temporal/random splits",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Output directory for JSON/CSV results",
    )
    parser.add_argument(
        "--figures-dir", required=True, type=Path, help="Output directory for plots"
    )

    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    # 1. Temporal Split
    temporal_dir = args.splits_dir / "temporal"
    if temporal_dir.exists():
        run_split_evaluation(
            temporal_dir, "temporal", args.results_dir, args.figures_dir, summary_rows
        )
    else:
        logging.error(f"Temporal split not found at {temporal_dir}")
        sys.exit(1)

    # 2. Random Split
    random_dir = args.splits_dir / "random"
    if random_dir.exists():
        run_split_evaluation(
            random_dir, "random", args.results_dir, args.figures_dir, summary_rows
        )
    else:
        logging.error(f"Random split not found at {random_dir}")
        sys.exit(1)

    # Write summary CSV
    csv_path = args.results_dir / "classical_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Split",
                "Model",
                "Accuracy",
                "Macro F1",
                "Weighted F1",
                "Vocab Size",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    logging.info(f"Saved summary CSV to {csv_path}")

    print("\n====================================================================")
    print("  CLASSICAL MODELS SUMMARY")
    print("====================================================================")
    for row in summary_rows:
        print(
            f"{row['Split'].capitalize():<10} | {row['Model']:<18} | Acc: {row['Accuracy']} | Macro F1: {row['Macro F1']} | W-F1: {row['Weighted F1']}"
        )


if __name__ == "__main__":
    main()
