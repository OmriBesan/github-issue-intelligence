"""Evaluate the final locked LinearSVC model on the held-out temporal test set."""

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Fixed label order for all outputs
LABELS = ["Bug", "Documentation", "Enhancement"]


def load_split(path: Path) -> pd.DataFrame:
    """Load a JSONL split."""
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_json(path, lines=True)


def build_final_pipeline() -> Pipeline:
    """Build the exact locked LinearSVC pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=None,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    C=0.3,
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000,
                    dual=False,
                ),
            ),
        ]
    )


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Final Temporal Test - Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_validation_vs_test(
    val_macro_f1: float, test_macro_f1: float, output_path: Path
):
    """Plot temporal validation vs test Macro F1."""
    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        ["Temporal Validation\n(2022-2024)", "Temporal Test\n(2024-2026)"],
        [val_macro_f1, test_macro_f1],
        color=["#2c3e50", "#27ae60"],
    )
    plt.ylim(0, 1.0)
    plt.ylabel("Macro F1 Score")
    plt.title("LinearSVC Performance Across Time Periods")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate final model on test set.")
    parser.add_argument(
        "--splits-dir", type=Path, default=Path("data/processed/splits")
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models/classical"))
    parser.add_argument("--results-dir", type=Path, default=Path("reports/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    # 1. Load train and val
    logger.info("Loading splits...")
    train_df = load_split(args.splits_dir / "temporal" / "train.jsonl")
    val_df = load_split(args.splits_dir / "temporal" / "validation.jsonl")

    # 3. Combine train and val
    final_train_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(
        f"Combined train ({len(train_df)}) + val ({len(val_df)}) "
        f"= {len(final_train_df)} records"
    )

    # 4. Verify no duplicates in combined train
    if final_train_df["issue_id"].duplicated().any():
        raise ValueError("Found duplicate issue_ids in the combined training set!")

    # 7. Load test set
    test_df = load_split(args.splits_dir / "temporal" / "test.jsonl")
    logger.info(f"Loaded temporal test set: {len(test_df)} records")
    if len(test_df) != 856:
        raise ValueError(f"Expected exactly 856 test records, got {len(test_df)}")

    # 5. Verify no overlap between train and test
    train_ids = set(final_train_df["issue_id"])
    test_ids = set(test_df["issue_id"])
    overlap = train_ids.intersection(test_ids)
    if overlap:
        raise ValueError(
            f"Found {len(overlap)} issue_ids overlapping between train and test!"
        )

    # Prep data
    X_train = final_train_df["combined_text"].fillna("").tolist()
    y_train = final_train_df["target"].tolist()
    X_test = test_df["combined_text"].fillna("").tolist()
    y_test = test_df["target"].tolist()

    # 6. Fit locked pipeline
    logger.info("Fitting final locked pipeline...")
    pipeline = build_final_pipeline()
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    # Inspect pipeline
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    # tfidf matrix shape from transform
    X_train_mat = pipeline.named_steps["tfidf"].transform(X_train)
    n_samples, n_features = X_train_mat.shape

    logger.info(
        f"Fitted in {train_time:.2f}s "
        f"(vocab={vocab_size:,}, shape={n_samples}x{n_features})"
    )

    # 8. Generate test predictions
    logger.info("Generating test predictions...")
    t0 = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time = time.time() - t0

    # 9. Calculate metrics
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, output_dict=True, labels=LABELS, target_names=LABELS
    )

    # Pred distribution
    pred_series = pd.Series(y_pred).value_counts().to_dict()
    pred_dist = {label: int(pred_series.get(label, 0)) for label in LABELS}

    # Format result dict
    results = {
        "model_name": "LinearSVC",
        "configuration": {
            "C": 0.3,
            "class_weight": "balanced",
            "tfidf_ngram_range": [1, 2],
            "tfidf_min_df": 5,
        },
        "training_time_s": train_time,
        "inference_time_s": inference_time,
        "vocabulary_size": vocab_size,
        "feature_matrix_shape": [n_samples, n_features],
        "test_metrics": {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": accuracy,
            "per_class": {
                label: {
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
                for label in LABELS
            },
        },
        "predicted_distribution": pred_dist,
    }

    # Save model
    args.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.models_dir / "final_linear_svc.joblib"
    joblib.dump(pipeline, model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    results["model_size_mb"] = model_size_mb

    # 10. Save outputs
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(
        args.results_dir / "final_temporal_test.json", "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    # Save CSV summary
    summary = pd.DataFrame(
        [
            {
                "Model": "LinearSVC (Final)",
                "Temporal_Validation_Macro_F1": 0.9087,
                "Temporal_Test_Macro_F1": macro_f1,
                "Delta": macro_f1 - 0.9087,
            }
        ]
    )
    summary.to_csv(args.results_dir / "final_model_summary.csv", index=False)

    # Generate plots
    plot_confusion_matrix(
        y_test, y_pred, args.figures_dir / "final_test_confusion_matrix.png"
    )
    plot_validation_vs_test(
        0.9087, macro_f1, args.figures_dir / "final_validation_vs_test.png"
    )

    logger.info(f"Test Macro F1: {macro_f1:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info("All final evaluation artifacts saved.")


if __name__ == "__main__":
    main()
