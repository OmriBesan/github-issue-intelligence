"""CLI script to run simple baselines."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
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
from issue_intelligence.models.baselines import (
    MajorityClassBaseline,
    StratifiedRandomBaseline,
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_labels(filepath: Path) -> list[str]:
    """Load only the target labels from a JSONL file."""
    labels = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            labels.append(rec["target"])
    return labels


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


def run_split_evaluation(
    split_dir: Path,
    split_name: str,
    seed: int,
    results_dir: Path,
    figures_dir: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    """Run baselines on a specific split (temporal or random)."""
    logging.info(f"--- Evaluating {split_name.upper()} split ---")

    train_file = split_dir / "train.jsonl"
    val_file = split_dir / "validation.jsonl"

    y_train = load_labels(train_file)
    y_val = load_labels(val_file)

    # Validate expected labels
    expected_labels = {"Bug", "Documentation", "Enhancement"}
    train_labels = set(y_train)
    if not expected_labels.issubset(train_labels):
        logging.warning(f"Train split missing expected labels! Found: {train_labels}")

    labels_order = ["Bug", "Documentation", "Enhancement"]

    results = {
        "split": split_name,
        "seed": seed,
        "train_size": len(y_train),
        "validation_size": len(y_val),
        "models": {},
    }

    # 1. Majority Class Baseline
    logging.info("Fitting MajorityClassBaseline...")
    maj_model = MajorityClassBaseline()
    maj_model.fit(y_train)
    maj_preds = maj_model.predict(len(y_val))
    maj_metrics = compute_metrics(y_val, maj_preds, labels=labels_order)
    maj_metrics["learned_majority_class"] = max(set(y_train), key=y_train.count)

    results["models"]["majority_class"] = maj_metrics

    plot_confusion_matrix(
        maj_metrics["confusion_matrix"],
        labels_order,
        f"Confusion Matrix: Majority Class ({split_name.capitalize()})",
        figures_dir / f"{split_name}_most_frequent_confusion_matrix.png",
    )

    summary_rows.append(
        {
            "Split": split_name,
            "Model": "Majority Class",
            "Accuracy": f"{maj_metrics['accuracy']:.4f}",
            "Macro F1": f"{maj_metrics['macro_f1']:.4f}",
            "Weighted F1": f"{maj_metrics['weighted_f1']:.4f}",
        }
    )

    # 2. Stratified Random Baseline
    logging.info("Fitting StratifiedRandomBaseline...")
    strat_model = StratifiedRandomBaseline(random_state=seed)
    strat_model.fit(y_train)
    strat_preds = strat_model.predict(len(y_val))

    strat_metrics = compute_metrics(y_val, strat_preds, labels=labels_order)
    results["models"]["stratified_random"] = strat_metrics

    plot_confusion_matrix(
        strat_metrics["confusion_matrix"],
        labels_order,
        f"Confusion Matrix: Stratified Random ({split_name.capitalize()})",
        figures_dir / f"{split_name}_stratified_confusion_matrix.png",
    )

    summary_rows.append(
        {
            "Split": split_name,
            "Model": "Stratified Random",
            "Accuracy": f"{strat_metrics['accuracy']:.4f}",
            "Macro F1": f"{strat_metrics['macro_f1']:.4f}",
            "Weighted F1": f"{strat_metrics['weighted_f1']:.4f}",
        }
    )

    # Save JSON results
    out_json = results_dir / f"baselines_{split_name}_validation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved {out_json}")


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Run baseline models.")
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    # 1. Temporal Split
    temporal_dir = args.splits_dir / "temporal"
    if temporal_dir.exists():
        run_split_evaluation(
            temporal_dir,
            "temporal",
            args.seed,
            args.results_dir,
            args.figures_dir,
            summary_rows,
        )
    else:
        logging.error(f"Temporal split not found at {temporal_dir}")
        sys.exit(1)

    # 2. Random Split
    random_dir = args.splits_dir / "random"
    if random_dir.exists():
        run_split_evaluation(
            random_dir,
            "random",
            args.seed,
            args.results_dir,
            args.figures_dir,
            summary_rows,
        )
    else:
        logging.error(f"Random split not found at {random_dir}")
        sys.exit(1)

    # Write summary CSV
    csv_path = args.results_dir / "baselines_summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Split", "Model", "Accuracy", "Macro F1", "Weighted F1"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    logging.info(f"Saved summary CSV to {csv_path}")

    print("\n====================================================================")
    print("  BASELINES SUMMARY")
    print("====================================================================")
    for row in summary_rows:
        print(
            f"{row['Split'].capitalize():<10} | {row['Model']:<18} | "
            f"Acc: {row['Accuracy']} | Macro F1: {row['Macro F1']} | "
            f"W-F1: {row['Weighted F1']}"
        )


if __name__ == "__main__":
    main()
