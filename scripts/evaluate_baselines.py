import argparse
import json
import logging
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

from issue_intelligence.baselines import MajorityClassBaseline, RuleBasedBaseline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(file_path: Path) -> tuple[list[str], list[str]]:
    """Loads a JSONL dataset and returns X (combined_text) and y (target)."""
    X = []
    y = []
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        return X, y

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            X.append(record.get("combined_text", ""))
            y.append(record.get("target", "Unknown"))
    return X, y


def evaluate_model(name: str, y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Computes basic metrics for a model's predictions."""
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    logger.info("--- %s ---", name)
    logger.info("Accuracy: %.4f", acc)
    logger.info("Macro F1: %.4f", macro_f1)
    # logger.info("\n" + classification_report(y_true, y_pred, zero_division=0))
    return {"Accuracy": acc, "Macro F1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline models.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing the split JSONL files.",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=Path("reports/baseline_results.md"),
        help="Path to save the markdown evaluation report.",
    )
    args = parser.parse_args()

    # Check if the data dir exists
    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        return

    # We will evaluate on both Temporal and Stratified splits for comparison.
    splits = ["temporal", "stratified"]

    # Store results to write to report later
    report_lines = [
        "# Baseline Model Evaluation Results\n",
        "This report contains the evaluation of simple baseline models "
        "on both the temporal and stratified splits.\n",
        "## Metrics Overview\n",
        "| Split Type | Model | Accuracy | Macro F1 |",
        "|---|---|---|---|",
    ]

    for split_type in splits:
        logger.info("=== Evaluating %s splits ===", split_type.upper())
        train_path = (
            args.data_dir / f"scikit-learn_issues_model_{split_type}_train.jsonl"
        )
        test_path = (
            args.data_dir / f"scikit-learn_issues_model_{split_type}_test.jsonl"
        )

        X_train, y_train = load_dataset(train_path)
        X_test, y_test = load_dataset(test_path)

        if not y_train or not y_test:
            logger.warning("Skipping %s due to missing data.", split_type)
            continue

        # 1. Majority Class Baseline (Custom)
        maj_model = MajorityClassBaseline()
        maj_model.fit(X_train, y_train)
        y_pred_maj = maj_model.predict(X_test)
        res_maj = evaluate_model("Custom Majority Class", y_test, y_pred_maj)
        report_lines.append(
            f"| {split_type.capitalize()} | Custom Majority Class | "
            f"{res_maj['Accuracy']:.4f} | {res_maj['Macro F1']:.4f} |"
        )

        # 2. Rule-Based Baseline
        rule_model = RuleBasedBaseline()
        rule_model.fit(X_train, y_train)
        y_pred_rule = rule_model.predict(X_test)
        res_rule = evaluate_model("Rule-Based Heuristics", y_test, y_pred_rule)
        report_lines.append(
            f"| {split_type.capitalize()} | Rule-Based Heuristics | "
            f"{res_rule['Accuracy']:.4f} | {res_rule['Macro F1']:.4f} |"
        )

        # 3. DummyClassifier (Stratified Random)
        dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
        dummy_strat.fit(X_train, y_train)
        y_pred_dummy = dummy_strat.predict(X_test)
        res_dummy = evaluate_model("sklearn Dummy (Stratified)", y_test, y_pred_dummy)
        report_lines.append(
            f"| {split_type.capitalize()} | sklearn Dummy (Stratified) | "
            f"{res_dummy['Accuracy']:.4f} | {res_dummy['Macro F1']:.4f} |"
        )

        logger.info("======================================\n")

    # Ensure reports directory exists
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Report written to %s", args.out_report)


if __name__ == "__main__":
    main()
