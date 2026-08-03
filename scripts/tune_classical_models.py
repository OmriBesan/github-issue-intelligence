"""Targeted hyperparameter tuning for SGDClassifier and LinearSVC."""

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
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.evaluation.metrics import compute_metrics
from issue_intelligence.evaluation.temporal_cv import (
    make_temporal_folds,
    summarise_folds,
)
from issue_intelligence.models.tuning import (
    AggregatedResult,
    FoldResult,
    aggregate_fold_results,
    get_phase1_candidates,
    get_phase2_candidates,
    select_best,
)

LABELS_ORDER = ["Bug", "Documentation", "Enhancement"]
MODEL_TYPES = ["SGDClassifier", "LinearSVC"]
UNTUNED_SCORES = {"SGDClassifier": 0.9077, "LinearSVC": 0.9060}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_jsonl(filepath: Path) -> list[dict[str, Any]]:
    recs = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    if not recs:
        raise ValueError(f"Empty file: {filepath}")
    return recs


def _evaluate_on_fold(
    candidate: Any,
    train_recs: list[dict[str, Any]],
    val_recs: list[dict[str, Any]],
    fold_index: int,
) -> FoldResult:
    """Fit a fresh pipeline on fold train and evaluate on fold val."""
    X_train = [r["combined_text"] for r in train_recs]
    y_train = [r["target"] for r in train_recs]
    X_val = [r["combined_text"] for r in val_recs]
    y_val = [r["target"] for r in val_recs]

    pipeline = candidate.build_pipeline()

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    preds = pipeline.predict(X_val).tolist()
    pred_time = time.time() - t0

    metrics = compute_metrics(y_val, preds, labels=LABELS_ORDER)
    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)

    return FoldResult(
        candidate=candidate,
        fold_index=fold_index,
        macro_f1=metrics["macro_f1"],
        weighted_f1=metrics["weighted_f1"],
        accuracy=metrics["accuracy"],
        training_time_s=fit_time,
        prediction_time_s=pred_time,
        vocabulary_size=vocab_size,
    )


def run_phase(
    model_type: str,
    candidates: list[Any],
    folds: list[dict[str, Any]],
) -> list[AggregatedResult]:
    """Run all candidates across all folds, return aggregated results."""
    aggregated = []
    for i, candidate in enumerate(candidates):
        fold_results = []
        for fold in folds:
            fr = _evaluate_on_fold(
                candidate,
                fold["train_records"],
                fold["val_records"],
                fold["fold_index"],
            )
            fold_results.append(fr)
        agg = aggregate_fold_results(candidate, fold_results)
        aggregated.append(agg)
        logging.info(
            f"  [{i + 1:>2}/{len(candidates)}] {candidate.config_id[:55]:<55} "
            f"mean_F1={agg.mean_macro_f1:.4f} std={agg.std_macro_f1:.4f}"
        )
    return aggregated


def plot_candidates(
    results: list[AggregatedResult],
    title: str,
    out_path: Path,
    untuned_score: float,
) -> None:
    """Bar chart: mean Macro F1 per candidate with error bars."""
    labels = [r.candidate.config_id[:40] for r in results]
    means = [r.mean_macro_f1 for r in results]
    stds = [r.std_macro_f1 for r in results]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color="steelblue", alpha=0.8)
    ax.axhline(
        untuned_score, color="red", linestyle="--", linewidth=1.5, label="Untuned"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(max(0, min(means) - 0.05), 1.0)
    ax.set_ylabel("Mean Macro F1 (inner folds)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fold_variation(
    best_result: AggregatedResult,
    title: str,
    out_path: Path,
) -> None:
    """Line plot: per-fold Macro F1 for the best configuration."""
    fold_indices = [fr.fold_index for fr in best_result.fold_results]
    fold_f1s = [fr.macro_f1 for fr in best_result.fold_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fold_indices, fold_f1s, marker="o", linewidth=2)
    ax.axhline(
        best_result.mean_macro_f1,
        color="green",
        linestyle="--",
        label=f"Mean={best_result.mean_macro_f1:.4f}",
    )
    ax.set_xlabel("Fold Index")
    ax.set_ylabel("Macro F1")
    ax.set_xticks(fold_indices)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_comparison(
    model_best: dict[str, AggregatedResult],
    val_results: dict[str, dict],
    out_path: Path,
) -> None:
    """Side-by-side: inner-fold mean F1 vs external temporal val F1."""
    models = list(model_best.keys())
    inner_f1 = [model_best[m].mean_macro_f1 for m in models]
    val_f1 = [val_results[m]["macro_f1"] for m in models]
    untuned_f1 = [UNTUNED_SCORES[m] for m in models]

    x = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, untuned_f1, width, label="Untuned (val)", color="salmon")
    ax.bar(x, inner_f1, width, label="Best (inner folds)", color="steelblue")
    ax.bar(x + width, val_f1, width, label="Best (temporal val)", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Macro F1")
    ax.set_title("Untuned vs Tuned: Inner Folds and External Validation")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_time_vs_f1(
    all_results: list[AggregatedResult],
    title: str,
    out_path: Path,
) -> None:
    """Scatter: training time vs mean Macro F1."""
    times = [r.mean_training_time_s for r in all_results]
    f1s = [r.mean_macro_f1 for r in all_results]
    labels = [r.candidate.config_id[:30] for r in all_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(times, f1s, alpha=0.7)
    for x, y, lbl in zip(times, f1s, labels):
        ax.annotate(lbl, (x, y), fontsize=6, alpha=0.7)
    ax.set_xlabel("Mean Training Time (s)")
    ax.set_ylabel("Mean Macro F1")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Tune classical NLP models.")
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load data — TRAINING SPLIT ONLY for inner folds                     #
    # Never access test split.                                             #
    # ------------------------------------------------------------------ #
    logging.info("Loading temporal training split for inner folds …")
    train_recs_all = load_jsonl(args.splits_dir / "temporal" / "train.jsonl")
    train_recs_all.sort(key=lambda r: (r["created_at"], r["issue_id"]))

    # External validation (accessed only AFTER inner tuning is complete)
    logging.info("Loading temporal validation (for final comparison only) …")
    val_recs = load_jsonl(args.splits_dir / "temporal" / "validation.jsonl")
    X_val = [r["combined_text"] for r in val_recs]
    y_val = [r["target"] for r in val_recs]

    # Random validation (for secondary comparison)
    logging.info("Loading random validation (for secondary comparison) …")
    rand_val_recs = load_jsonl(args.splits_dir / "random" / "validation.jsonl")
    X_rand_val = [r["combined_text"] for r in rand_val_recs]
    y_rand_val = [r["target"] for r in rand_val_recs]

    # ------------------------------------------------------------------ #
    # Create inner temporal folds                                          #
    # ------------------------------------------------------------------ #
    logging.info("Creating inner temporal folds …")
    folds = make_temporal_folds(train_recs_all, n_folds=3)
    summarise_folds(folds)

    # ================================================================== #
    # TUNING LOOP                                                          #
    # ================================================================== #
    all_fold_rows: list[dict] = []
    all_candidate_rows: list[dict] = []
    model_best: dict[str, AggregatedResult] = {}
    all_model_results: dict[str, list[AggregatedResult]] = {}

    for model_type in MODEL_TYPES:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"  Phase 1: {model_type}")
        logging.info(f"{'=' * 60}")

        p1_candidates = get_phase1_candidates(model_type)
        p1_results = run_phase(model_type, p1_candidates, folds)
        best_p1 = select_best(p1_results)

        logging.info(
            f"  Best Phase 1 config: {best_p1.candidate.config_id} "
            f"mean_F1={best_p1.mean_macro_f1:.4f}"
        )

        logging.info(f"  Phase 2: {model_type} (TF-IDF variants)")
        p2_candidates = get_phase2_candidates(model_type, best_p1.candidate.clf_params)
        p2_results = run_phase(model_type, p2_candidates, folds)
        all_results = p1_results + p2_results
        all_model_results[model_type] = all_results
        best_overall = select_best(all_results)
        model_best[model_type] = best_overall

        logging.info(
            f"  Best overall: {best_overall.candidate.config_id} "
            f"mean_F1={best_overall.mean_macro_f1:.4f} "
            f"std={best_overall.std_macro_f1:.4f}"
        )

        # Collect CSV rows
        for agg in all_results:
            all_candidate_rows.append(
                {
                    "model": model_type,
                    "config_id": agg.candidate.config_id,
                    "phase": agg.candidate.phase,
                    "mean_macro_f1": f"{agg.mean_macro_f1:.4f}",
                    "std_macro_f1": f"{agg.std_macro_f1:.4f}",
                    "mean_weighted_f1": f"{agg.mean_weighted_f1:.4f}",
                    "mean_accuracy": f"{agg.mean_accuracy:.4f}",
                    "mean_train_time_s": f"{agg.mean_training_time_s:.2f}",
                    "mean_vocab_size": f"{agg.mean_vocabulary_size:.0f}",
                    "clf_params": json.dumps(agg.candidate.clf_params),
                    "tfidf_params": json.dumps(agg.candidate.tfidf_params),
                }
            )
            for fr in agg.fold_results:
                all_fold_rows.append(
                    {
                        "model": model_type,
                        "config_id": agg.candidate.config_id,
                        "fold": fr.fold_index,
                        "macro_f1": f"{fr.macro_f1:.4f}",
                        "weighted_f1": f"{fr.weighted_f1:.4f}",
                        "accuracy": f"{fr.accuracy:.4f}",
                        "train_time_s": f"{fr.training_time_s:.2f}",
                        "vocab_size": fr.vocabulary_size,
                    }
                )

        # Plots
        plot_candidates(
            all_results,
            f"{model_type}: All Candidates (inner-fold Macro F1)",
            args.figures_dir / f"{model_type.lower()}_candidates.png",
            UNTUNED_SCORES[model_type],
        )
        plot_fold_variation(
            best_overall,
            f"{model_type}: Best Config per Fold",
            args.figures_dir / f"{model_type.lower()}_fold_variation.png",
        )
        plot_time_vs_f1(
            all_results,
            f"{model_type}: Training Time vs Macro F1",
            args.figures_dir / f"{model_type.lower()}_time_vs_f1.png",
        )

    # ================================================================== #
    # EXTERNAL VALIDATION (single evaluation after inner tuning)           #
    # ================================================================== #
    logging.info("\n=== External Validation Evaluation ===")
    X_train_full = [r["combined_text"] for r in train_recs_all]
    y_train_full = [r["target"] for r in train_recs_all]

    temporal_val_results: dict[str, dict] = {}
    random_val_results: dict[str, dict] = {}
    comparison_rows: list[dict] = []

    for model_type, best in model_best.items():
        pipeline = best.candidate.build_pipeline()

        t0 = time.time()
        pipeline.fit(X_train_full, y_train_full)
        fit_time = time.time() - t0

        # Temporal validation
        t0 = time.time()
        preds_val = pipeline.predict(X_val).tolist()
        pred_time = time.time() - t0

        tv_metrics = compute_metrics(y_val, preds_val, labels=LABELS_ORDER)
        tv_metrics["training_time_s"] = fit_time
        tv_metrics["prediction_time_s"] = pred_time
        tv_metrics["vocabulary_size"] = len(pipeline.named_steps["tfidf"].vocabulary_)
        temporal_val_results[model_type] = tv_metrics

        # Random validation
        preds_rand = pipeline.predict(X_rand_val).tolist()
        rv_metrics = compute_metrics(y_rand_val, preds_rand, labels=LABELS_ORDER)
        random_val_results[model_type] = rv_metrics

        untuned = UNTUNED_SCORES[model_type]
        delta = tv_metrics["macro_f1"] - untuned
        logging.info(
            f"  {model_type}: temporal_val F1={tv_metrics['macro_f1']:.4f} "
            f"(delta={delta:+.4f})"
        )
        logging.info(f"  {model_type}: random_val F1={rv_metrics['macro_f1']:.4f}")

        comparison_rows.append(
            {
                "model": model_type,
                "config_id": best.candidate.config_id,
                "untuned_temporal_f1": f"{untuned:.4f}",
                "tuned_inner_mean_f1": f"{best.mean_macro_f1:.4f}",
                "tuned_temporal_val_f1": f"{tv_metrics['macro_f1']:.4f}",
                "tuned_random_val_f1": f"{rv_metrics['macro_f1']:.4f}",
                "delta_temporal": f"{delta:+.4f}",
                "vocab_size": tv_metrics["vocabulary_size"],
                "train_time_s": f"{fit_time:.2f}",
            }
        )

    plot_comparison(
        model_best, temporal_val_results, args.figures_dir / "comparison.png"
    )

    # ================================================================== #
    # SAVE ALL RESULTS                                                     #
    # ================================================================== #

    # Fold-level CSV
    with open(
        args.results_dir / "classical_tuning_fold_results.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_fold_rows)

    # All-candidates CSV
    with open(
        args.results_dir / "classical_tuning_all_candidates.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(all_candidate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_candidate_rows)

    # Validation comparison CSV
    with open(
        args.results_dir / "classical_tuning_validation_comparison.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    # Best config JSON
    best_json = {}
    for model_type, best in model_best.items():
        tv = temporal_val_results[model_type]
        rv = random_val_results[model_type]
        best_json[model_type] = {
            "config_id": best.candidate.config_id,
            "clf_params": best.candidate.clf_params,
            "tfidf_params": best.candidate.tfidf_params,
            "inner_fold_mean_macro_f1": best.mean_macro_f1,
            "inner_fold_std_macro_f1": best.std_macro_f1,
            "temporal_val_macro_f1": tv["macro_f1"],
            "temporal_val_accuracy": tv["accuracy"],
            "temporal_val_weighted_f1": tv["weighted_f1"],
            "temporal_val_per_class": tv["per_class"],
            "random_val_macro_f1": rv["macro_f1"],
            "vocabulary_size": tv["vocabulary_size"],
            "training_time_s": tv["training_time_s"],
            "prediction_time_s": tv["prediction_time_s"],
            "untuned_temporal_f1": UNTUNED_SCORES[model_type],
            "delta": tv["macro_f1"] - UNTUNED_SCORES[model_type],
        }

    with open(
        args.results_dir / "classical_tuning_best.json", "w", encoding="utf-8"
    ) as f:
        json.dump(best_json, f, indent=2)

    logging.info("All results saved.")

    # Summary print
    print("\n====================================================================")
    print("  TUNING SUMMARY")
    print("====================================================================")
    for row in comparison_rows:
        print(
            f"{row['model']:<18} | inner_F1={row['tuned_inner_mean_f1']} "
            f"| temporal_val={row['tuned_temporal_val_f1']} "
            f"(delta={row['delta_temporal']}) "
            f"| random_val={row['tuned_random_val_f1']}"
        )


if __name__ == "__main__":
    main()
