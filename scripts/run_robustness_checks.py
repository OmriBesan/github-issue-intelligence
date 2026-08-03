"""Robustness checks for Stage 3B: validate classical model results."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.evaluation.metrics import compute_metrics
from issue_intelligence.evaluation.robustness import (
    extract_body_only,
    extract_misclassified,
    extract_title_only,
    find_exact_duplicates,
    find_near_duplicates,
    get_decision_margins,
    group_by_length_bucket,
    group_by_year,
    margin_from_scores,
    mask_label_words,
    prefix_free_subset,
)
from issue_intelligence.models.model_registry import get_model

LABELS_ORDER = ["Bug", "Documentation", "Enhancement"]
FOCUS_MODELS = ["SGDClassifier", "LinearSVC"]


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


def evaluate(
    X_train: list[str],
    y_train: list[str],
    X_val: list[str],
    y_val: list[str],
    model_name: str,
) -> dict[str, Any]:
    """Fit a fresh model pipeline on X_train and evaluate on X_val."""
    pipeline = get_model(model_name)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val).tolist()
    metrics = compute_metrics(y_val, preds, labels=LABELS_ORDER)
    metrics["model"] = model_name
    return metrics, pipeline, preds


def plot_variant_comparison(
    results: dict[str, dict], title: str, out_path: Path
) -> None:
    """Bar-chart comparing Macro F1 across input variants for each model."""
    variants = list(results.keys())
    model_names = FOCUS_MODELS
    x = np.arange(len(variants))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(model_names):
        values = []
        for v in variants:
            m = results[v].get(model, {})
            values.append(m.get("macro_f1", 0.0))
        ax.bar(x + i * width, values, width, label=model)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0.3376, color="red", linestyle="--", linewidth=1, label="Baseline")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_year_performance(year_results: dict, out_path: Path) -> None:
    """Line plot of per-year Macro F1."""
    years = sorted(year_results.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in FOCUS_MODELS:
        values = [year_results[y].get(model, {}).get("macro_f1", 0.0) for y in years]
        ax.plot(years, values, marker="o", label=model)
    ax.set_xlabel("Validation Year")
    ax.set_ylabel("Macro F1")
    ax.set_title("Per-Year Macro F1 (Temporal Validation)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_length_performance(length_results: dict, out_path: Path) -> None:
    """Bar chart of Macro F1 by text-length bucket."""
    bucket_order = ["short", "medium", "long", "very_long"]
    buckets = [b for b in bucket_order if b in length_results]
    x = np.arange(len(buckets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(FOCUS_MODELS):
        values = [
            length_results[b].get(model, {}).get("macro_f1", 0.0) for b in buckets
        ]
        counts = [length_results[b].get("count", 0) for b in buckets]
        bars = ax.bar(x + i * width, values, width, label=model)
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={cnt}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(buckets)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 by Text-Length Bucket")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_margin_distribution(margins: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(margins, bins=40, edgecolor="white")
    ax.set_xlabel("Decision Margin (best - second best)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_near_dup_histogram(sims: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=40, range=(0, 1), edgecolor="white")
    ax.set_xlabel("Max Train Cosine Similarity")
    ax.set_ylabel("# Validation Docs")
    ax.set_title("Near-Duplicate Similarity Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(description="Run Stage 3B robustness checks.")
    parser.add_argument("--splits-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load temporal split only — NEVER load test split                    #
    # ------------------------------------------------------------------ #
    logging.info("Loading temporal train and validation …")
    train_recs = load_jsonl(args.splits_dir / "temporal" / "train.jsonl")
    val_recs = load_jsonl(args.splits_dir / "temporal" / "validation.jsonl")
    logging.info(f"  Train: {len(train_recs)}  Val: {len(val_recs)}")

    y_train = [r["target"] for r in train_recs]
    y_val = [r["target"] for r in val_recs]

    # ================================================================== #
    # EXPERIMENT 1-4: INPUT VARIANTS                                      #
    # ================================================================== #
    input_variants = {
        "combined (repro)": (
            [r["combined_text"] for r in train_recs],
            [r["combined_text"] for r in val_recs],
        ),
        "title_only": (
            [extract_title_only(r) for r in train_recs],
            [extract_title_only(r) for r in val_recs],
        ),
        "body_only": (
            [extract_body_only(r) for r in train_recs],
            [extract_body_only(r) for r in val_recs],
        ),
        "label_masked": (
            [mask_label_words(r["combined_text"]) for r in train_recs],
            [mask_label_words(r["combined_text"]) for r in val_recs],
        ),
    }

    variant_results: dict[str, dict] = {}
    for variant_name, (X_tr, X_v) in input_variants.items():
        logging.info(f"Variant: {variant_name}")
        variant_results[variant_name] = {}
        for model_name in FOCUS_MODELS:
            metrics, _, _ = evaluate(X_tr, y_train, X_v, y_val, model_name)
            variant_results[variant_name][model_name] = metrics
            logging.info(
                f"  {model_name:<18} Macro F1: {metrics['macro_f1']:.4f}"
                f"  Acc: {metrics['accuracy']:.4f}"
            )

    plot_variant_comparison(
        variant_results,
        "Macro F1 by Input Variant (Temporal Val)",
        args.figures_dir / "variant_macro_f1.png",
    )

    # ================================================================== #
    # EXPERIMENT 5: PREFIX-FREE SUBSET                                    #
    # ================================================================== #
    logging.info("Experiment 5: prefix-free subset …")
    try:
        pf_val_recs = prefix_free_subset(val_recs)
        pf_y_val = [r["target"] for r in pf_val_recs]
        pf_X_val = [r["combined_text"] for r in pf_val_recs]
        X_train_combined = [r["combined_text"] for r in train_recs]

        prefix_free_results: dict[str, dict] = {}
        for model_name in FOCUS_MODELS:
            pipeline = get_model(model_name)
            pipeline.fit(X_train_combined, y_train)
            pf_preds = pipeline.predict(pf_X_val).tolist()
            pf_metrics = compute_metrics(pf_y_val, pf_preds, labels=LABELS_ORDER)
            prefix_free_results[model_name] = pf_metrics
            logging.info(
                f"  {model_name:<18} PrefixFree Macro F1: {pf_metrics['macro_f1']:.4f}"
                f"  n={len(pf_val_recs)}"
            )
    except ValueError as e:
        logging.warning(f"Prefix-free subset error: {e}")
        prefix_free_results = {}

    # ================================================================== #
    # EXPERIMENT 6: TEXT-LENGTH BUCKETS                                   #
    # ================================================================== #
    logging.info("Experiment 6: text-length buckets …")
    bucket_groups = group_by_length_bucket(val_recs)
    length_bucket_results: dict[str, dict] = {}

    X_train_combined = [r["combined_text"] for r in train_recs]
    # Fit once on combined training data
    fitted_pipelines = {}
    for model_name in FOCUS_MODELS:
        p = get_model(model_name)
        p.fit(X_train_combined, y_train)
        fitted_pipelines[model_name] = p

    for bucket, brecs in bucket_groups.items():
        if len(brecs) < 3:
            logging.warning(f"Skipping bucket '{bucket}': only {len(brecs)} records")
            continue
        bX = [r["combined_text"] for r in brecs]
        by = [r["target"] for r in brecs]
        length_bucket_results[bucket] = {"count": len(brecs)}
        for model_name in FOCUS_MODELS:
            preds = fitted_pipelines[model_name].predict(bX).tolist()
            m = compute_metrics(by, preds, labels=LABELS_ORDER)
            length_bucket_results[bucket][model_name] = m
            logging.info(
                f"  Bucket={bucket:<10} {model_name:<18} "
                f"Macro F1: {m['macro_f1']:.4f}  n={len(brecs)}"
            )

    plot_length_performance(
        length_bucket_results, args.figures_dir / "length_bucket_macro_f1.png"
    )

    # ================================================================== #
    # EXPERIMENT 7: PERFORMANCE BY YEAR                                   #
    # ================================================================== #
    logging.info("Experiment 7: performance by validation year …")
    year_groups = group_by_year(val_recs)
    year_results: dict[str, dict] = {}

    for year, yrecs in sorted(year_groups.items()):
        yX = [r["combined_text"] for r in yrecs]
        yy = [r["target"] for r in yrecs]
        if len(set(yy)) < 2:
            logging.warning(f"Skipping year {year}: fewer than 2 classes")
            continue
        year_results[year] = {}
        for model_name in FOCUS_MODELS:
            preds = fitted_pipelines[model_name].predict(yX).tolist()
            m = compute_metrics(yy, preds, labels=LABELS_ORDER)
            year_results[year][model_name] = m
            logging.info(
                f"  Year={year} {model_name:<18} "
                f"Macro F1: {m['macro_f1']:.4f}  n={len(yrecs)}"
            )

    plot_year_performance(year_results, args.figures_dir / "year_macro_f1.png")

    # ================================================================== #
    # EXPERIMENT 8: NEAR-DUPLICATE INVESTIGATION                          #
    # ================================================================== #
    logging.info("Experiment 8: near-duplicate investigation …")
    train_texts = [r["combined_text"] for r in train_recs]
    val_texts = [r["combined_text"] for r in val_recs]
    train_ids = [r["issue_id"] for r in train_recs]
    val_ids = [r["issue_id"] for r in val_recs]

    exact_dupes = find_exact_duplicates(train_texts, val_texts, train_ids, val_ids)
    logging.info(f"  Exact duplicates: {len(exact_dupes)}")

    # Use the SGD pipeline TF-IDF for near-dup similarity
    sgd_pipe = fitted_pipelines["SGDClassifier"]
    tfidf = sgd_pipe.named_steps["tfidf"]
    train_matrix = tfidf.transform(train_texts)
    val_matrix = tfidf.transform(val_texts)

    near_dup_results = find_near_duplicates(
        train_matrix,
        val_matrix,
        train_ids,
        val_ids,
        thresholds=(0.90, 0.95, 0.99),
    )
    for thresh, pairs in near_dup_results.items():
        logging.info(f"  Near-duplicates @{thresh}: {len(pairs)}")

    # Compute max similarity for each val doc (for histogram)
    logging.info("  Computing max train similarities for histogram …")
    max_sims = []
    BATCH = 100
    for start in range(0, val_matrix.shape[0], BATCH):
        end = min(start + BATCH, val_matrix.shape[0])
        sims = cosine_similarity(val_matrix[start:end], train_matrix)
        max_sims.extend(sims.max(axis=1).tolist())

    plot_near_dup_histogram(max_sims, args.figures_dir / "near_dup_similarity.png")

    near_dup_csv_rows = [
        {
            "threshold": thresh,
            "n_pairs": len(pairs),
            "pct_of_val": f"{100 * len(pairs) / len(val_recs):.1f}%",
        }
        for thresh, pairs in near_dup_results.items()
    ]
    near_dup_csv_rows.insert(
        0,
        {
            "threshold": "exact",
            "n_pairs": len(exact_dupes),
            "pct_of_val": f"{100 * len(exact_dupes) / len(val_recs):.1f}%",
        },
    )

    # ================================================================== #
    # EXPERIMENT 9: DECISION MARGINS                                      #
    # ================================================================== #
    logging.info("Experiment 9: decision margins …")
    val_combined = [r["combined_text"] for r in val_recs]
    margin_data: dict[str, Any] = {}

    for model_name in FOCUS_MODELS:
        pipe = fitted_pipelines[model_name]
        scores = get_decision_margins(pipe, val_combined)
        if scores is not None:
            margins = margin_from_scores(scores)
            preds = pipe.predict(val_combined).tolist()
            correct_mask = np.array([p == t for p, t in zip(preds, y_val)])
            margin_data[model_name] = {
                "margins": margins,
                "preds": preds,
                "correct_mask": correct_mask,
            }
            avg_correct = (
                float(margins[correct_mask].mean()) if correct_mask.any() else 0.0
            )
            avg_wrong = (
                float(margins[~correct_mask].mean()) if (~correct_mask).any() else 0.0
            )
            logging.info(
                f"  {model_name}: avg margin "
                f"correct={avg_correct:.4f}  wrong={avg_wrong:.4f}"
            )
            plot_margin_distribution(
                margins,
                f"{model_name} Decision Margins",
                args.figures_dir / f"{model_name.lower()}_margins.png",
            )

    # ================================================================== #
    # EXPERIMENT 10: MISCLASSIFICATION REVIEW                             #
    # ================================================================== #
    logging.info("Experiment 10: misclassification review …")
    misclassified_examples: list[dict[str, Any]] = []
    for model_name in FOCUS_MODELS:
        pipe = fitted_pipelines[model_name]
        preds = pipe.predict(val_combined).tolist()
        margins_arr = margin_data.get(model_name, {}).get("margins")
        examples = extract_misclassified(
            val_recs, y_val, preds, margins=margins_arr, n_per_class=15
        )
        for e in examples:
            e["model"] = model_name
        misclassified_examples.extend(examples)

    # ================================================================== #
    # INFLUENTIAL-TERM REVIEW (top 20 per class)                          #
    # ================================================================== #
    logging.info("Collecting top-20 influential terms …")
    from issue_intelligence.models.classical import get_top_features

    term_rows: list[dict[str, Any]] = []
    for model_name in FOCUS_MODELS:
        pipe = fitted_pipelines[model_name]
        top_feats = get_top_features(pipe, LABELS_ORDER, top_n=20)
        for label, terms in top_feats.items():
            for rank, (term, coef) in enumerate(terms, 1):
                term_rows.append(
                    {
                        "model": model_name,
                        "class": label,
                        "rank": rank,
                        "term": term,
                        "coefficient": f"{coef:.4f}",
                    }
                )

    # ================================================================== #
    # SAVE ALL RESULTS                                                     #
    # ================================================================== #

    # robustness_details.json
    details = {
        "variant_results": {
            v: {
                m: {
                    "macro_f1": mets["macro_f1"],
                    "accuracy": mets["accuracy"],
                    "per_class": mets["per_class"],
                }
                for m, mets in models.items()
            }
            for v, models in variant_results.items()
        },
        "prefix_free_results": {
            m: {"macro_f1": mets["macro_f1"], "per_class": mets["per_class"]}
            for m, mets in prefix_free_results.items()
        },
        "year_results": {
            yr: {
                m: {"macro_f1": mets["macro_f1"], "per_class": mets["per_class"]}
                for m, mets in models.items()
            }
            for yr, models in year_results.items()
        },
        "length_bucket_results": {
            bkt: {
                m: {
                    "macro_f1": mets.get("macro_f1", 0),
                    "n": length_bucket_results[bkt]["count"],
                }
                for m, mets in bkt_data.items()
                if m in FOCUS_MODELS
            }
            for bkt, bkt_data in length_bucket_results.items()
        },
        "near_duplicates": {
            "exact_count": len(exact_dupes),
            "near_dup_counts": {str(t): len(p) for t, p in near_dup_results.items()},
        },
        "max_train_similarities": {
            "mean": float(np.mean(max_sims)),
            "median": float(np.median(max_sims)),
            "p90": float(np.percentile(max_sims, 90)),
            "p95": float(np.percentile(max_sims, 95)),
            "p99": float(np.percentile(max_sims, 99)),
        },
    }
    with open(args.results_dir / "robustness_details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    # robustness_summary.csv
    summary_rows = []
    for variant, models in variant_results.items():
        for model, mets in models.items():
            summary_rows.append(
                {
                    "experiment": f"input_variant:{variant}",
                    "model": model,
                    "macro_f1": f"{mets['macro_f1']:.4f}",
                    "accuracy": f"{mets['accuracy']:.4f}",
                    "n_val": len(val_recs),
                }
            )
    for model, mets in prefix_free_results.items():
        summary_rows.append(
            {
                "experiment": "prefix_free_subset",
                "model": model,
                "macro_f1": f"{mets['macro_f1']:.4f}",
                "accuracy": f"{mets['accuracy']:.4f}",
                "n_val": len(pf_val_recs) if pf_val_recs else 0,
            }
        )
    for yr, models in year_results.items():
        for model, mets in models.items():
            n = len(year_groups[yr])
            summary_rows.append(
                {
                    "experiment": f"year:{yr}",
                    "model": model,
                    "macro_f1": f"{mets['macro_f1']:.4f}",
                    "accuracy": f"{mets['accuracy']:.4f}",
                    "n_val": n,
                }
            )
    for bkt, bkt_data in length_bucket_results.items():
        cnt = bkt_data["count"]
        for m in FOCUS_MODELS:
            mets = bkt_data.get(m, {})
            if mets:
                summary_rows.append(
                    {
                        "experiment": f"length_bucket:{bkt}",
                        "model": m,
                        "macro_f1": f"{mets.get('macro_f1', 0):.4f}",
                        "accuracy": f"{mets.get('accuracy', 0):.4f}",
                        "n_val": cnt,
                    }
                )

    with open(
        args.results_dir / "robustness_summary.csv", "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["experiment", "model", "macro_f1", "accuracy", "n_val"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # misclassified_examples.csv
    with open(
        args.results_dir / "misclassified_examples.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fields = [
            "model",
            "issue_id",
            "true_label",
            "predicted_label",
            "margin",
            "clean_title",
            "combined_length",
            "created_at",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(misclassified_examples)

    # near_duplicate_review.csv
    with open(
        args.results_dir / "near_duplicate_review.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "n_pairs", "pct_of_val"])
        writer.writeheader()
        writer.writerows(near_dup_csv_rows)

    # influential_term_review.csv
    with open(
        args.results_dir / "influential_term_review.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "class", "rank", "term", "coefficient"]
        )
        writer.writeheader()
        writer.writerows(term_rows)

    logging.info("All results saved.")

    # Print summary
    print("\n====================================================================")
    print("  ROBUSTNESS SUMMARY")
    print("====================================================================")
    for row in summary_rows:
        print(
            f"{row['experiment']:<35} | {row['model']:<18} | "
            f"Macro F1: {row['macro_f1']} | n={row['n_val']}"
        )


if __name__ == "__main__":
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: F811

    main()
