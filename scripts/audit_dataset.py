"""
scripts/audit_dataset.py
=========================
Command-line script for Stage 1B: runs the full dataset audit and
generates all figures outside of Jupyter.

Usage
-----
.venv\\Scripts\\python scripts\\audit_dataset.py ^
    --input   data\\raw\\scikit-learn_issues_sample.json ^
    --metadata data\\raw\\scikit-learn_issues_sample_metadata.json ^
    --figures-dir reports\\figures

Run with --help for full argument descriptions.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Switch to the non-interactive Agg backend so the script runs without
# a display (plt.switch_backend works in matplotlib 3.3+).
plt.switch_backend("Agg")

# Make the src/ package importable when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.audit import (  # noqa: E402
    PROVISIONAL_TYPE_LABELS,
    build_label_scheme_stats,
    build_provisional_subset,
    categorize_labels_provisionally,
    compute_label_cooccurrence,
    compute_label_distribution_by_year,
    compute_label_frequencies,
    compute_labels_per_issue,
    compute_missing_data,
    compute_state_distribution,
    compute_temporal_distribution,
    compute_text_lengths,
    compute_yearly_distribution,
    describe_lengths,
    detect_leakage,
    load_issues,
    load_metadata,
)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit_dataset.py",
        description=(
            "Run the Stage 1B dataset audit on collected GitHub issues. "
            "Prints a text summary and saves figures to the specified directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to the issues JSON file "
            "(e.g. data/raw/scikit-learn_issues_sample.json)."
        ),
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to the metadata JSON file.",
    )
    parser.add_argument(
        "--figures-dir",
        default="reports/figures",
        help="Directory where PNG figures will be saved (default: reports/figures).",
    )
    return parser


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

FIG_W = 10   # default figure width (inches)
FIG_H = 6    # default figure height (inches)
DPI   = 150  # output resolution


def _save(fig: plt.Figure, path: Path, label: str) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved {label} -> {path.name}")


def make_label_frequency_chart(freq: Counter, figures_dir: Path) -> None:
    top = freq.most_common(30)
    if not top:
        return
    labels, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(FIG_W, 9))
    y = range(len(labels))
    ax.barh(list(y), list(counts), color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(list(labels), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of issues")
    ax.set_title("Top 30 Label Frequencies — scikit-learn issues sample")
    ax.grid(axis="x", alpha=0.35)
    _save(fig, figures_dir / "top_label_frequencies.png", "top_label_frequencies")


def make_labels_per_issue_chart(counts: list[int], figures_dir: Path) -> None:
    counter = Counter(counts)
    max_labels = max(counter)
    x = list(range(max_labels + 1))
    y = [counter.get(i, 0) for i in x]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, y, color="#55A868", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of labels per issue")
    ax.set_ylabel("Number of issues")
    ax.set_title("Label Count Distribution per Issue")
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.35)
    _save(fig, figures_dir / "labels_per_issue_distribution.png", "labels_per_issue")


def make_type_distribution_chart(class_counts: Counter, figures_dir: Path) -> None:
    if not class_counts:
        return
    items = sorted(class_counts.items(), key=lambda x: -x[1])
    labels, counts = zip(*items)
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color="#C44E52", edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{count}\n({100*count/total:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("Number of issues")
    ax.set_title("Provisional Issue-Type Distribution (single-type subset)")
    ax.grid(axis="y", alpha=0.35)
    plt.xticks(rotation=15, ha="right")
    _save(fig, figures_dir / "type_label_distribution.png", "type_label_distribution")


def make_temporal_chart(monthly: Counter, figures_dir: Path) -> None:
    if not monthly:
        return
    months = sorted(monthly)
    counts = [monthly[m] for m in months]
    fig, ax = plt.subplots(figsize=(FIG_W, 5))
    ax.bar(months, counts, color="#8172B2", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of issues created")
    ax.set_title("Issue Creation by Month")
    ax.grid(axis="y", alpha=0.35)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _save(fig, figures_dir / "creation_dates_by_month.png", "temporal_distribution")


def make_text_length_chart(lengths: dict[str, list[int]], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels_map = {
        "title": "Title length (chars)",
        "body": "Body length (chars)",
        "combined": "Combined length (chars)",
    }
    for ax, key in zip(axes, ["title", "body", "combined"]):
        data = lengths[key]
        # Cap at 99th percentile for readability
        cap = int(np.percentile(data, 99))
        clipped = [min(v, cap) for v in data]
        ax.hist(clipped, bins=40, color="#64B5CD", edgecolor="white", linewidth=0.4)
        ax.set_xlabel(labels_map[key])
        ax.set_ylabel("Issues")
        ax.set_title(f"{labels_map[key]}\n(capped at 99th pct = {cap})")
        ax.grid(axis="y", alpha=0.35)
    fig.suptitle("Text Length Distributions", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, figures_dir / "text_length_distribution.png", "text_lengths")


def make_cooccurrence_heatmap(
    issues: list[dict],
    freq: Counter,
    figures_dir: Path,
    top_n: int = 20,
) -> None:
    """Build a co-occurrence heatmap for the top-N labels."""
    top_labels = [lb for lb, _ in freq.most_common(top_n)]
    cooc = compute_label_cooccurrence(issues, selected_labels=top_labels)

    n = len(top_labels)
    matrix = np.zeros((n, n), dtype=int)
    label_idx = {lb: i for i, lb in enumerate(top_labels)}

    for (la, lb), count in cooc.items():
        if la in label_idx and lb in label_idx:
            i, j = label_idx[la], label_idx[lb]
            matrix[i, j] = count
            matrix[j, i] = count

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_labels, fontsize=8)
    # Annotate cells with count (only if > 0)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() * 0.5 else "black",
                )
    plt.colorbar(im, ax=ax, label="Co-occurrence count")
    ax.set_title(f"Label Co-occurrence Heatmap (top {top_n} labels)")
    plt.tight_layout()
    _save(fig, figures_dir / "label_cooccurrence_heatmap.png", "cooccurrence_heatmap")


def make_yearly_chart(yearly: Counter, figures_dir: Path) -> None:
    """Bar chart of issues per year — key for historical coverage check."""
    if not yearly:
        return
    years = sorted(yearly.keys())
    counts = [yearly[y] for y in years]
    fig, ax = plt.subplots(figsize=(max(8, len(years) * 0.7), 5))
    bars = ax.bar(years, counts, color="#5B8DB8", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fontsize=8, padding=3)
    ax.set_xlabel("Year")
    ax.set_ylabel("Issues")
    ax.set_title("Issues Created per Year")
    ax.grid(axis="y", alpha=0.35)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(fig, figures_dir / "issues_per_year.png", "yearly_distribution")


def make_label_by_year_chart(
    issues: list[dict],
    type_labels: list[str],
    figures_dir: Path,
) -> None:
    """Stacked area chart — how each type label's usage evolved by year."""
    by_year = compute_label_distribution_by_year(issues, type_labels)
    if not by_year:
        return
    years = sorted(by_year.keys())
    fig, ax = plt.subplots(figsize=(max(8, len(years) * 0.8), 5))
    colors = ["#E05A54", "#5B8DB8", "#5CAD6A", "#D4A843", "#9B6BB5"]
    bottom = [0] * len(years)
    for label, color in zip(type_labels, colors):
        counts = [by_year.get(y, {}).get(label, 0) for y in years]
        ax.bar(years, counts, bottom=bottom, label=label,
               color=color, edgecolor="white", linewidth=0.3)
        bottom = [b + c for b, c in zip(bottom, counts)]
    ax.set_xlabel("Year")
    ax.set_ylabel("Single-type issues")
    ax.set_title("Type Label Distribution by Year")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save(fig, figures_dir / "type_labels_by_year.png", "label_by_year")


# ---------------------------------------------------------------------------
# Text summary printer
# ---------------------------------------------------------------------------


def print_summary(
    issues: list[dict],
    metadata: dict,
    freq: Counter,
    missing: dict,
    lengths: dict,
    state_dist: Counter,
    monthly: Counter,
    leakage: dict,
    subset_result: dict,
    categories: dict,
) -> None:
    total = len(issues)
    lpi = compute_labels_per_issue(issues)
    with_labels = sum(1 for c in lpi if c > 0)
    multi_label = sum(1 for c in lpi if c > 1)

    # Date coverage
    yearly = compute_yearly_distribution(issues)
    years = sorted(yearly.keys())
    n_years = (int(years[-1]) - int(years[0]) + 1) if len(years) >= 2 else 1

    print("\n" + "=" * 65)
    print("  DATASET AUDIT SUMMARY")
    print("=" * 65)
    print(f"  Repository              : {metadata.get('repository')}")
    print(f"  Collection timestamp    : {metadata.get('collection_timestamp')}")
    sort_val = metadata.get('sort', 'created')
    dir_val = metadata.get('direction', 'desc')
    print(f"  Sort / direction        : {sort_val} / {dir_val}")
    print(f"  State filter            : {metadata.get('state_filter')}")
    print()
    print("  --- Overview ---")
    print(f"  Total issues            : {total}")
    for state, cnt in sorted(state_dist.items()):
        print(f"  State [{state:<6}]        : {cnt}  ({100*cnt/total:.1f}%)")
    print()
    print("  --- Date Coverage ---")
    earliest = min(years) if years else "N/A"
    latest = max(years) if years else "N/A"
    print(f"  Earliest year           : {earliest}")
    print(f"  Latest year             : {latest}")
    print(f"  Years covered           : {n_years}")
    print(f"  Metadata earliest date  : {metadata.get('earliest_created_at', 'N/A')}")
    print(f"  Metadata latest date    : {metadata.get('latest_created_at', 'N/A')}")
    print()
    if years:
        print("  Issues per year:")
        for y in years:
            bar_len = int(30 * yearly[y] / max(yearly.values()))
            print(f"    {y}: {yearly[y]:>5}  {'|' * bar_len}")
    print()
    print("  --- Labels ---")
    without = total - with_labels
    pct_without = 100 * without / total
    print(f"  With labels             : {with_labels}  ({100*with_labels/total:.1f}%)")
    print(f"  Without labels          : {without}  ({pct_without:.1f}%)")
    print(f"  With multiple labels    : {multi_label}  ({100*multi_label/total:.1f}%)")
    print(f"  Unique label names      : {len(freq)}")
    print()
    print("  --- Missing Data ---")
    print(f"  Missing/empty title     : {missing['total_title_problems']}")
    print(f"  Missing/empty body      : {missing['total_body_problems']}")
    print()
    print("  --- Text Lengths (combined title+body) ---")
    desc = describe_lengths(lengths["combined"])
    print(f"  Min                     : {int(desc['min'])} chars")
    print(f"  Median                  : {int(desc['median'])} chars")
    print(f"  Mean                    : {desc['mean']:.0f} chars")
    print(f"  95th percentile         : {int(desc['p95'])} chars")
    print(f"  Max                     : {int(desc['max'])} chars")
    print()
    print("  --- Leakage Check ---")
    print(f"  Titles with type prefix : {len(leakage['prefix_matches'])}")
    print(f"  Titles with label name  : {len(leakage['label_in_title'])}")
    print()
    print("  --- Provisional Type Subset (original 5-class scheme) ---")
    print(f"  Type labels used        : {sorted(PROVISIONAL_TYPE_LABELS)}")
    print(f"  Usable issues           : {subset_result['total_usable']}")
    print(f"  Excluded (no type label): {subset_result['excluded_unlabelled']}")
    print(f"  Excluded (multi-type)   : {subset_result['excluded_multi_type']}")
    print()
    print("  Class counts:")
    class_items = sorted(
        subset_result["class_counts"].items(), key=lambda x: -x[1]
    )
    usable = subset_result["total_usable"]
    for label, cnt in class_items:
        pct = 100 * cnt / usable if usable else 0
        print(f"    {label:<25} {cnt:>4}  ({pct:.1f}%)")

    # --- Label scheme comparison ---
    print()
    print("  --- Label Scheme Comparison ---")
    type_labels_list = list(PROVISIONAL_TYPE_LABELS)

    # Scheme A: Original 5-class
    scheme_a = {lb: [lb] for lb in type_labels_list}
    stats_a = build_label_scheme_stats(issues, scheme_a)

    # Scheme B: Merge RFC into Enhancement; keep Build/CI
    scheme_b = {
        "Bug": ["Bug"],
        "Documentation": ["Documentation"],
        "Enhancement": ["New Feature", "RFC"],
        "Build / CI": ["Build / CI"],
    }
    stats_b = build_label_scheme_stats(issues, scheme_b)

    # Scheme C: 3-class core
    scheme_c = {
        "Bug": ["Bug"],
        "Documentation": ["Documentation"],
        "Enhancement": ["New Feature", "RFC", "Build / CI"],
    }
    stats_c = build_label_scheme_stats(issues, scheme_c)

    schemes = [
        ("A — Original 5-class", stats_a),
        ("B — 4-class (RFC merged into Enhancement)", stats_b),
        ("C — 3-class core", stats_c),
    ]
    for name, stats in schemes:
        ratio = stats["imbalance_ratio"]
        ratio_str = f"{ratio:.1f}:1" if ratio is not None else "N/A"
        print(f"\n  Scheme {name}")
        print(f"    Usable issues  : {stats['total_usable']}")
        print(f"    Min class count: {stats['min_class_count']}")
        print(f"    Imbalance ratio: {ratio_str}")
        print("    Classes:")
        for cls, cnt in sorted(
            stats["class_counts"].items(), key=lambda x: -x[1]
        ):
            pct = 100 * cnt / stats["total_usable"] if stats["total_usable"] else 0
            print(f"      {cls:<25} {cnt:>5}  ({pct:.1f}%)")

    print()
    print("  --- Top 20 Labels ---")
    for rank, (label, cnt) in enumerate(freq.most_common(20), 1):
        print(f"  {rank:>2}. {label:<40} {cnt}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading issues   : {args.input}")
    print(f"Loading metadata : {args.metadata}")
    print(f"Figures dir      : {figures_dir}")

    issues = load_issues(args.input)
    metadata = load_metadata(args.metadata)

    freq = compute_label_frequencies(issues)
    lpi = compute_labels_per_issue(issues)
    missing = compute_missing_data(issues)
    lengths = compute_text_lengths(issues)
    state_dist = compute_state_distribution(issues)
    monthly = compute_temporal_distribution(issues)
    leakage = detect_leakage(issues)
    subset_result = build_provisional_subset(issues)
    categories = categorize_labels_provisionally(freq)

    print("\nGenerating figures...")
    make_label_frequency_chart(freq, figures_dir)
    make_labels_per_issue_chart(lpi, figures_dir)
    make_type_distribution_chart(subset_result["class_counts"], figures_dir)
    make_temporal_chart(monthly, figures_dir)
    make_text_length_chart(lengths, figures_dir)
    make_cooccurrence_heatmap(issues, freq, figures_dir)
    # Stage 1C — additional charts
    make_yearly_chart(compute_yearly_distribution(issues), figures_dir)
    make_label_by_year_chart(issues, list(PROVISIONAL_TYPE_LABELS), figures_dir)

    print_summary(
        issues, metadata, freq, missing, lengths,
        state_dist, monthly, leakage, subset_result, categories,
    )


if __name__ == "__main__":
    main()
