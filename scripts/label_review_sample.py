"""
scripts/label_review_sample.py
================================
Generate a human-review CSV for selected raw labels plus label-scheme
statistics to support the Stage 1D label mapping decision.

Usage
-----
.venv\\Scripts\\python scripts\\label_review_sample.py ^
    --input data\\raw\\scikit-learn_issues_combined.json ^
    --output reports\\label_mapping_review.csv ^
    --n 40 --seed 42

The script:
  1. Loads the combined issue dataset.
  2. For each of the four review labels (Enhancement, New Feature, RFC,
     Build / CI), samples up to --n issues with a fixed random seed.
  3. Saves the sample as a CSV with: issue_id, issue_number, created_at,
     title, body_preview, labels, url, sampled_for_label.
  4. Prints statistics: co-occurrence with Bug and Documentation, pairwise
     overlap, frequency by year, and common additional labels.
  5. Evaluates the candidate 3-class scheme:
       Bug       -> ["Bug"]
       Documentation -> ["Documentation"]
       Enhancement   -> ["Enhancement", "New Feature", "RFC"]
     and reports counts, imbalance, and leakage.

Run with --help for full argument descriptions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.audit import (  # noqa: E402
    build_label_scheme_stats,
    compute_label_distribution_by_year,
    detect_leakage,
    load_issues,
)
from issue_intelligence.data.combine import (  # noqa: E402
    build_cooccurrence_stats,
    sample_issues_by_label,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REVIEW_LABELS = ["Enhancement", "New Feature", "RFC", "Build / CI"]

# The candidate 3-class scheme — Build/CI excluded from type target
CANDIDATE_SCHEME = {
    "Bug": ["Bug"],
    "Documentation": ["Documentation"],
    "Enhancement": ["Enhancement", "New Feature", "RFC"],
}

# Reference labels for co-occurrence analysis
REFERENCE_LABELS = ["Bug", "Documentation"]


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label_review_sample.py",
        description=(
            "Generate a label-review CSV and statistics for human "
            "inspection of the Enhancement / New Feature / RFC / Build/CI "
            "label semantics."
        ),
    )
    parser.add_argument(
        "--input",
        default="data/raw/scikit-learn_issues_combined.json",
        help=(
            "Path to the combined issues JSON file "
            "(default: data/raw/scikit-learn_issues_combined.json)."
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/label_mapping_review.csv",
        help=(
            "Output path for the review CSV file "
            "(default: reports/label_mapping_review.csv)."
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=40,
        metavar="N",
        help="Maximum sample size per label (default: 40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    return parser


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

_BODY_PREVIEW_CHARS = 300


def _body_preview(body: str | None) -> str:
    if not body:
        return ""
    text = str(body).strip().replace("\r\n", " ").replace("\n", " ")
    if len(text) > _BODY_PREVIEW_CHARS:
        return text[:_BODY_PREVIEW_CHARS] + "..."
    return text


def write_review_csv(
    issues_by_label: dict[str, list[dict]],
    output_path: Path,
) -> int:
    """Write the review CSV and return the total number of rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sampled_for_label",
        "issue_id",
        "issue_number",
        "created_at",
        "title",
        "body_preview",
        "labels",
        "url",
    ]
    total_rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for label_name, issues in issues_by_label.items():
            for issue in issues:
                writer.writerow(
                    {
                        "sampled_for_label": label_name,
                        "issue_id": issue.get("id"),
                        "issue_number": issue.get("number"),
                        "created_at": issue.get("created_at"),
                        "title": issue.get("title", ""),
                        "body_preview": _body_preview(issue.get("body")),
                        "labels": "; ".join(issue.get("labels") or []),
                        "url": issue.get("html_url", ""),
                    }
                )
                total_rows += 1
    return total_rows


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------


def print_statistics(
    issues: list[dict],
    cooc_stats: dict,
) -> None:
    print("\n" + "=" * 65)
    print("  LABEL REVIEW STATISTICS")
    print("=" * 65)

    # Candidate counts
    print("\n  --- Candidate Label Counts ---")
    for label in REVIEW_LABELS:
        cnt = sum(1 for i in issues if label in (i.get("labels") or []))
        print(f"    {label:<20}: {cnt:>5} issues")

    # Co-occurrence with Bug and Documentation
    print("\n  --- Co-occurrence with Bug / Documentation ---")
    print(f"  {'Label':<22} | {'+ Bug':>7} | {'+ Documentation':>16} | {'Total':>6}")
    print(f"  {'-' * 22}-+-{'-' * 7}-+-{'-' * 16}-+-{'-' * 6}")
    for label in REVIEW_LABELS:
        with_target = cooc_stats["issues_with_target"].get(label, 0)
        with_bug = cooc_stats["cooccurrence"].get(label, {}).get("Bug", 0)
        with_doc = cooc_stats["cooccurrence"].get(label, {}).get("Documentation", 0)
        print(f"  {label:<22} | {with_bug:>7} | {with_doc:>16} | {with_target:>6}")

    # Pairwise overlap between the four labels
    print("\n  --- Pairwise Overlap (issues carrying BOTH labels) ---")
    for i, a in enumerate(REVIEW_LABELS):
        for b in REVIEW_LABELS[i + 1 :]:
            count = sum(
                1
                for iss in issues
                if a in (iss.get("labels") or []) and b in (iss.get("labels") or [])
            )
            print(f"    {a:<22} + {b:<22}: {count}")

    # Frequency by year
    print("\n  --- Frequency by Year ---")
    all_four_candidates = [
        i for i in issues if any(lb in (i.get("labels") or []) for lb in REVIEW_LABELS)
    ]
    by_year = compute_label_distribution_by_year(all_four_candidates, REVIEW_LABELS)
    years = sorted(by_year.keys())
    header = f"  {'Year':<6}" + "".join(f" | {lb[:10]:>10}" for lb in REVIEW_LABELS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for y in years:
        row = f"  {y:<6}"
        for lb in REVIEW_LABELS:
            row += f" | {by_year[y].get(lb, 0):>10}"
        print(row)

    # Common additional labels per target label
    print("\n  --- Common Additional Labels (top 10 per label) ---")
    for label in REVIEW_LABELS:
        label_issues = [i for i in issues if label in (i.get("labels") or [])]
        other_labels: Counter = Counter()
        for iss in label_issues:
            for lb in iss.get("labels") or []:
                if lb != label:
                    other_labels[lb] += 1
        top = other_labels.most_common(10)
        print(f"\n    [{label}]")
        for lb, cnt in top:
            print(f"      {lb:<35}: {cnt}")


def print_candidate_scheme_results(
    issues: list[dict],
) -> None:
    """Evaluate the candidate 3-class scheme and print results."""
    print("\n" + "=" * 65)
    print("  CANDIDATE SCHEME EVALUATION")
    print("=" * 65)
    print("\n  Mapping:")
    for cls, raw_labels in CANDIDATE_SCHEME.items():
        print(f"    {cls:<22}: {raw_labels}")
    print("\n  Build / CI is excluded from the type target.\n")

    stats = build_label_scheme_stats(issues, CANDIDATE_SCHEME)
    usable = stats["total_usable"]
    print(f"  Total usable issues     : {usable}")
    print(f"  Excluded (no type lbl)  : {stats['excluded_unlabelled']}")
    print(f"  Excluded (multi-class)  : {stats['excluded_multi_class']}")
    ratio = stats["imbalance_ratio"]
    ratio_str = f"{ratio:.2f}:1" if ratio is not None else "N/A"
    print(f"  Imbalance ratio         : {ratio_str}")
    print(f"  Min class count         : {stats['min_class_count']}")
    print()
    print("  Class counts:")
    for cls, cnt in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        pct = 100 * cnt / usable if usable else 0
        print(f"    {cls:<25} {cnt:>5}  ({pct:.1f}%)")

    # Year distribution per class
    print("\n  Year distribution per class:")
    for cls, raw_labels in CANDIDATE_SCHEME.items():
        class_issues = [
            i
            for i in issues
            if len(
                [
                    lb
                    for lb in (i.get("labels") or [])
                    if any(lb in CANDIDATE_SCHEME[c] for c in CANDIDATE_SCHEME)
                ]
            )
            == 1
            and any(lb in raw_labels for lb in (i.get("labels") or []))
        ]
        by_year = Counter(
            i["created_at"][:4] for i in class_issues if i.get("created_at")
        )
        years_str = ", ".join(f"{y}:{c}" for y, c in sorted(by_year.items()))
        print(f"    {cls:<22}: {years_str}")

    # Leakage
    print("\n  Leakage check (on candidate usable subset):")
    # Build usable subset
    usable_issues = []
    for i in issues:
        classes = {
            cls
            for cls, raw_labels in CANDIDATE_SCHEME.items()
            if any(lb in raw_labels for lb in (i.get("labels") or []))
        }
        if len(classes) == 1:
            usable_issues.append(i)
    leakage = detect_leakage(usable_issues)
    print(f"    Titles with type prefix : {len(leakage['prefix_matches'])}")
    print(f"    Titles with label name  : {len(leakage['label_in_title'])}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    print(f"Loading issues: {input_path}")
    issues = load_issues(input_path)
    print(f"  Loaded {len(issues)} issues")

    # Sample for each review label
    print(f"\nSampling up to {args.n} issues per label (seed={args.seed})...")
    issues_by_label: dict[str, list[dict]] = {}
    for label in REVIEW_LABELS:
        sample = sample_issues_by_label(issues, label, n=args.n, seed=args.seed)
        n_label = sum(1 for i in issues if label in (i.get("labels") or []))
        issues_by_label[label] = sample
        print(f"  {label:<22}: {len(sample):>3} sampled  (from {n_label} total)")

    # Write CSV
    rows = write_review_csv(issues_by_label, output_path)
    print(f"\n[OK] Review CSV saved: {output_path}  ({rows} rows)")

    # Co-occurrence statistics
    cooc_stats = build_cooccurrence_stats(
        issues,
        target_labels=REVIEW_LABELS,
        reference_labels=REFERENCE_LABELS,
    )

    # Print statistics
    print_statistics(issues, cooc_stats)

    # Candidate scheme evaluation
    print_candidate_scheme_results(issues)

    print(f"\n[OK] Done.  Review CSV: {output_path}")


if __name__ == "__main__":
    main()
