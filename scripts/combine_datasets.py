"""
scripts/combine_datasets.py
============================
CLI script: merge multiple raw issue JSON files into one deduplicated
combined dataset.

Usage
-----
.venv\\Scripts\\python scripts\\combine_datasets.py ^
    --inputs data\\raw\\scikit-learn_issues_history.json ^
             data\\raw\\scikit-learn_issues_additional.json ^
             data\\raw\\scikit-learn_issues_sample.json ^
    --output data\\raw\\scikit-learn_issues_combined.json

The script:
  1. Loads all input files in order.
  2. Deduplicates by GitHub issue ID (first occurrence wins).
  3. Sorts the result by created_at ascending.
  4. Saves the combined issues JSON and a combined metadata JSON.
  5. Prints a concise summary including issues per year and gap analysis.

Run with --help for full argument descriptions.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

# Make the src package importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.combine import (  # noqa: E402
    find_yearly_gaps,
    load_and_combine,
    save_combined,
)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="combine_datasets.py",
        description=(
            "Merge multiple raw GitHub issue JSON files into a single "
            "deduplicated dataset."
        ),
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Two or more raw issue JSON files to combine.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/scikit-learn_issues_combined.json",
        help=(
            "Output path for the combined JSON file "
            "(default: data/raw/scikit-learn_issues_combined.json)."
        ),
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2011,
        metavar="YEAR",
        help="First year to check for coverage gaps (default: 2011).",
    )
    return parser


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(
    issues: list[dict],
    combine_stats: dict,
    gap_info: dict,
    output_path: Path,
) -> None:
    total = len(issues)
    dates = [i["created_at"] for i in issues if i.get("created_at")]
    earliest = min(dates) if dates else "N/A"
    latest = max(dates) if dates else "N/A"

    print("\n" + "=" * 65)
    print("  COMBINED DATASET SUMMARY")
    print("=" * 65)
    print(f"  Files combined          : {combine_stats['files_loaded']}")
    print(f"  Duplicates removed      : {combine_stats['duplicates_found']}")
    print(f"  Unique issues saved     : {total}")
    print(f"  Earliest issue date     : {earliest}")
    print(f"  Latest issue date       : {latest}")
    print()
    print("  --- Issues per source file ---")
    for path, count in combine_stats["issues_per_file"].items():
        print(f"    {count:>5}  {path}")
    print()
    print("  --- Issues per year ---")
    for year in sorted(gap_info["yearly_counts"].keys()):
        count = gap_info["yearly_counts"][year]
        bar_len = int(35 * count / max(gap_info["yearly_counts"].values()))
        print(f"    {year}: {count:>5}  {'|' * bar_len}")
    print()
    if gap_info["gaps"]:
        print(f"  [WARN] Coverage gaps detected: {', '.join(gap_info['gaps'])}")
    else:
        print(f"  [OK] No gaps from {gap_info['min_year']} to {gap_info['max_year']}.")
    print()
    print(f"  Output: {output_path}")
    meta_path = output_path.with_name(output_path.stem + "_metadata.json")
    print(f"  Metadata: {meta_path}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            print(f"[ERROR] Input file not found: {p}", file=sys.stderr)
            sys.exit(1)

    output_path = Path(args.output)

    print(f"Loading {len(input_paths)} input file(s)...")
    for p in input_paths:
        print(f"  {p}")

    issues, combine_stats = load_and_combine(input_paths, sort_by_date=True)

    # Gap analysis
    gap_info = find_yearly_gaps(issues, min_year=args.min_year)

    # Build metadata
    dates = [i["created_at"] for i in issues if i.get("created_at")]
    metadata = {
        "type": "combined",
        "combination_timestamp": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "source_files": [str(p) for p in input_paths],
        "files_loaded": combine_stats["files_loaded"],
        "duplicates_found": combine_stats["duplicates_found"],
        "total_unique_issues": combine_stats["total_issues"],
        "issues_per_file": combine_stats["issues_per_file"],
        "earliest_created_at": min(dates) if dates else None,
        "latest_created_at": max(dates) if dates else None,
        "yearly_counts": gap_info["yearly_counts"],
        "coverage_gaps": gap_info["gaps"],
    }

    save_combined(issues, metadata, output_path)

    print_summary(issues, combine_stats, gap_info, output_path)


if __name__ == "__main__":
    main()
