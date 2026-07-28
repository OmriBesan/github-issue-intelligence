"""
scripts/collect_issues.py
==========================
Command-line script that collects GitHub issues and prints basic statistics.

Usage
-----
# Most-recent 500 issues (Stage 1A style)
.venv\\Scripts\\python scripts\\collect_issues.py \\
    --owner scikit-learn \\
    --repo  scikit-learn \\
    --state all \\
    --max-issues 500 \\
    --output data\\raw\\scikit-learn_issues_sample.json

# Full historical collection — oldest first (Stage 1C)
.venv\\Scripts\\python scripts\\collect_issues.py \\
    --owner     scikit-learn \\
    --repo      scikit-learn \\
    --state     all \\
    --sort      created \\
    --direction asc \\
    --max-issues 5000 \\
    --output data\\raw\\scikit-learn_issues_history.json

The script reads GITHUB_TOKEN from a .env file (or the environment).
The token is never printed or written to any output file.

Run with --help for full argument descriptions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

# Ensure the src/ package is importable when the script is run directly
# without installing the package in editable mode.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.github_client import (  # noqa: E402
    GitHubAPIError,
    GitHubIssueCollector,
    MissingTokenError,
)

# ---------------------------------------------------------------------------
# Logging — shows INFO-level messages on stdout during collection
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_issues.py",
        description=(
            "Collect GitHub issues from a public repository "
            "and save them as JSON. Pull requests are excluded automatically."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Most-recent 500 issues (Stage 1A)
  python scripts/collect_issues.py \\
      --owner scikit-learn --repo scikit-learn \\
      --state all --max-issues 500 \\
      --output data/raw/scikit-learn_issues_sample.json

  # Historical 5000 issues — oldest first (Stage 1C)
  python scripts/collect_issues.py \\
      --owner scikit-learn --repo scikit-learn \\
      --state all --sort created --direction asc --max-issues 5000 \\
      --output data/raw/scikit-learn_issues_history.json

Environment:
  GITHUB_TOKEN  GitHub Personal Access Token (classic, public_repo scope).
                Set in .env or as an environment variable.
        """,
    )
    parser.add_argument(
        "--owner",
        required=True,
        help="GitHub repository owner, e.g. 'scikit-learn'.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository name, e.g. 'scikit-learn'.",
    )
    parser.add_argument(
        "--state",
        choices=["open", "closed", "all"],
        default="all",
        help="Issue state to collect (default: all).",
    )
    parser.add_argument(
        "--sort",
        choices=["created", "updated", "comments"],
        default="created",
        help=(
            "API sort field "
            "(default: created)."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=["asc", "desc"],
        default="desc",
        help=(
            "Sort direction: 'desc' (newest first, default) or "
            "'asc' (oldest first, use for historical coverage)."
        ),
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=500,
        metavar="N",
        help="Maximum number of regular issues to collect (default: 500).",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        metavar="N",
        help=(
            "API page number to start from (default: 1). "
            "Use > 1 to resume a previous collection. "
            "Combine outputs using combine_datasets.py."
        ),
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="ISO8601",
        help=(
            "ISO-8601 timestamp (e.g. '2018-05-29T00:00:00Z'). "
            "Only return issues with updated_at >= since. "
            "Preferred over --start-page for large repositories."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/raw/scikit-learn_issues_sample.json",
        help=(
            "Path for the output JSON file "
            "(default: data/raw/scikit-learn_issues_sample.json)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="SECONDS",
        help="Request timeout in seconds (default: 30).",
    )
    return parser


# ---------------------------------------------------------------------------
# Statistics printer
# ---------------------------------------------------------------------------

def print_statistics(issues: list[dict], metadata: dict) -> None:
    """Print a concise summary of the collected dataset."""
    total = len(issues)
    if total == 0:
        print("\n[WARN] No issues collected.")
        return

    with_labels = sum(1 for i in issues if i.get("labels"))
    without_labels = total - with_labels
    missing_body = sum(
        1 for i in issues if not i.get("body") or not str(i["body"]).strip()
    )

    # Date range
    dates = [i["created_at"] for i in issues if i.get("created_at")]
    earliest = min(dates) if dates else "N/A"
    latest = max(dates) if dates else "N/A"

    # Label frequencies
    all_labels: list[str] = []
    for issue in issues:
        all_labels.extend(issue.get("labels") or [])
    label_counts = Counter(all_labels)
    top_labels = label_counts.most_common(20)

    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)
    print(f"  Repository          : {metadata.get('repository')}")
    sort_dir = f"{metadata.get('sort')} / {metadata.get('direction')}"
    print(f"  Sort / direction    : {sort_dir}")
    print(f"  State filter        : {metadata.get('state_filter')}")
    print(f"  API items inspected : {metadata.get('api_items_inspected')}")
    print(f"  Pull requests excl. : {metadata.get('pull_requests_excluded')}")
    print(f"  Duplicates skipped  : {metadata.get('duplicates_skipped', 0)}")
    print(f"  Regular issues saved: {total}")
    print(f"  Authenticated       : {metadata.get('authenticated')}")
    print(f"  Rate limit remaining: {metadata.get('remaining_rate_limit')}")
    print()
    pct_with = 100 * with_labels / total
    pct_without = 100 * without_labels / total
    pct_missing = 100 * missing_body / total
    print(f"  With labels         : {with_labels}  ({pct_with:.1f}%)")
    print(f"  Without labels      : {without_labels}  ({pct_without:.1f}%)")
    print(f"  Missing/empty body  : {missing_body}  ({pct_missing:.1f}%)")
    print()
    print(f"  Earliest issue      : {earliest}")
    print(f"  Latest issue        : {latest}")
    print()
    print(f"  Top {len(top_labels)} labels:")
    for rank, (label, count) in enumerate(top_labels, start=1):
        print(f"    {rank:>2}. {label:<40} {count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = Path(args.output)

    collector = GitHubIssueCollector(timeout=args.timeout)

    try:
        issues, metadata = collector.collect(
            owner=args.owner,
            repo=args.repo,
            state=args.state,
            sort=args.sort,
            direction=args.direction,
            start_page=args.start_page,
            since=args.since,
            max_issues=args.max_issues,
            output_path=output_path,
        )
    except MissingTokenError as exc:
        logger.error("Token error: %s", exc)
        sys.exit(1)
    except GitHubAPIError as exc:
        logger.error("GitHub API error: %s", exc)
        sys.exit(1)

    print_statistics(issues, metadata)

    print(f"\n[OK] Issues saved to  : {output_path}")
    stem = output_path.stem
    meta_path = output_path.with_name(stem + "_metadata" + output_path.suffix)
    print(f"[OK] Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
