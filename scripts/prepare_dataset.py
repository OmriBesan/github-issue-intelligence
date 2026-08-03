"""
scripts/prepare_dataset.py
===========================
CLI script that builds the final cleaned modelling dataset from the combined
raw scikit-learn issues file.

Usage
-----
.venv\\Scripts\\python scripts\\prepare_dataset.py [OPTIONS]

Options
-------
--input   PATH   Path to the combined raw issues JSON.
                 Default: data/raw/scikit-learn_issues_combined.json
--out-dir DIR    Directory for processed output files.
                 Default: data/processed
--prefix PREFIX  Base filename prefix for output files.
                 Default: scikit-learn_issues_model
--sample  N      Print N random examples per class (default: 3, 0 to skip).
--seed    INT    Random seed for the example sampler (default: 42).

Output files
------------
<out-dir>/<prefix>.csv
<out-dir>/<prefix>.jsonl
<out-dir>/<prefix>_metadata.json
"""  # noqa: D400 (script docstring)

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# Make the src package importable when called directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.audit import load_issues  # noqa: E402
from issue_intelligence.data.preprocessing import (  # noqa: E402
    LABEL_SCHEME,
    build_dataset,
    build_metadata,
    save_csv,
    save_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prepare the cleaned modelling dataset for scikit-learn issues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="data/raw/scikit-learn_issues_combined.json",
        metavar="PATH",
        help="Path to the combined raw issues JSON file.",
    )
    p.add_argument(
        "--out-dir",
        default="data/processed",
        metavar="DIR",
        help="Output directory for processed files.",
    )
    p.add_argument(
        "--prefix",
        default="scikit-learn_issues_model",
        metavar="PREFIX",
        help="Base filename prefix for output files.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=3,
        metavar="N",
        help="Number of random examples per class to display (0 = skip).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the example sampler.",
    )
    return p


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 68


def _print_section(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_example(
    idx: int,
    rec: dict,
    show_body_chars: int = 200,
) -> None:
    prefix_flag = "  [PREFIX STRIPPED]" if rec.get("_prefix_removed") else ""
    print(f"\n  Example {idx + 1} -- #{rec['issue_number']}{prefix_flag}")
    print(f"    target       : {rec['target']}")
    print(f"    raw_title    : {rec['raw_title'][:80]!r}")
    print(f"    clean_title  : {rec['clean_title'][:80]!r}")
    body_preview = rec["clean_body"][:show_body_chars].replace("\n", " \\n ")
    if len(rec["clean_body"]) > show_body_chars:
        body_preview += "..."
    print(f"    clean_body   : {body_preview!r}")
    print(f"    labels       : {rec['original_labels']}")
    print(f"    created_at   : {rec['created_at']}")


def _verify_csv_jsonl_equivalence(
    csv_path: Path,
    jsonl_path: Path,
    records: list[dict],
) -> bool:
    """
    Verify that the CSV and JSONL files contain the same records.

    Checks:
    1. Line counts match.
    2. issue_id order matches.
    3. target values match.

    Returns True if all checks pass.
    """
    import csv as _csv

    ok = True

    # CSV rows
    with csv_path.open(encoding="utf-8") as fh:
        csv_rows = list(_csv.DictReader(fh))
    # JSONL rows
    jsonl_rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    ]

    if len(csv_rows) != len(jsonl_rows):
        logger.error(
            "EQUIVALENCE FAIL: CSV has %d rows, JSONL has %d rows",
            len(csv_rows),
            len(jsonl_rows),
        )
        ok = False
    else:
        logger.info("CSV/JSONL row count match: %d rows each.", len(csv_rows))

    if ok:
        csv_ids = [r["issue_id"] for r in csv_rows]
        jsonl_ids = [str(r["issue_id"]) for r in jsonl_rows]
        if csv_ids != jsonl_ids:
            logger.error(
                "EQUIVALENCE FAIL: issue_id order differs between CSV and JSONL."
            )
            ok = False
        else:
            logger.info("CSV/JSONL issue_id order: MATCH.")

    if ok:
        csv_targets = [r["target"] for r in csv_rows]
        jsonl_targets = [r["target"] for r in jsonl_rows]
        if csv_targets != jsonl_targets:
            logger.error(
                "EQUIVALENCE FAIL: target values differ between CSV and JSONL."
            )
            ok = False
        else:
            logger.info("CSV/JSONL target values: MATCH.")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    prefix = args.prefix

    csv_path = out_dir / f"{prefix}.csv"
    jsonl_path = out_dir / f"{prefix}.jsonl"
    meta_path = out_dir / f"{prefix}_metadata.json"

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    logger.info("Loading raw issues from: %s", input_path)
    try:
        issues = load_issues(input_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot load input: %s", exc)
        sys.exit(1)

    logger.info("Loaded %d raw records.", len(issues))

    # ------------------------------------------------------------------
    # 2. Build dataset
    # ------------------------------------------------------------------
    logger.info("Applying label mapping and text preprocessing…")
    records, stats = build_dataset(issues)

    usable = stats["usable"]
    if usable == 0:
        logger.error(
            "No usable records after preprocessing. "
            "Check the input file and label mapping."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    logger.info("Saving CSV  : %s", csv_path)
    save_csv(records, csv_path)

    logger.info("Saving JSONL: %s", jsonl_path)
    save_jsonl(records, jsonl_path)

    # Metadata needs the output files to already exist (for hashing).
    metadata = build_metadata(stats, records, input_path, csv_path, jsonl_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    logger.info("Saving meta : %s", meta_path)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    _print_section("PREPROCESSING SUMMARY")

    print(f"\n  Input file        : {input_path}")
    print(f"  Total raw records : {stats['total_raw']:,}")
    print()
    print(f"  Excluded - no target class    : {stats['excluded_no_target']:>6,}")
    print(f"  Excluded - multi-target class : {stats['excluded_multi_target']:>6,}")
    print(f"  Excluded - duplicate ID       : {stats['excluded_duplicate']:>6,}")
    print(f"  Excluded - empty text         : {stats['excluded_empty_text']:>6,}")
    print("  " + "-" * 43)
    print(f"  Usable records                : {usable:>6,}")
    print()

    class_counts = stats["class_counts"]
    print("  Class counts:")
    for cls, cnt in class_counts.items():
        pct = 100 * cnt / usable
        bar = "#" * int(pct / 2)
        print(f"    {cls:<18}: {cnt:>5,}  ({pct:.1f}%)  {bar}")
    print()
    print(f"  Imbalance ratio       : {stats['imbalance_ratio']:.3f}:1")
    print(
        f"  Title prefixes removed: {stats['prefix_removals']:,}"
        f"  ({100 * stats['prefix_removals'] / usable:.1f}% of usable)"
    )
    print()

    dates = [r["created_at"] for r in records if r.get("created_at")]
    if dates:
        print(f"  Earliest issue : {min(dates)}")
        print(f"  Latest issue   : {max(dates)}")

    # ------------------------------------------------------------------
    # 5. Verify CSV/JSONL equivalence
    # ------------------------------------------------------------------
    _print_section("CSV / JSONL EQUIVALENCE CHECK")
    ok = _verify_csv_jsonl_equivalence(csv_path, jsonl_path, records)
    if ok:
        print("\n  [OK] CSV and JSONL are equivalent.")
    else:
        print("\n  [FAIL] CSV and JSONL differ — inspect the output files.")

    # ------------------------------------------------------------------
    # 6. Examples per class
    # ------------------------------------------------------------------
    if args.sample > 0:
        rng = random.Random(args.seed)
        _print_section("RANDOM EXAMPLES PER CLASS")
        for cls in LABEL_SCHEME:
            class_records = [r for r in records if r["target"] == cls]
            sample = rng.sample(class_records, min(args.sample, len(class_records)))
            print(f"\n  --- {cls} ({len(class_records)} total) ---")
            for i, rec in enumerate(sample):
                _print_example(i, rec)

    # ------------------------------------------------------------------
    # 7. Prefix removal examples
    # ------------------------------------------------------------------
    prefix_records = [r for r in records if r.get("_prefix_removed")]
    _print_section(f"PREFIX REMOVAL EXAMPLES  ({len(prefix_records)} total)")
    rng2 = random.Random(args.seed)
    sample_prefixes = rng2.sample(prefix_records, min(8, len(prefix_records)))
    for i, rec in enumerate(sample_prefixes):
        print(f"\n  Example {i + 1} — #{rec['issue_number']}")
        print(f"    raw_title  : {rec['raw_title'][:80]!r}")
        print(f"    clean_title: {rec['clean_title'][:80]!r}")

    # ------------------------------------------------------------------
    # 8. Done
    # ------------------------------------------------------------------
    _print_section("OUTPUT FILES")
    print(f"\n  {csv_path}")
    print(f"  {jsonl_path}")
    print(f"  {meta_path}")
    print()


if __name__ == "__main__":
    main()
