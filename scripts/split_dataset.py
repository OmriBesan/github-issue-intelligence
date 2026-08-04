"""
scripts/split_dataset.py
========================
CLI script that takes the cleaned modelling dataset and generates train, validation,
and test splits using both temporal and stratified random strategies.

Usage
-----
.venv\\Scripts\\python scripts\\split_dataset.py [OPTIONS]

Options
-------
--input     PATH   Path to the cleaned JSONL dataset.
                   Default: data/processed/scikit-learn_issues_model.jsonl
--out-dir   DIR    Directory for processed output files.
                   Default: data/processed
--prefix    PREFIX Base filename prefix for output files.
                   Default: scikit-learn_issues_model
--train-frac FLOAT Fraction of data for training (default: 0.7)
--val-frac   FLOAT Fraction of data for validation (default: 0.15)
--seed      INT    Random seed for the stratified split (default: 42)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore

# Make the src package importable when called directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from issue_intelligence.data.preprocessing import save_csv, save_jsonl  # noqa: E402
from issue_intelligence.data.split import (  # noqa: E402
    stratified_random_split,
    temporal_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split the cleaned dataset into train/val/test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="data/processed/scikit-learn_issues_model.jsonl",
        metavar="PATH",
        help="Path to the cleaned JSONL file.",
    )
    p.add_argument(
        "--out-dir",
        default="data/processed",
        metavar="DIR",
        help="Output directory for split files.",
    )
    p.add_argument(
        "--prefix",
        default="scikit-learn_issues_model",
        metavar="PREFIX",
        help="Base filename prefix for output files.",
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of data for training.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of data for validation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the stratified random split.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    prefix = args.prefix

    logger.info("Loading processed records from: %s", input_path)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info("Loaded %d records.", len(records))

    # Check fraction valid
    if args.train_frac + args.val_frac > 1.0:
        logger.error("train-frac + val-frac cannot exceed 1.0")
        sys.exit(1)

    logger.info("Generating temporal split...")
    temp_train, temp_val, temp_test = temporal_split(
        records, train_frac=args.train_frac, val_frac=args.val_frac
    )

    logger.info("Generating stratified random split...")
    strat_train, strat_val, strat_test = stratified_random_split(
        records, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )

    # Save logic
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "temporal_train": temp_train,
        "temporal_val": temp_val,
        "temporal_test": temp_test,
        "stratified_train": strat_train,
        "stratified_val": strat_val,
        "stratified_test": strat_test,
    }

    for name, split_records in splits.items():
        base_name = f"{prefix}_{name}"
        csv_path = out_dir / f"{base_name}.csv"
        jsonl_path = out_dir / f"{base_name}.jsonl"

        logger.info(
            "Saving %s split to %s and %s (%d records)",
            name,
            csv_path.name,
            jsonl_path.name,
            len(split_records),
        )
        save_csv(split_records, csv_path)
        save_jsonl(split_records, jsonl_path)

    logger.info("Data splitting complete.")


if __name__ == "__main__":
    main()
